import os
import subprocess
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import alpha_clip
import loralib as lora
from utils import concat_all_gather, is_dist_avail_and_initialized
from scheduler import cosine_lr
from dataset.imagenet_s_test import Imagenet_S
from dataset.mask_image_test import COCO_Masked_Test
from dataset.alpha_grit import Alpha_GRIT
from datetime import datetime

simple_templates = ['a photo of a {}.']

class CLIP_Clean_Train():
    def __init__(self, local_rank=3, lr=4e-5, weight_decay=0.02, log_scale=4.6052, lora_rank=-1, common_pair=0.0,
                 para_gamma=0.01, exp_name="auto", warmup_length=200, epoch_num=1, subnum=10000, distributed=False):
        self.local_rank = local_rank
        self.distributed = distributed
        self.model = self.load_model(lora_rank)
        torch.cuda.set_device(device=f'cuda:{local_rank}')
        self.model = self.model.float().cuda()
        self.batch_size = 12
        self.num_epoch = epoch_num
        self.lr = lr
        self.subnum = subnum
        self.logdir = self.get_logdir(exp_name, lr, weight_decay, warmup_length, log_scale, lora_rank, common_pair, para_gamma, epoch_num, subnum)
        self.ckptdir = os.path.join(self.logdir, "ckpt/")
        os.makedirs(self.ckptdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)
        # self.model.visual = torch.nn.parallel.DistributedDataParallel(self.model.visual, device_ids=[local_rank],
        #                                                               output_device=local_rank,
        #                                                               find_unused_parameters=True)
        # logit scale
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * log_scale)
        self.optimizer = self.configure_optimizer(lora_rank, para_gamma, lr)
        self.para_gamma = para_gamma
        self.scaler = torch.cuda.amp.GradScaler()

    def load_model(self, lora_rank: int):
        if lora_rank == -1:
            model, _ = alpha_clip.load("ViT-L/14@336px", device='cpu', lora_adapt=False, rank=-1)
        else:
            model, _ = alpha_clip.load("ViT-L/14", device='cpu', lora_adapt=True, rank=lora_rank)
        return model

    def get_logdir(self, exp_name: str, lr: float, weight_decay: float, warmup_length: int, log_scale: float,
                   lora_rank: int, common_pair: float, para_gamma: float, epoch_num: int, subnum: int) -> str:
        date_str = datetime.now().strftime("%Y%m%d")
        base_logdir = f"log/{date_str}_grit_1m/lr={lr}_wd={weight_decay}_wl={warmup_length}_logs={log_scale}_L14_336_lora={lora_rank}_cp={common_pair}_para_gamma={para_gamma}_e{epoch_num}_16xb_subnum={subnum}"
        logdir = base_logdir
        suffix = 1
        while os.path.exists(logdir):
            logdir = f"{base_logdir}_{suffix}"
            suffix += 1
        return logdir

    def configure_optimizer(self, lora_rank: int, para_gamma: float, lr: float):
        conv_opt_paras = []
        other_opt_paras = []
        if lora_rank != -1:
            lora.mark_only_lora_as_trainable(self.model)
            for k, v in self.model.named_parameters():
                if v.requires_grad:
                    other_opt_paras.append(v)
                elif "conv1_alpha" in k:
                    v.requires_grad_(True)
                    conv_opt_paras.append(v)
        else:
            for k, v in self.model.named_parameters():
                v.requires_grad_(False)
            for k, v in self.model.visual.named_parameters():
                v.requires_grad_(True)
                if "conv1_alpha" in k:
                    conv_opt_paras.append(v)
                else:
                    other_opt_paras.append(v)
        optimizer = optim.AdamW(
            [
                {"params": conv_opt_paras, "lr": lr},
                {"params": other_opt_paras, "lr": lr * para_gamma}
            ],
        )
        return optimizer

    @torch.no_grad()
    def zeroshot_classifier(self, classnames, templates, local_rank=0):
        zeroshot_weights = []
        for classname in tqdm(classnames, disable=(dist.get_rank() != 0 if dist.is_initialized() else False)):
            texts = [template.format(classname) for template in templates]
            texts = alpha_clip.tokenize(texts).cuda()
            class_embeddings = self.model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def inference(self, images, masks, texts):
        image_features = self.model.visual(images, masks)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_feat_all = concat_all_gather(image_features)
        text_feat_all = concat_all_gather(text_features)

        sim_i2t = torch.matmul(image_features, text_feat_all.T)
        sim_t2i = torch.matmul(image_feat_all, text_features.T).T
        sim_i2t = self.model.logit_scale.exp() * sim_i2t
        sim_t2i = self.model.logit_scale.exp() * sim_t2i
        rank = dist.get_rank() if is_dist_avail_and_initialized() else 0
        bs = images.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(images.device)
        loss_itc = (F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)) / 2
        return loss_itc

    def train_epoch(self, dataloader, test_loaders, epoch, start_iter=0, amp=False, eval_ratio=0.1):
        running_loss = 0.0
        num_batches_per_epoch = len(dataloader)
        eval_step = int(num_batches_per_epoch * eval_ratio)
        for i, (images, masks, texts) in enumerate(tqdm(dataloader, disable=(dist.get_rank() != 0 if dist.is_initialized() else False))):
            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue
            self.optimizer.zero_grad()
            self.scheduler(step)
            images = images.cuda()
            masks = masks.cuda()
            texts = alpha_clip.tokenize(texts).cuda()
            if amp:
                with torch.cuda.amp.autocast():
                    loss = self.inference(images, masks, texts)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.inference(images, masks, texts)
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item()
            batch_num = i + 1
            if (i + 1) % 500 == 0:
                self.log_metrics(running_loss, 500, step, test_loaders)
                running_loss = 0.0
            if eval_step > 0 and (i + 1) % eval_step == 0:
                self.evaluate(step, test_loaders)
        return running_loss / batch_num

    def log_metrics(self, running_loss, interval, step, test_loaders):
        loss = running_loss / interval
        loss = torch.tensor(loss).cuda()
        if dist.is_initialized():
            dist.all_reduce(loss)
            loss = loss.item() / torch.distributed.get_world_size()
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.writer.add_scalar("hyper/lr", self.optimizer.param_groups[0]['lr'], step)
            self.writer.add_scalar("logit_scale/train", self.model.logit_scale.item(), step)
            self.writer.add_scalar("Loss/train", loss, step)
            print("=====================================")
            print(f"train lr (alpha conv) step {step}: {self.optimizer.param_groups[0]['lr']}")
            print(f"train lr (other layer) step {step}: {self.optimizer.param_groups[1]['lr']}")
            print(f"train logit_scale step {step}: {self.model.logit_scale.item()}")
            print(f"train loss step {step}: {loss}")
            print("=====================================")
            if step % 500 == 0 and step != 0:
                torch.save(self.model.visual.state_dict(), self.ckptdir + f'iter_{step}.pth')

    @torch.no_grad()
    def test_epoch(self, dataloader, desc="Evaluating"):
        temp_corr_dict = dict()
        for images, masks, target in tqdm(dataloader, disable=(dist.get_rank() != 0 if dist.is_initialized() else False), desc=desc, leave=False, mininterval=20.0):
            images, masks, target = images.cuda(), masks.cuda(), target.cuda()
            image_features = self.model.visual(images, masks)
            # image_features = self.model.visual(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score = torch.matmul(image_features, self.text_embeddings)
            pred = score.topk(1, dim=1)[1].squeeze(dim=1)
            pred_5 = score.topk(5, dim=1)[1].squeeze(dim=1)
            for i in range(target.shape[0]):
                target_item = target[i].item()
                if target_item not in temp_corr_dict:
                    temp_corr_dict[target_item] = [0, 0, 0]
                temp_corr_dict[target_item][0] += 1
                if target_item == pred[i].item():
                    temp_corr_dict[target_item][1] += 1
                if target_item in pred_5[i].tolist():
                    temp_corr_dict[target_item][2] += 1
        return temp_corr_dict

    def evaluate(self, step, test_loaders):
        self.model.visual.eval()
        for test_name, test_loader in test_loaders.items():
            tqdm.write(f"Zeroshot Classifier Evaluating {test_name} at step {step}")
            self.text_embeddings = self.zeroshot_classifier(test_loader.dataset.classes, simple_templates, self.local_rank)
            temp_corr_dict = self.test_epoch(test_loader, desc=f"Evaluating {test_name}")
            output = self.gather_output(temp_corr_dict)
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.log_test_results(test_name, step, output)
        self.model.visual.train()

    def gather_output(self, temp_corr_dict):
        if dist.is_initialized():
            output = [None] * dist.get_world_size()
            dist.all_gather_object(output, temp_corr_dict)
        else:
            output = [temp_corr_dict]
        return output

    def log_test_results(self, test_name, step, output):
        final_dict = self.aggregate_output(output)
        acc1, acc5 = self.calculate_accuracy(final_dict)
        print("=====================================")
        print(f"test {test_name} acc-1 step {step}: {acc1}")
        print(f"test {test_name} acc-5 step {step}: {acc5}")
        print("=====================================")
        self.writer.add_scalar(f"{test_name}_Acc1/test", acc1, step)
        self.writer.add_scalar(f"{test_name}_Acc5/test", acc5, step)

    def aggregate_output(self, output):
        final_dict = dict()
        for dic in output:
            for k, v in dic.items():
                if k not in final_dict.keys():
                    final_dict[k] = v
                else:
                    final_dict[k][0] += v[0]
                    final_dict[k][1] += v[1]
                    final_dict[k][2] += v[2]
        return final_dict

    def calculate_accuracy(self, final_dict):
        acc1, acc5, num_class = 0.0, 0.0, 0
        for v in final_dict.values():
            acc1 += v[1] / v[0]
            acc5 += v[2] / v[0]
            num_class += 1
        acc1 /= num_class
        acc5 /= num_class
        return acc1, acc5

    def test(self):
        self.model.visual.eval()
        testset = Imagenet_S()
        self.text_embeddings = self.zeroshot_classifier(testset.classes, simple_templates, self.local_rank)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
        with torch.no_grad():
            temp_corr_dict = self.test_epoch(testloader, desc="Testing")
            output = self.gather_output(temp_corr_dict)
            if not dist.is_initialized() or self.local_rank == 0:
                final_dict = self.aggregate_output(output)
                acc1, acc5 = self.calculate_accuracy(final_dict)
                print("=====================================")
                print(f"test mean of per class acc-1 step 0: {acc1}")
                print(f"test mean of per class acc-5 step 0: {acc5}")
                print("=====================================")
        return

    def train(self, common_pair=False, resume=False, amp=False, warmup_length=200, eval_ratio=0.2):
        testset_image_s = Imagenet_S(hi_res=True)
        testset_image_s_all_one = Imagenet_S(hi_res=True, all_one=True)
        testset_coco = COCO_Masked_Test(hi_res=True)

        # demo dataset
        # trainset = Alpha_GRIT(ids_file='grit_1m_keys_lightly.pkl',
        #                       root_pth='/data2/shared/grit/grit-1m-img-with-mask/',
        #                       common_pair=common_pair, subnum=self.subnum, hi_res=True)
        trainset = Alpha_GRIT(ids_file='grit_coyo_1_keys.pkl', root_pth='/data2/user_data/wenwen/data/GRIT/train/coyo_1_train/',
                              common_pair=common_pair, subnum=self.subnum, hi_res=True)

        test_loaders = self.setup_test_loaders(testset_coco, testset_image_s, testset_image_s_all_one)
        train_loader = self.setup_train_loader(trainset)
        self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length, steps=5000, para_gamma=self.para_gamma)
        start_epoch, resume_iter = self.resume_training(resume, train_loader)
        for epoch in range(start_epoch, self.num_epoch):
            if (trainset.__len__() * epoch) > 4000 * self.batch_size * 256:
                break
            loss = self.train_epoch(train_loader, test_loaders, epoch, start_iter=resume_iter, amp=amp, eval_ratio=eval_ratio)

    def setup_test_loaders(self, *testsets):
        test_loaders = {}
        for name, testset in zip(['COCO', 'Imagenet-S', 'Imagenet-S_all_one'], testsets):
            test_sampler = torch.utils.data.SequentialSampler(testset)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, sampler=test_sampler, num_workers=0, pin_memory=True)
            test_loaders[name] = test_loader
        return test_loaders

    def setup_train_loader(self, trainset):
        train_sampler = torch.utils.data.SequentialSampler(trainset)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)
        return train_loader

    def resume_training(self, resume, train_loader):
        start_epoch, resume_iter = 0, 0
        if resume and os.listdir(self.ckptdir):
            resume_pth = os.listdir(self.ckptdir)[-1]
            resume_iter = int(resume_pth[5:-4])
            start_epoch = resume_iter // len(train_loader)
            map_location = {'cuda:0': f'cuda:{self.local_rank}'}
            self.model.visual.load_state_dict(torch.load(os.path.join(self.ckptdir, resume_pth), map_location=map_location))
            print(f"load resumed checkpoint: {resume_pth}")
        return start_epoch, resume_iter

def setup_distributed(backend="nccl", port=None, distributed=False):
    if not distributed:
        return 0
    num_gpus = torch.cuda.device_count()
    if "SLURM_JOB_ID" in os.environ:
        rank, world_size, addr = setup_slurm_environment(num_gpus, port)
    else:
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    return rank % num_gpus

def setup_slurm_environment(num_gpus, port):
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    os.environ["MASTER_PORT"] = str(port) if port else os.environ.get("MASTER_PORT", "29991")
    os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank % num_gpus)
    os.environ["RANK"] = str(rank)
    return rank, world_size, addr

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--lr', default=4e-5, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='wd.')
    parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
    parser.add_argument('--lora_rank', default=-1, type=int, help='lora rank (-1 to not use lora).')
    parser.add_argument('--common_pair', default=0.0, type=float, help='propotion to use image with all 1 alpha and whole caption.')
    parser.add_argument('--para_gamma', default=0.01, type=float, help='para_gamma of other parameters')
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    parser.add_argument("--amp", action="store_true", help="bf16 taining.")
    parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    parser.add_argument("--epoch_num", default=4, type=int, help="number of epochs.")
    parser.add_argument("--subnum", default=1e4, type=float, help="sub data number.")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training.")
    parser.add_argument("--eval_ratio", default=0.1, type=float, help="Evaluation ratio during each epoch.")
    args = parser.parse_args()
    local_rank = setup_distributed(distributed=args.distributed)
    trainer = CLIP_Clean_Train(
        local_rank=local_rank,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_scale=args.log_scale,
        lora_rank=args.lora_rank,
        common_pair=args.common_pair,
        para_gamma=args.para_gamma,
        exp_name=args.exp_name,
        warmup_length=args.warmup_length,
        epoch_num=args.epoch_num,
        subnum=int(args.subnum)
    )
    trainer.train(common_pair=args.common_pair, resume=args.resume, amp=args.amp, warmup_length=args.warmup_length, eval_ratio=args.eval_ratio)