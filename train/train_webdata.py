import os
import sys
sys.path.append('../')
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import argparse
import alpha_clip
from tqdm import tqdm
import loralib as lora
from utils import concat_all_gather, is_dist_avail_and_initialized
from scheduler import cosine_lr
from dataset.imagenet_s_test_edges import Imagenet_S_Edges
from dataset.webdata_with_edges import WebData_With_Edges
from datetime import datetime

simple_templates = ['a photo of a {}.']

class CLIP_Clean_Train():
    def __init__(self, local_rank=0, lr=2e-4, log_scale=4.6052, model_name="ViT-L/14", hi_res=False, lora_rank=-1,
                 para_gamma=0.01, exp_name="auto", epoch_num=1, distributed=False, batch_size=8, resume=False, resume_log_dir=None):
        self.local_rank = local_rank
        self.distributed = distributed
        self.model_name = model_name
        self.model = self.load_model(self.model_name, lora_rank)
        self.hi_res = hi_res
        torch.cuda.set_device(device=f'cuda:{local_rank}')
        self.model = self.model.float().cuda()
        self.batch_size = batch_size
        self.num_epoch = epoch_num
        self.lr = lr
        self.resume_log_dir = resume_log_dir
        self.logdir = self.get_logdir(exp_name, lr, model_name, epoch_num, resume)
        self.ckptdir = os.path.join(self.logdir, "ckpt/")
        os.makedirs(self.ckptdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * log_scale)
        self.optimizer = self.configure_optimizer(lora_rank, para_gamma, lr)
        self.para_gamma = para_gamma
        self.scaler = torch.cuda.amp.GradScaler()
        self.best_accuracy = 0.0

        # Print parameters at initialization
        self.print_parameters()

    def print_parameters(self):
        print("========== CLIP_Clean_Train Parameters ==========")
        print(f"Local Rank: {self.local_rank}")
        print(f"Model Name: {self.model_name}")
        print(f"Hi-Res: {self.hi_res}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Number of Epochs: {self.num_epoch}")
        print(f"Learning Rate: {self.lr}")
        print(f"Resume Log Directory: {self.resume_log_dir}")
        print(f"Log Directory: {self.logdir}")
        print(f"Checkpoint Directory: {self.ckptdir}")
        print(f"Para Gamma: {self.para_gamma}")
        print(f"Logit Scale: {self.model.logit_scale.item()}")
        print("================================================")

    def load_model(self, model_name: str, lora_rank: int):
        if lora_rank == -1:
            model, _ = alpha_clip.load(model_name, device='cpu', lora_adapt=False, rank=-1)
            # model, _ = alpha_clip.load("ViT-L/14@336px","/mnt/shared/models/alphaclip/model_zoo/clip_l14_336_grit_20m_4xe.pth", device='cpu', lora_adapt=False, rank=-1)
        else:
            model, _ = alpha_clip.load(model_name, device='cpu', lora_adapt=True, rank=lora_rank)
        return model

    def get_logdir(self, exp_name: str, lr: float, model_name: str, epoch_num: int, resume: bool) -> str:
        if resume:
            if self.resume_log_dir is None:
                raise ValueError("Please specify the logdir for resume training.")
            else:
                return self.resume_log_dir
        date_str = datetime.now().strftime("%Y%m%d")
        base_logdir = f"log/{date_str}_grit_edges/lr={lr}_model_name={model_name}_e{epoch_num}_resume={resume}"
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
    def zeroshot_classifier(self, classnames, templates):
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
        for i, (images, masks, captions) in enumerate(tqdm(dataloader)):
            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue
            self.optimizer.zero_grad()
            self.scheduler(step)
            images = images.cuda()
            masks = masks.cuda()
            texts = alpha_clip.tokenize(captions).cuda()
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
                self.log_metrics(running_loss, 500, step)
                running_loss = 0.0
            if eval_step > 0 and (i + 1) % eval_step == 0:
                self.evaluate(step, test_loaders, save_checkpoint=True)
        return running_loss / batch_num

    def log_metrics(self, running_loss, interval, step):
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

    def evaluate(self, step, test_loaders, save_checkpoint=False):
        torch.cuda.empty_cache()
        self.model.visual.eval()
        current_accuracy = 0.0
        for test_name, test_loader in test_loaders.items():
            tqdm.write(f"Zeroshot Classifier Evaluating {test_name} at step {step}")
            self.text_embeddings = self.zeroshot_classifier(test_loader.dataset.classes, simple_templates)
            temp_corr_dict = self.test_epoch(test_loader, desc=f"Evaluating {test_name}")
            output = self.gather_output(temp_corr_dict)
            if not dist.is_initialized() or dist.get_rank() == 0:
                acc1, acc5 = self.log_test_results(test_name, step, output)
                current_accuracy += acc1  # Sum up accuracies from all test sets

        # Calculate average accuracy across all test sets
        current_accuracy /= len(test_loaders)

        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            if save_checkpoint:
                self.save_checkpoint(step)
                print(f"New best accuracy: {self.best_accuracy:.4f}. Checkpoint saved at step {step}.")

        self.model.visual.train()
        torch.cuda.empty_cache()

    def save_checkpoint(self, step):
        if not dist.is_initialized() or dist.get_rank() == 0:
            checkpoint_path = os.path.join(self.ckptdir, 'best_model.pth')
            torch.save({
                'step': step,
                'model_state_dict': self.model.visual.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_accuracy': self.best_accuracy,
            }, checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")

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
        return acc1, acc5

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
        testset = Imagenet_S_Edges()
        self.text_embeddings = self.zeroshot_classifier(testset.classes, simple_templates)
        sampler = DistributedSampler(dataset=testset, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size * 6, sampler=sampler, num_workers=8, pin_memory=True)
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

    def train(self, train_data_path, edges_path, num_workers, resume, amp, warmup_length, eval_ratio):
        testset_image_s = Imagenet_S_Edges(hi_res=self.hi_res)
        test_loaders = self.setup_test_loaders(testset_image_s)

        trainset = WebData_With_Edges(data_path=train_data_path, edges_path=edges_path)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, num_workers=num_workers, pin_memory=True)

        self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length, steps=5000, para_gamma=self.para_gamma)
        start_epoch, resume_iter = self.resume_training(resume, train_loader)

        print("Evaluating model before training starts...")
        self.evaluate(0, test_loaders)

        for epoch in range(start_epoch, self.num_epoch):
            loss = self.train_epoch(train_loader, test_loaders, epoch, start_iter=resume_iter, amp=amp, eval_ratio=eval_ratio)

    def setup_test_loaders(self, *testsets):
        test_loaders = {}
        # for name, testset in zip(['COCO', 'Imagenet-S', 'Imagenet-S_all_one'], testsets):
        for name, testset in zip(['Imagenet-S'], testsets):
            test_sampler = torch.utils.data.SequentialSampler(testset)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size * 6, sampler=test_sampler, num_workers=8, pin_memory=True)
            test_loaders[name] = test_loader
        return test_loaders

    def resume_training(self, resume, train_loader):
        start_epoch, resume_iter = 0, 0
        if resume and os.listdir(self.ckptdir):
            checkpoint_path = os.path.join(self.ckptdir, 'best_model.pth')
            if os.path.exists(checkpoint_path):
                map_location = {'cuda:0': f'cuda:{self.local_rank}'}
                checkpoint = torch.load(checkpoint_path, map_location=map_location)
                self.model.visual.load_state_dict(checkpoint['model_state_dict'])
                self.best_accuracy = checkpoint['best_accuracy']
                resume_iter = checkpoint['step']
                start_epoch = resume_iter // len(train_loader)
                print(f"Resumed from checkpoint: {checkpoint_path}")
                print(f"Best accuracy so far: {self.best_accuracy:.4f}")
            else:
                print("No checkpoint found. Starting from scratch.")
        return start_epoch, resume_iter


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--model_name', default='/mnt/shared/unibench/models/open-clip/CLIP-convnext_base_w-laion2B-s13B-b82K/open_clip_pytorch_model.bin', type=str)
    parser.add_argument('--hi_res', default=False, type=bool)
    parser.add_argument('--lr', default=2e-4, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=2e-2, type=float, help='wd.')
    parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
    parser.add_argument('--lora_rank', default=-1, type=int, help='lora rank (-1 to not use lora).')
    parser.add_argument('--common_pair', default=0.1, type=float, help='propotion to use image with all 1 alpha and whole caption.')
    parser.add_argument('--para_gamma', default=0.01, type=float, help='para_gamma of other parameters')
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    parser.add_argument("--amp", action="store_true", help="bf16 taining.")
    parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    parser.add_argument("--epoch_num", default=6, type=int, help="number of epochs.")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training.")
    parser.add_argument("--eval_ratio", default=0.2, type=float, help="Evaluation ratio during each epoch.")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--resume_log_dir", default="log/auto", type=str)
    parser.add_argument("--train_data_path", default="/mnt/shared/data/CC3M/cc3m/{00000..00331}.tar", type=str)
    parser.add_argument("--edges_path", default="/mnt/shared/data/DINO_SAM2_Data/cc3m/", type=str)
    args = parser.parse_args()

    local_rank = args.local_rank
    trainer = CLIP_Clean_Train(
        local_rank=args.local_rank,
        model_name=args.model_name,
        hi_res=args.hi_res,
        lr=args.lr,
        log_scale=args.log_scale,
        lora_rank=args.lora_rank,
        para_gamma=args.para_gamma,
        exp_name=args.exp_name,
        epoch_num=args.epoch_num,
        batch_size=args.batch_size,
        resume=args.resume,
        resume_log_dir=args.resume_log_dir
    )
    trainer.train(train_data_path=args.train_data_path, edges_path=args.edges_path, num_workers=args.num_workers, resume=args.resume, amp=args.amp, warmup_length=args.warmup_length, eval_ratio=args.eval_ratio)
