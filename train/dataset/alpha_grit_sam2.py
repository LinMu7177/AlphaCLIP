import cv2
import random
import json
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from pycocotools import mask as maskUtils

PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
MASK_FILL = [int(255 * c) for c in PIXEL_MEAN]


def get_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise IOError(f"Error reading file: {filepath}. Error: {str(e)}")


clip_standard_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

hi_clip_standard_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((336, 336), interpolation=Image.BICUBIC),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

res_clip_standard_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((336, 336), interpolation=Image.BICUBIC),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(0.5, 0.26)
])

hi_mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((336, 336)),
    transforms.Normalize(0.5, 0.26)
])

res_mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((336, 336)),
    transforms.Normalize(0.5, 0.26)
])


def crop_center(img, croph, cropw):
    h, w = img.shape[:2]
    starth = h // 2 - (croph // 2)
    startw = w // 2 - (cropw // 2)
    return img[starth:starth + croph, startw:startw + cropw, :]


class Alpha_GRIT_SAM2(Dataset):
    def __init__(self, ids_file, root_pth, common_pair=0.0, hi_res=False, subnum=None):
        if subnum is not None:
            self.ids = pickle.load(open(ids_file, 'rb'))[:subnum]
        else:
            self.ids = pickle.load(open(ids_file, 'rb'))
        self.root_pth = root_pth
        self.with_common_pair_prop = common_pair
        if hi_res:
            self.mask_transform = res_mask_transform
            self.clip_standard_transform = res_clip_standard_transform
        else:
            self.mask_transform = mask_transform
            self.clip_standard_transform = clip_standard_transform

    def __len__(self):
        return len(self.ids)

    def rle_to_mask(self, rle):
        return maskUtils.decode(rle)

    def load_mask(self, pkl_file_path):
        with open(pkl_file_path, 'rb') as f:
            masks = pickle.load(f)

        shape = masks[0]['segmentation']['size']
        combined_edges = np.zeros(shape, dtype=np.uint8)
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        for mask_data in sorted_masks:
            segmentation = mask_data['segmentation']
            rle_counts = segmentation['counts']
            binary_mask = self.rle_to_mask({'size': shape, 'counts': rle_counts})
            binary_mask = np.clip(binary_mask, 0, 1)
            edges = cv2.Canny(binary_mask.astype(np.uint8), 1, 1)
            combined_edges = np.maximum(combined_edges, edges)
        return combined_edges

    def __getitem__(self, index):
        id = self.ids[index]
        ann = json.loads(get_file(self.root_pth + str(id) + '.json'))
        image_data = get_file(self.root_pth + str(id) + '.jpg')
        img = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        mask = self.load_mask('/data2/shared/data/SAM2_Dataset/GRIT/train_1m_sam2_split/folder_1/' + str(id) + '_mask.pkl')

        if mask.shape != img.shape[:2]:
            img = np.rot90(img)
        rgba = np.concatenate((img, np.expand_dims(mask, axis=-1)), axis=-1)
        h, w = rgba.shape[:2]

        if max(h, w) == w:
            pad = (w - h) // 2
            l, r = pad, w - h - pad
            rgba = np.pad(rgba, ((l, r), (0, 0), (0, 0)), 'constant', constant_values=0)
        else:
            pad = (h - w) // 2
            l, r = pad, h - w - pad
            rgba = np.pad(rgba, ((0, 0), (l, r), (0, 0)), 'constant', constant_values=0)

        rgb = rgba[:, :, :-1]
        mask = rgba[:, :, -1]
        image_torch = self.clip_standard_transform(rgb)
        mask_torch = self.mask_transform(mask * 255)

        return image_torch, mask_torch, ann['caption']
