import os
import json
import torch
from torch.utils.data import Dataset
from pycocotools import mask as maskUtils
import cv2
import pickle
from scipy.ndimage import convolve
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
from nltk.corpus import wordnet

PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
MASK_FILL = [int(255 * c) for c in PIXEL_MEAN]

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
    starth = h//2 - (croph//2)
    startw = w//2 - (cropw//2)    
    return img[starth:starth+croph, startw:startw+cropw, :]

class Imagenet_S_SAM2(Dataset):
    def __init__(self, ann_file='/data2/shared/imagenet-s/data/imagenet_919.json', root_pth='/data2/shared/imagenet-s/data/', hi_res=False, all_one=False):
        if torch.cuda.device_count() == 1:
            ann_file = '/mnt/shared/data/imagenet-s/imagenet_919.json'
            root_pth = '/mnt/shared/data/imagenet-s/'

        self.anns = json.load(open(ann_file, 'r'))
        self.root_pth = root_pth
        cats = []
        for ann in self.anns:
            if ann['category_word'] not in cats:
                cats.append(ann['category_word'])
            ann['cat_index'] = len(cats) - 1
        self.classes = []
        for cat_word in cats:
            synset = wordnet.synset_from_pos_and_offset('n', int(cat_word[1:]))
            synonyms = [x.name() for x in synset.lemmas()]
            self.classes.append(synonyms[0])
            
        self.choice = "center_crop"
        if hi_res:
            self.mask_transform = res_mask_transform
            self.clip_standard_transform = res_clip_standard_transform
        else:
            self.mask_transform = mask_transform
            self.clip_standard_transform = clip_standard_transform

        self.all_one = all_one

    def __len__(self):
        return len(self.anns)

    def rle_to_mask(self, rle):
        return maskUtils.decode(rle)

    def binary_mask_edges(self, binary_mask):
        kernel = np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]])

        edges = convolve(binary_mask.astype(np.float32), kernel)
        edges = np.clip(edges, 0, 1)
        return edges.astype(np.uint8)

    def load_mask(self, pkl_file_path, image_shape):
        if os.path.exists(pkl_file_path):
            with open(pkl_file_path, 'rb') as f:
                masks = pickle.load(f)

            shape = masks[0]['segmentation']['size']
            combined_edges = np.zeros(shape, dtype=np.uint8)
            sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

            for mask_data in sorted_masks:
                if mask_data['area'] < 1000:
                    continue
                segmentation = mask_data['segmentation']
                rle_counts = segmentation['counts']
                binary_mask = self.rle_to_mask({'size': shape, 'counts': rle_counts})
                # binary_mask = np.clip(binary_mask, 0, 1)
                # edges = cv2.Canny(binary_mask.astype(np.uint8), 1, 1)
                edges = self.binary_mask_edges(binary_mask)
                combined_edges = np.maximum(combined_edges, edges)
            return combined_edges
        else:
            return np.ones(image_shape[:2], dtype=np.uint8)

    def __getitem__(self, index):
        ann = self.anns[index]
        image = cv2.imread(self.root_pth + ann['image_pth'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self.root_pth + 'image_sam2/' + os.path.basename(ann['image_pth']).replace('.JPEG','') + '_mask.pkl'
        mask = self.load_mask(mask_path, image.shape)

        rgba = np.concatenate((image, np.expand_dims(mask, axis=-1)), axis=-1)
        h, w = rgba.shape[:2]
        
        if self.choice == "padding":
            if max(h, w) == w:
                pad = (w - h) // 2
                l, r = pad, w - h - pad
                rgba = np.pad(rgba, ((l, r), (0, 0), (0, 0)), 'constant', constant_values=0)
            else:
                pad = (h - w) // 2
                l, r = pad, h - w - pad
                rgba = np.pad(rgba, ((0, 0), (l, r), (0, 0)), 'constant', constant_values=0)
        else:
            if min(h, w) == h:
                rgba = crop_center(rgba, h, h)
            else:
                rgba = crop_center(rgba, w, w)
        rgb = rgba[:, :, :-1]
        mask = rgba[:, :, -1]
        image_torch = self.clip_standard_transform(rgb)
        bi_mask = mask == 1
        h, w = bi_mask.shape[-2:]
        in_height = np.max(bi_mask, axis=-1)
        in_height_coords = np.max(bi_mask, axis=-1) * np.arange(h)
        b_e = in_height_coords.max()
        in_height_coords = in_height_coords + h * (~in_height)
        t_e = in_height_coords.min()
        in_width = np.max(bi_mask, axis=-2)
        in_width_coords = np.max(bi_mask, axis=-2) * np.arange(w)
        r_e = in_width_coords.max()
        in_width_coords = in_width_coords + w * (~in_width)
        l_e = in_width_coords.min()
        if self.all_one:
            mask_torch = self.mask_transform(np.ones_like(mask) * 255)
        else: 
            mask_torch = self.mask_transform(mask * 255)

        return image_torch, mask_torch, ann['cat_index']

if __name__ == "__main__":
    data = Imagenet_S_SAM2()
    for i in tqdm(range(data.__len__())):
        data.__getitem__(i)