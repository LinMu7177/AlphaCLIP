import os
import pickle
import numpy as np
from PIL import Image
import webdataset as wds
from torchvision import transforms
from torch.utils.data import IterableDataset
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

class WebData_With_Edges(IterableDataset):
    def __init__(self, data_path, edges_path, hi_res=False):
        self.data_path = data_path
        self.edges_path = edges_path
        self.hi_res = hi_res
        self._length = None

        if hi_res:
            self.mask_transform = hi_mask_transform
            self.clip_standard_transform = hi_clip_standard_transform
        else:
            self.mask_transform = mask_transform
            self.clip_standard_transform = clip_standard_transform

        self.dataset = wds.WebDataset(self.data_path).decode("pil").to_tuple("__key__", "jpg", "txt")

    def __len__(self):
        if self._length is None:
            self._length = sum(1 for _ in self.dataset)
        return self._length

    def rle_to_mask(self, rle):
        return maskUtils.decode(rle)

    def load_mask(self, edges_demo_path, image_shape):
        if os.path.exists(edges_demo_path):
            with open(edges_demo_path, 'rb') as f:
                combined_edges = pickle.load(f)
            rle = {'size': combined_edges['size'], 'counts': combined_edges['counts']}
            mask = self.rle_to_mask(rle)
            return mask
        else:
            return np.ones(image_shape[:2], dtype=np.uint8)

    def __iter__(self):
        for key, img, txt in self.dataset:
            caption, file_name, img = txt, key, np.array(img.convert('RGB'))

            edge_path = os.path.join(self.edges_path, file_name + '_edges.pkl')
            edge = self.load_mask(edge_path, img.shape)

            if edge.shape != img.shape[:2]:
                img = np.rot90(img)
            rgba = np.concatenate((img, np.expand_dims(edge, axis=-1)), axis=-1)
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
            image_torch = self.clip_standard_transform(Image.fromarray(rgb))
            mask_torch = self.mask_transform(Image.fromarray(mask * 255))

            yield image_torch, mask_torch, caption
