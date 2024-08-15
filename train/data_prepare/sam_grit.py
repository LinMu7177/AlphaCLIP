import os
import json
import torch
import pickle
import tarfile
import argparse
import numpy as np
from PIL import Image
import io
from tqdm import tqdm
from itertools import groupby
from typing import Any, Dict, List
from pycocotools import mask as mask_utils
from segment_anything import sam_model_registry, SamPredictor

def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def process_tar_file(tar_pth: str, predictor, output_dir: str):
    result = dict()
    with tarfile.open(tar_pth, 'r') as t:
        image_pths = [pth for pth in t.getnames() if pth[-4:] == '.jpg']
        for img in tqdm(image_pths):
            ann = json.load(t.extractfile(img.replace('.jpg', '.json')))
            tarinfo = t.getmember(img)
            image = t.extractfile(tarinfo)
            image = image.read()
            pil_img = Image.open(io.BytesIO(image)).convert("RGB")
            image_h = pil_img.height
            image_w = pil_img.width
            grounding_list = ann['ref_exps']
            try:
                predictor.set_image(np.array(pil_img))
            except:
                print(np.array(pil_img).shape)
                print(img)
                print(ann['id'])
            segs = []
            for i, (phrase_s, phrase_e, x1_norm, y1_norm, x2_norm, y2_norm, score) in enumerate(grounding_list):
                x1, y1, x2, y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
                input_box = np.array(([x1, y1, x2, y2]))
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                rle = binary_mask_to_rle(np.asfortranarray(masks[0]))
                seg = coco_encode_rle(rle)
                segs.append(seg)
            result[ann['id']] = segs
    output_pkl_path = os.path.join(output_dir, os.path.basename(tar_pth).replace('.tar', '.pkl'))
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(result, f)

def generate_file_list(input_dir: str, start: int, end: int) -> List[str]:
    return [os.path.join(input_dir, f"{i:05d}.tar") for i in range(start, end + 1)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True, help='Path to the directory containing tar files')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to the directory to save pkl files')
    parser.add_argument('--start', type=int, required=True, help='Start index of tar files to process')
    parser.add_argument('--end', type=int, required=True, help='End index of tar files to process')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use for processing')
    args = parser.parse_args()

    tar_files = generate_file_list(args.input_dir, args.start, args.end)

    sam_checkpoint = "/mnt/shared/models/SAM/SAM-vit-h/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = f"cuda:{args.gpu_id}"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for tar_file in tar_files:
        process_tar_file(tar_file, predictor, args.output_dir)

if __name__ == "__main__":
    main()
