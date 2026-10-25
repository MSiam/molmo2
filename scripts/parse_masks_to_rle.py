import argparse
import os
from glob import glob
import subprocess
from tqdm import tqdm
import json

def split_match(split_names, root):
    for name in split_names:
        if name in root:
            return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Molmo2 Annotations as RLE Masks")
    parser.add_argument("--in_dir", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--start_idx", type=int, default=0)
    args = parser.parse_args()

    if args.split == "train":
        split_names = ["all", "train"]
    else:
        split_names = [args.split]

    out_path = os.path.join(args.out_dir, args.split, "MasksRLE")
    os.makedirs(out_path, exist_ok=True)

    in_mask_path = os.path.join(args.in_dir, args.split, "mask_dict.json")
    in_exp_path = os.path.join(args.in_dir, args.split, "meta_expressions.json")

    if not os.path.exists(in_mask_path):
        raise Exception(f"Input director doesnt exist {in_mask_path}")
    if not os.path.exists(in_exp_path):
        raise Exception(f"Input director doesnt exist {in_exp_path}")

    exp_dict = json.load(open(in_exp_path))['videos']
    mask_dict = json.load(open(in_mask_path))

    queries = [[], [], []]
    for vid_name in exp_dict:
        vid_dict = exp_dict[vid_name]
        for exp_id in vid_dict['expressions']:
            anno_ids = vid_dict['expressions'][exp_id]['anno_id']
            queries[0].append(vid_name)
            queries[1].append(exp_id)
            queries[2].append(anno_ids)

    n_encoded = 0
    n_skipped = 0
    for (video_id, query_id, anno_ids) in zip(*queries):
        query_output = os.path.join(out_path, video_id, f"{query_id}.json")
        if os.path.exists(query_output):
            n_skipped += 1
            continue

        mask_annot = {}
        for mask_idx, anno_id in enumerate(anno_ids):
            anno_id = str(anno_id) # mask_dict keys are strings
            mask_annot[mask_idx] = mask_dict[anno_id]
        os.makedirs(os.path.join(out_path, video_id), exist_ok=True)
        with open(query_output, 'w') as f:
            json.dump(mask_annot, f)
        n_encoded += 1
    print("# Encoded mask ", n_encoded, "# skipped masks ", n_skipped)
