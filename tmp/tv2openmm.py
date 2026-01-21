# convert_torchvision_resnet50_to_mmdet.py (CORRECTED)
import torch
import argparse
import os

def convert(src_path, dst_path):
    print(f"Loading checkpoint from {src_path}")
    ckpt = torch.load(src_path, map_location='cpu')
    
    # 提取 state_dict
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print("Found 'state_dict'")
    elif 'model' in ckpt:
        state_dict = ckpt['model']
        print("Found 'model'")
    else:
        state_dict = ckpt
        print("Using top-level as state_dict")

    # 仅移除分类头（fc），其余保留原名！
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('fc'):
            print(f"Skipping: {k}")
            continue
        new_state_dict[k] = v  # ←←← 关键：不改名！

    torch.save(new_state_dict, dst_path)
    print(f"\n✅ Saved to {dst_path} (keys: {len(new_state_dict)})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('dst')
    args = parser.parse_args()
    convert(args.src, args.dst)