# extract_model_state.py
import torch
import sys



input_path = '/data2/yangjiangang/codes/Robust_Vision/local_results/robust_exp/vr/algorithms/rewind/in1k/resnet/resnet50_imagenet_baseline_se0_te30_lrs0.0_s649/model_best.pth.tar'
output_path = '/data2/yangjiangang/codes/Robust_Vision/local_results/robust_exp/vr/algorithms/rewind/in1k/resnet/resnet50_imagenet_baseline_se0_te30_lrs0.0_s649/model_best_clean.pth.tar'

ckpt = torch.load(input_path, map_location='cpu')
if 'model' in ckpt:
    state_dict = ckpt['model']
else:
    raise KeyError("Key 'model' not found in checkpoint")

torch.save(state_dict, output_path)
print(f"Saved clean state_dict to {output_path}")