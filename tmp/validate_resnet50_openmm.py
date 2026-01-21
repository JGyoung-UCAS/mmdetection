from mmdet.models.backbones import ResNet
import torch

# Build OpenMMLab ResNet-50 (same as torchvision structure)
backbone = ResNet(
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=-1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=False,
    style='pytorch'  # Must match torchvision
)

# Load converted weights
state_dict = torch.load('/data2/yangjiangang/codes/Robust_Vision/local_results/robust_exp/vr/algorithms/rewind/in1k/resnet/resnet50_imagenet_baseline_se0_te30_lrs0.0_s649/model_best_for_mmdet.pth.tar', map_location='cpu')
backbone.load_state_dict(state_dict, strict=True)  # strict=True ensures perfect match

# Test forward
x = torch.randn(1, 3, 224, 224)
outputs = backbone(x)
for i, out in enumerate(outputs):
    print(f"C{i+2} shape: {out.shape}")