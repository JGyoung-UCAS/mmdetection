_base_ = [
    '../../configs/_base_/models/retinanet_r50_fpn.py',
    '../../configs/_base_/datasets/coco_detection.py',
    '../../configs/_base_/schedules/schedule_1x.py', '../../configs/_base_/default_runtime.py',
    '../../configs/retinanet/retinanet_tta.py'
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
    
train_dataloader=dict(batch_size=4,num_workers=4)
model=dict(backbone=dict(init_cfg=dict(type='Pretrained',checkpoint='/data2/yangjiangang/codes/Robust_Vision/local_results/robust_exp/vr/finetune/rewind/in1k/rewind/in1k/resnet/resnet50_imagenet_baseline_se0_te30_lrs0.0_s649/model_best_for_mmdet.pth.tar')))
