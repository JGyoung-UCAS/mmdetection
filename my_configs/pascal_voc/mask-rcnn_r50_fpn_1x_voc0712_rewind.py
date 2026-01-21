_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py',
]

# VOC has 20 classes
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=dict(num_classes=20),
    )
)

max_epochs = 4
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[3],
        gamma=0.1,
    )
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
)

auto_scale_lr = dict(enable=False, base_batch_size=16)

#self_defined
train_dataloader=dict(batch_size=4,num_workers=4)
model=dict(backbone=dict(init_cfg=dict(type='Pretrained',checkpoint='/data2/yangjiangang/codes/Robust_Vision/local_results/robust_exp/vr/finetune/rewind/in1k/rewind/in1k/resnet/resnet50_imagenet_baseline_se0_te30_lrs0.0_s649/model_best_for_mmdet.pth.tar')))
