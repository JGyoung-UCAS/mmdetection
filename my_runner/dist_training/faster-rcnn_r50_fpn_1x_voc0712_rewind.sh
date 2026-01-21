#!/usr/bin/env bash

CONFIG1='my_configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712_rewind.py'

GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS_ids="1,2,3,4"
WORK_DIR1='local_results/od/voc/fasterRCNN_r50_fpn_1x_rewind'


#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPUS_ids python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG1 \
    --work-dir $WORK_DIR1 \
    --auto-scale-lr \
    --launcher pytorch ${@:3}

# CUDA_VISIBLE_DEVICES=$GPUS_ids python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     tools/train.py \
#     $CONFIG2 \
#     --work-dir $WORK_DIR2 \
#     --auto-scale-lr \
#     --launcher pytorch ${@:3}


