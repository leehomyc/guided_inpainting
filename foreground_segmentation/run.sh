#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python -m foreground_segmentation.coco_fg train --dataset "/data/public/MSCOCO"

# test original COCO
python -m third_party.Mask_RCNN.coco train --dataset="/data/public/MSCOCO" --model="third_party/Mask_RCNN/mask_rcnn_coco.h5" --year=2017