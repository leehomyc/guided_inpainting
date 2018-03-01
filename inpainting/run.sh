#!/usr/bin/env bash

python train.py --name coco_inpainting_256_real_guided_all --gpu_ids 0 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting_guided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/data/public/MSCOCO/annotations/instances_train2017.json'

python test.py --name coco_inpainting_256_real_guided --gpu_ids 0 --checkpoints_dir /media/ssd/harry/guided_inpainting/log/coco_inpainting_256_real_guided/20180131-225045 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/hdc/public/val2017 --model inpainting_guided --no_instance --resize_or_crop resize --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/hdc/public/annotations/instances_val2017.json' --how_many 200

python -m inpainting.train --name coco_inpainting_256_dilation_2_exp1_unguided --gpu_ids 0 --checkpoints_dir /home/xiaofen2/harry/guided_inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /home/datasets/COCO_2017/train2017 --model inpainting_unguided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/home/datasets/COCO_2017/annotations/instances_train2017.json' --interpolated_conv --dilation 2


python -m inpainting.train --name coco_inpainting_256_dilation_2_exp1_unguided_no_interpolated_conv --gpu_ids 1 --checkpoints_dir /home/xiaofen2/harry/guided_inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /home/datasets/COCO_2017/train2017 --model inpainting_unguided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/home/datasets/COCO_2017/annotations/instances_train2017.json' --dilation 2

python -m inpainting.train --name coco_inpainting_256_dilation_2_exp1_unguided_no_interpolated_conv_no_dilationn --gpu_ids 2 --checkpoints_dir /home/xiaofen2/harry/guided_inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /home/datasets/COCO_2017/train2017 --model inpainting_unguided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 4 --keep_hole_only --ann_path '/home/datasets/COCO_2017/annotations/instances_train2017.json'

python -m inpainting.train --name coco_inpainting_256_dilation_2_exp1_unguided_no_dilation --gpu_ids 3 --checkpoints_dir /home/xiaofen2/harry/guided_inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /home/datasets/COCO_2017/train2017 --model inpainting_unguided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/home/datasets/COCO_2017/annotations/instances_train2017.json' --interpolated_conv


python -m inpainting.train --name inpainting_ade20k --gpu_ids 0 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/ --model inpainting_ade20k --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only