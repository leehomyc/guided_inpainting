#!/usr/bin/env bash

# vanilla
python train.py --name coco_inpainting --gpu_ids 0 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 572 --fineSize 512 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting --no_instance --resize_or_crop scale_width_and_crop --tf_log

# keep hole
python train.py --name coco_inpainting_hole --gpu_ids 1 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 572 --fineSize 512 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting --no_instance --resize_or_crop scale_width_and_crop --tf_log --keep_hole_only

# 256
python train.py --name coco_inpainting_256 --gpu_ids 1 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting --no_instance --resize_or_crop scale_width_and_crop --tf_log

# increase number of discriminators
python train.py --name coco_inpainting_256_batch_32_num_D_3 --gpu_ids 2 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 16 --num_D 3

# increase feature loss
python train.py --name coco_inpainting_256_batch_32_num_D_3_lambda_feat_100 --gpu_ids 3 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 16 --num_D 3 --lambda_feat 100

# dilation
python train.py --name coco_inpainting_256_batch_32_num_D_3_lambda_feat_100_dilate_2 --gpu_ids 0 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 16 --num_D 3 --lambda_feat 100 --dilation 2

# transposed convolution
python train.py --name coco_inpainting_256_batch_32_num_D_3_lambda_feat_100_dilate_2_interpolated_conv --gpu_ids 1 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --num_D 3 --lambda_feat 100 --dilation 2 --interpolated_conv

# inpainting grid model
python train.py --name coco_inpainting_256_inpainting_grid --gpu_ids 3 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 215 --fineSize 192 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting_grid --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8