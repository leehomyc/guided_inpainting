#!/usr/bin/env bash

# Vanilla pix2pix training for inpainting.
python train.py --dataroot /data/public/MSCOCO/train2017 --name coco_inpainting --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode inpainting --no_lsgan --norm batch --pool_size 0 --checkpoints_dir '/data/log/guided_inpainting/log'

# Add a global discriminator.
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /data/public/MSCOCO/train2017 --name coco_inpainting --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode inpainting --no_lsgan --norm batch --pool_size 0 --enable_global_disc --batchSize 32 --checkpoints_dir '/data/log/guided_inpainting/log_global_disc'

#