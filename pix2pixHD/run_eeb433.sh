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

# inpainting and keep hole only. The experiment runs on Weiyue's computer
python train.py --name coco_inpainting_256_inpainting_keep_hole_only --gpu_ids 3 --checkpoints_dir /media/ssd/harry/guided_inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/hdc/public/train2017 --model inpainting --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only

# inpainting and keep hole only. However, we add a local discriminator. The experiment runs on Weiyue's computer
python train.py --name coco_inpainting_256_inpainting_keep_hole_only_local_discriminator --gpu_ids 2 --checkpoints_dir /media/ssd/harry/guided_inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/hdc/public/train2017 --model inpainting --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --use_local_discriminator

# inpainting and keep hole only. We increase the hole size such that it is slightly The experiment runs on Weiyue's computer.
# remember to change the experiment name.
python train.py --name coco_inpainting_256_inpainting_keep_hole_only_larger_hole --gpu_ids 1 --checkpoints_dir /media/ssd/harry/guided_inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/hdc/public/train2017 --model inpainting --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --overlapPred 0

# inpainting and keep hole only. The experiment runs on eeb433's computer
python train.py --name coco_inpainting_256_inpainting_keep_hole_only --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/pix2pixHD/log --loadSize 572 --fineSize 512 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/train2017 --model inpainting --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 4 --keep_hole_only --overlapPred 36

# test
python test.py --name coco_inpainting_256_inpainting_keep_hole_only/20180121-195015 --gpu_ids 1 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/pix2pixHD/log --loadSize 512 --fineSize 512 --label_nc 0 --dataroot /home/eeb433/Documents/Yuhang/guided_inpainting/pix2pixHD/examples --model inpainting --no_instance --resize_or_crop resize --tf_log --batchSize 4 --keep_hole_only --overlapPred 36 --n_blocks_global 9

# guided inpainting training for all classes
python train.py --name coco_inpainting_256_real_guided_all --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/train2017 --model inpainting_guided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_train2017.json'

# guided harmonization using gray object in all classes
python train.py --name coco_inpainting_256_object_color --gpu_ids 1 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/train2017 --model inpainting_object --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_train2017.json' --use_color

# guided inpainting testing
python test.py --name coco_inpainting_256_real_guided_all --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/log/coco_inpainting_256_real_guided_all/20180201-171136 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/val2017 --model inpainting_guided --no_instance --resize_or_crop resize --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_val2017.json' --how_many 200

# harmonization testing
python test.py --name coco_inpainting_256_object_color --gpu_ids 1 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/log_pix2pix_hd/coco_inpainting_256_object_color/20180201-171747 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/val2017 --model inpainting_object --no_instance --resize_or_crop resize --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_val2017.json' --how_many 200 --use_color

# guided inpainting training for all classes coco_inpainting_256_real_guided_all 		baseline
python -m inpainting.train --name coco_inpainting_256_baseline --gpu_ids 1 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/train2017 --model inpainting_guided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_train2017.json'

# guided inpainting training for all classes coco_inpainting_256_real_guided_all_interpolated_conv 			exp 1
python -m inpainting.train --name coco_inpainting_256_dilation_2_exp1 --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/train2017 --model inpainting_guided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_train2017.json' --interpolated_conv --dilation 2

# guided inpainting training for all classes coco_inpainting_256_real_guided_all_interpolated_conv no_ganFeat_loss no_vgg_loss recon_loss 			exp 2
python -m inpainting.train --name coco_inpainting_256_dilation_2_recon_no_vgg_no_ganFeat_exp2 --gpu_ids 1 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/train2017 --model inpainting_guided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_train2017.json' --interpolated_conv --dilation 2 --no_ganFeat_loss --no_vgg_loss --recon_loss
# guided inpainting training for all classes coco_inpainting_256_real_guided_all_interpolated_conv no_ganFeat_loss no_vgg_loss recon_loss lambda_recon 1000			exp 2_2
python -m inpainting.train --name coco_inpainting_256_dilation_2_recon_1000_no_vgg_no_ganFeat_exp2 --gpu_ids 1 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/train2017 --model inpainting_guided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_train2017.json' --interpolated_conv --dilation 2 --no_ganFeat_loss --no_vgg_loss --recon_loss --lambda_recon 1000
# guided inpainting training for all classes coco_inpainting_256_real_guided_all_interpolated_conv no_ganFeat_loss no_vgg_loss recon_loss lambda_recon 2500			exp 2_3
python -m inpainting.train --name coco_inpainting_256_dilation_2_recon_2500_no_vgg_no_ganFeat_exp2 --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/train2017 --model inpainting_guided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_train2017.json' --interpolated_conv --dilation 2 --no_ganFeat_loss --no_vgg_loss --recon_loss --lambda_recon 2500

# guided inpainting training for all classes coco_inpainting_256_real_guided_all_interpolated_conv  globalGAN_loss			exp 3
python -m inpainting.train --name coco_inpainting_256_dilation_2_recon_no_vgg_no_ganFeat_globalGAN_exp3 --gpu_ids 1 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/train2017 --model inpainting_guided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_train2017.json' --interpolated_conv --dilation 2 --globalGAN_loss

# guided inpainting training for all classes coco_inpainting_256_real_guided_all_interpolated_conv  globalGAN_loss	recon_loss lambda_recon 2500			exp 3_2
python -m inpainting.train --name coco_inpainting_256_dilation_2_recon_no_vgg_no_ganFeat_globalGAN_exp3_2 --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/train2017 --model inpainting_guided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_train2017.json' --interpolated_conv --dilation 2 --globalGAN_loss --recon_loss --lambda_recon 2500


# guided inpainting testing --baseline
python -m inpainting.test --name coco_inpainting_256_baseline --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log/coco_inpainting_256_baseline/20180214-223059 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/val2017 --model inpainting_guided --no_instance --resize_or_crop resize --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_val2017.json' --how_many 200

# guided inpainting testing --exp1
python -m inpainting.test --name coco_inpainting_256_dilation_2_exp1 --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log/coco_inpainting_256_dilation_2_exp1/20180214-223156 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/val2017 --model inpainting_guided --no_instance --resize_or_crop resize --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_val2017.json' --how_many 200 --interpolated_conv --dilation 2 

# guided inpainting testing --exp2
python -m inpainting.test --name coco_inpainting_256_dilation_2_recon_no_vgg_no_ganFeat_exp2 --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log/coco_inpainting_256_dilation_2_recon_no_vgg_no_ganFeat_exp2/20180217-165519 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/val2017 --model inpainting_guided --no_instance --resize_or_crop resize --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_val2017.json' --how_many 200 --interpolated_conv --dilation 2 

python -m inpainting.test --name coco_inpainting_256_dilation_2_recon_no_vgg_no_ganFeat_exp2_2 --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log/coco_inpainting_256_dilation_2_recon_1000_no_vgg_no_ganFeat_exp2/20180219-203859 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/val2017 --model inpainting_guided --no_instance --resize_or_crop resize --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_val2017.json' --how_many 200 --interpolated_conv --dilation 2 

python -m inpainting.test --name coco_inpainting_256_dilation_2_recon_no_vgg_no_ganFeat_exp2_3 --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log/coco_inpainting_256_dilation_2_recon_2500_no_vgg_no_ganFeat_exp2/20180219-203835 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/val2017 --model inpainting_guided --no_instance --resize_or_crop resize --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_val2017.json' --how_many 200 --interpolated_conv --dilation 2 

# guided inpainting testing --exp3
python -m inpainting.test --name coco_inpainting_256_dilation_2_recon_no_vgg_no_ganFeat_globalGAN_exp3 --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log/coco_inpainting_256_dilation_2_recon_no_vgg_no_ganFeat_globalGAN_exp3/20180221-185929 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/val2017 --model inpainting_guided --no_instance --resize_or_crop resize --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_val2017.json' --how_many 200 --interpolated_conv --dilation 2 

python -m inpainting.test --name coco_inpainting_256_dilation_2_recon_no_vgg_no_ganFeat_globalGAN_exp3_2 --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log/coco_inpainting_256_dilation_2_recon_no_vgg_no_ganFeat_globalGAN_exp3/20180221-190139 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/val2017 --model inpainting_guided --no_instance --resize_or_crop resize --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_val2017.json' --how_many 200 --interpolated_conv --dilation 2 


# guided inpainting testing --exp1 old
python -m inpainting.test --name coco_inpainting_256_dilation_2_old_3 --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log/coco_inpainting_256_dilation_2_old_3/20180214-135932 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/val2017 --model inpainting_guided --no_instance --resize_or_crop resize --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_val2017.json' --how_many 200 --interpolated_conv --dilation 2 





# inpainting and keep hole only. coco_no_guided_inpainting_256_interpolated_conv 			exp 1
python -m inpainting.train --name coco_no_guided_inpainting_256_exp1 --gpu_ids 1 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/train2017 --model inpainting --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --overlapPred 36 --interpolated_conv --dilation 2


# guided inpainting training for all classes coco_inpainting_256_real_guided_all_interpolated_conv  globalGAN_loss	recon_loss lambda_recon 2500			exp 3_2
python -m inpainting.train --name coco_no_guided_inpainting_256_exp3_2 --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/train2017 --model inpainting --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --overlapPred 36 --interpolated_conv --dilation 2 --globalGAN_loss --recon_loss --lambda_recon 2500




