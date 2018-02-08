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

# inpainting and keep hole only. We increase the hole size such that it becomes larger (128x128). The experiment runs on Weiyue's computer.
# remember to change the experiment name.
python train.py --name coco_inpainting_256_inpainting_keep_hole_only_larger_hole --gpu_ids 1 --checkpoints_dir /media/ssd/harry/guided_inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/hdc/public/train2017 --model inpainting --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --overlapPred 0

# this is to train 512 with 256 hole.
python train.py --name coco_inpainting_512_inpainting_keep_hole_only_larger_hole --gpu_ids 0 --checkpoints_dir /media/ssd/harry/guided_inpainting/log --loadSize 572 --fineSize 512 --label_nc 0 --dataroot /media/hdc/public/train2017 --model inpainting --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --overlapPred 0

# inpainting and keep hole only. We increase the hole size such that it becomes larger (128x128). We also increase the number of ResNet blocks from 9 to 18. The experiment runs on Weiyue's computer. We also need to reduce the batch size given larger model size.
# remember to change the experiment name.
python train.py --name coco_inpainting_256_inpainting_keep_hole_only_larger_hole_more_resnet_blocks --gpu_ids 0 --checkpoints_dir /media/ssd/harry/guided_inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/hdc/public/train2017 --model inpainting --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 4 --keep_hole_only --overlapPred 0 --n_blocks_global 18

# inpainting and keep hole only. We increase the hole size such that it becomes larger (128x128). We also increase the number of ResNet blocks from 9 to 18. The experiment runs on OBen's computer. We also need to reduce the batch size given larger model size.
# remember to change the experiment name.
python train.py --name coco_inpainting_256_inpainting_keep_hole_only_larger_hole_more_resnet_blocks --gpu_ids 3 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 4 --keep_hole_only --overlapPred 0 --n_blocks_global 18

# test script. The results are saved in ./results folder.
python test.py --name coco_inpainting_256_inpainting_keep_hole_only_larger_hole_more_resnet_blocks/20180121-190059 --gpu_ids 2 --checkpoints_dir /media/ssd/harry/guided_inpainting/log --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/ssd/harry/guided_inpainting/original_input --model inpainting --no_instance --resize_or_crop resize --tf_log --batchSize 4 --keep_hole_only --overlapPred 32 --n_blocks_global 18

python test.py --name coco_inpainting_256_inpainting_keep_hole_only --gpu_ids 1 --checkpoints_dir /media/ssd/harry/guided_inpainting/log/coco_inpainting_256_inpainting_keep_hole_only --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/ssd/harry/guided_inpainting/original_input_gt --model inpainting --no_instance --resize_or_crop resize --tf_log --batchSize 4 --keep_hole_only --overlapPred 32

# guided inpainting with content weight set to default 0.25
python train.py --name coco_inpainting_256_guided --gpu_ids 0 --checkpoints_dir /media/ssd/harry/guided_inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/hdc/public/train2017 --model inpainting_guided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only

# guided inpainting with content weight set to 1
python train.py --name coco_inpainting_256_guided_content_weight_1 --gpu_ids 0 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting_guided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --content_weight 1

# guided inpainting with content weight set to 0.75
python train.py --name coco_inpainting_256_guided_content_weight_0.75 --gpu_ids 1 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting_guided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --content_weight 0.75

# guided inpainting with content weight set to 0.5
python train.py --name coco_inpainting_256_guided_content_weight_1 --gpu_ids 0 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting_guided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --content_weight 0.5

# object inpainting
python train.py --name coco_inpainting_256_object --gpu_ids 1 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting_object --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/data/public/MSCOCO/annotations/instances_train2017.json'

# test script
python test.py --name coco_inpainting_256_object --gpu_ids 2 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd/coco_inpainting_256_object/20180130-010239 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/val2017 --model inpainting_object --no_instance --resize_or_crop resize --tf_log --batchSize 4 --keep_hole_only --ann_path '/data/public/MSCOCO/annotations/instances_val2017.json' --how_many 200

python test.py --name coco_inpainting_256_object --gpu_ids 2 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd/coco_inpainting_256_object/20180130-010239 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting_object --no_instance --resize_or_crop resize --tf_log --batchSize 4 --keep_hole_only --ann_path '/data/public/MSCOCO/annotations/instances_train2017.json'

# inpainting with gray scale guidance
python train.py --name coco_inpainting_256_color --gpu_ids 0 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting_color --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only

# inpainting use mask
python train.py --name coco_inpainting_256_color_mask --gpu_ids 1 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting_color --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --use_mask

# inpainting use mask
python train.py --name coco_inpainting_256_object_color --gpu_ids 1 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting_object --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/data/public/MSCOCO/annotations/instances_train2017.json' --use_color

python test.py --name coco_inpainting_256_object --gpu_ids 2 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd/coco_inpainting_256_object/20180130-010239 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting_object --no_instance --resize_or_crop resize --tf_log --batchSize 4 --keep_hole_only --ann_path '/data/public/MSCOCO/annotations/instances_train2017.json'

# test object color
python test.py --name coco_inpainting_256_object_color --gpu_ids 2 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd/coco_inpainting_256_object_color/20180130-181109 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/val2017 --model inpainting_object --no_instance --resize_or_crop resize --tf_log --batchSize 4 --keep_hole_only --ann_path '/data/public/MSCOCO/annotations/instances_val2017.json' --use_color --how_many 100

# inpainting and keep hole only. The experiment runs on Weiyue's computer. The image is 128 and the hole is 64
python train.py --name coco_inpainting_128_inpainting_keep_hole_only --gpu_ids 0 --checkpoints_dir /media/ssd/harry/guided_inpainting/log --loadSize 143 --fineSize 128 --label_nc 0 --dataroot /media/hdc/public/train2017 --model inpainting --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 16 --keep_hole_only --overlapPred 0

# object inpainting 128
python train.py --name coco_inpainting_128_object --gpu_ids 1 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 143 --fineSize 128 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting_object --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/data/public/MSCOCO/annotations/instances_train2017.json'

# guided inpainting
python train.py --name coco_inpainting_256_real_guided --gpu_ids 0 --checkpoints_dir /media/ssd/harry/guided_inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/hdc/public/train2017 --model inpainting_guided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/hdc/public/annotations/instances_train2017.json'

# guided inpainting testing
python test.py --name coco_inpainting_256_real_guided --gpu_ids 0 --checkpoints_dir /media/ssd/harry/guided_inpainting/log/coco_inpainting_256_real_guided/20180131-225045 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/hdc/public/val2017 --model inpainting_guided --no_instance --resize_or_crop resize --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/hdc/public/annotations/instances_val2017.json' --how_many 200

# guided inpainting training for all classes
python train.py --name coco_inpainting_256_real_guided_all --gpu_ids 0 --checkpoints_dir /media/ssd/harry/guided_inpainting/log --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /media/hdc/public/train2017 --model inpainting_guided --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/hdc/public/annotations/instances_train2017.json'

# guided harmonization using gray object in all classes
python train.py --name coco_inpainting_256_object_color --gpu_ids 1 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 286 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting_object --no_instance --resize_or_crop resize_and_crop --tf_log --batchSize 8 --keep_hole_only --ann_path '/data/public/MSCOCO/annotations/instances_train2017.json' --use_color

# guided inpainting testing
python test.py --name coco_inpainting_256_real_guided --gpu_ids 0 --checkpoints_dir /media/ssd/harry/guided_inpainting/log/coco_inpainting_256_real_guided/20180131-225045 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/hdc/public/val2017 --model inpainting_guided --no_instance --resize_or_crop resize --tf_log --batchSize 8 --keep_hole_only --ann_path '/media/hdc/public/annotations/instances_val2017.json' --how_many 200

# harmonization testing
python test.py --name coco_inpainting_256_object_color --gpu_ids 1 --checkpoints_dir /data/log/guided_inpainting/log_pix2pix_hd --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /data/public/MSCOCO/train2017 --model inpainting_object --no_instance --resize_or_crop resize --tf_log --batchSize 8 --keep_hole_only --ann_path '/data/public/MSCOCO/annotations/instances_train2017.json' --how_many 200 --use_color

python test.py --name coco_inpainting_256_real_guided_test_pair --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/log/coco_inpainting_256_real_guided_all/20180201-171136 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/val2017 --model inpainting_test --no_instance --resize_or_crop resize --tf_log --batchSize 1 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_val2017.json' --how_many 1

python test.py --name coco_inpainting_256_real_harmonization_test_pair --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/log_pix2pix_hd/coco_inpainting_256_object_color/20180201-171747 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/val2017 --model harmonization_test --no_instance --resize_or_crop resize --tf_log --batchSize 1 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_val2017.json' --how_many 1

python pix2pixHD/test_ip_hm.py --name coco_inpainting_256_real_guided_test_pair --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/log/coco_inpainting_256_real_guided_all/20180201-171136 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/val2017 --model inpainting_test --no_instance --resize_or_crop resize --tf_log --batchSize 1 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_val2017.json' --how_many 1