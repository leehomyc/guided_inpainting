#!/usr/bin/env bash

python test_ip_hm.py --name coco_inpainting_256_test_pair --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log/coco_inpainting_256_dilation_2_exp1/20180214-223156 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/val2017 --model inpainting_test --no_instance --resize_or_crop resize --tf_log --batchSize 1 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_val2017.json' --how_many 1

python -m test_ip_hm --name coco_inpainting_256_test_pair --gpu_ids 0 --checkpoints_dir /home/eeb433/Documents/Yuhang/guided_inpainting/inpainting/log/coco_inpainting_256_dilation_2_exp1/20180214-223156 --loadSize 256 --fineSize 256 --label_nc 0 --dataroot /media/eeb433/Storage/Dataset/MSCOCO/val2017 --model inpainting_test --no_instance --resize_or_crop resize --tf_log --batchSize 1 --keep_hole_only --ann_path '/media/eeb433/Storage/Dataset/MSCOCO/annotations/instances_val2017.json' --how_many 1