#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
	--data data/refer \
	--dataset unc \
	--val val \
	--batch-size 16 \
    -j 1 \
	--visual-interval 100 \
	--tensorboard \
	--sync-bn \
	--snapshot models/resnet101/sanet_best_model_unc.pth \
	--backbone resnet101