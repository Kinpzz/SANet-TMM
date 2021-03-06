#! /bin/bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
    --data data/refer \
    --dataset unc \
    --split train \
    --val val \
    --time 20 \
    --batch-size 16 \
    --seed 1111 \
    -j 1 \
    --lr 2.5e-5 \
    --visual-interval 500 \
    --tensorboard \
    --lang-layers 1 \
    --sync-bn \
    --os 16 \
    --snapshot models/dpn92_extra-b040e4a9b.pth \
    --patience 0 \
    --start-epoch 0 \
    --backbone dpn92 \
    --save-folder models/dpn92