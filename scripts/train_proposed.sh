#!/bin/bash
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=21229 --nnodes 1 --nproc-per-node 2 main.py \
    --save-frequency 5 \
    --val-frequency 2  \
    --save-most-recent \
    --precision="fp32" \
    --batch-size=192 \
    --num-mixes=4 \
    --lr=1e-4 \
    --wd=0.0 \
    --epochs=40 \
    --workers=4 \
    --loss-sch=10000 \
    --warmup=2000 \
    --report-to "wandb" \
    --wandb-entity "ytsrt" \
    --wandb-project "FxEncoderPlusPlus" \
    --wandb-notes "FxEncoderPlusPlus" \
    --name="FxEncoderPlusPlus" \
    --gather-with-grad \
    --use-bn-sync 
    
    
