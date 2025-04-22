#!/bin/bash

mlx_lm.lora \
    --model "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B" \
    --train \
    --data "data/sft" \
    --iters 1 \
    --batch-size 1 \
    --num-layers 4 \
    --mask-prompt \
    --fine-tune-type lora \
    --grad-checkpoint \
    --save-every 1 \
    --max-seq-length 8000
