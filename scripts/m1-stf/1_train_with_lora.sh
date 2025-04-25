#!/bin/bash

mlx_lm.lora \
    --model "model/DeepSeek-R1-Distill-Qwen-1.5B-4bit" \
    --train \
    --data "data/sft" \
    --iters 100 \
    --batch-size 1 \
    --num-layers 4 \
    --learning-rate 1e-06 \
    --mask-prompt \
    --fine-tune-type lora \
    --grad-checkpoint \
    --save-every 5 \
    --max-seq-length 8000
