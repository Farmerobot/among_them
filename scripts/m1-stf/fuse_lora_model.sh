#!/bin/bash

mlx_lm.fuse \
    --model "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit" \
    --adapter-path adapters \
    --save-path model/fine-tuned_1.5B-4bit
