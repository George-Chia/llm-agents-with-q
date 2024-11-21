#!/bin/bash

python run.py \
    --backend llama31-3 \
    --data_split test \
    --n_generate_sample 1 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1 \
    --iterations 1 \
    --save_path trajectories-T1 \
    --log logs/llama31.log \
    --max_depth 7 \
    --algorithm simple \
    --enable_fastchat_conv \
    --enable_seq_mode \
    --conv_template llama-3 