#!/bin/bash

python run.py \
    --backend gpt-4-turbo \
    --data_split test \
    --n_generate_sample 1 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 1 \
    --save_path trajectories \
    --log logs/gpt-4-turbo.log \
    --max_depth 7 \
    --algorithm simple \
    --enable_fastchat_conv \
    --enable_seq_mode \
    ${@}