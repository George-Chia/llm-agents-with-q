#!/bin/bash

num_workers=10


explore_model_name=gpt-4o-mini
exp_name=gpt-4o-mini


for ((j=0;j<${num_workers};j=j+1)); do
    part_idx=$((j))
        python run.py \
            --backend gpt-4o-mini \
            --data_split test \
            --part_num ${num_workers} \
            --part_idx ${part_idx} \
            --n_generate_sample 1 \
            --n_evaluate_sample 1 \
            --prompt_sample cot \
            --temperature 1.0 \
            --iterations 1 \
            --save_path trajectories-reflection \
            --log logs/gpt-4o-mini.log \
            --max_depth 7 \
            --algorithm simple \
            --enable_fastchat_conv \
            --enable_seq_mode \
            --enable_reflection &
    echo $! >> logs/${exp_name}-eval_pid.txt
done