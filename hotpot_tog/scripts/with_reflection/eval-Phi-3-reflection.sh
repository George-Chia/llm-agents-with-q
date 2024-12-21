#!/bin/bash

num_workers=6


explore_model_name=Phi-3
exp_name=Phi-3-eval

for ((j=0;j<${num_workers};j=j+1)); do
    part_idx=$((j))
        python run.py \
        --backend Phi-3-${j} \
        --data_split test \
        --part_num ${num_workers} \
        --part_idx ${part_idx} \
        --n_generate_sample 1 \
        --n_evaluate_sample 1 \
        --prompt_sample cot \
        --temperature 1.0 \
        --iterations 1 \
        --save_path trajectories-reflection \
        --log logs/eval-Phi-3.log \
        --max_depth 10 \
        --algorithm simple \
        --enable_fastchat_conv \
        --enable_seq_mode \
        --conv_template phi3 \
        --enable_reflection &
    echo $! >> logs/${exp_name}-eval_pid.txt
done