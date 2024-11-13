#!/bin/bash

num_workers=4


explore_model_name=Phi-3
exp_name=Phi-3

for ((j=0;j<${num_workers};j=j+1)); do
    part_idx=$((j))
    python run.py \
        --backend ${explore_model_name}-${j} \
        --data_split train \
        --part_num ${num_workers} \
        --part_idx ${part_idx} \
        --n_generate_sample 5 \
        --n_evaluate_sample 1 \
        --rollout_width 1 \
        --prompt_sample cot \
        --temperature 1.0 \
        --iterations 70 \
        --save_path trajectories \
        --log logs/collect_trajectories_part-${j}.log \
        --max_depth 7 \
        --algorithm mcts \
        --enable_fastchat_conv \
        --conv_template phi3 &
    echo $! >> logs/${exp_name}-eval_pid.txt
done