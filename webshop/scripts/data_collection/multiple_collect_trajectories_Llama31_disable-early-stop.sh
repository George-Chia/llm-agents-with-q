#!/bin/bash

num_workers=4
node_num=9

explore_model_name=llama31
exp_name=llama31-exploration


for ((j=0;j<${num_workers};j=j+1)); do
    part_idx=$((j))
    python run.py \
    --backend ${explore_model_name}-${j} \
    --data_split train \
    --part_num ${num_workers} \
    --part_idx ${part_idx} \
    --n_generate_sample 5 \
    --temperature 1.0 \
    --iterations 20 \
    --log logs/collect_trajectories_part-${j}.log \
    --save_path trajectories-disable-early-stop \
    --max_depth 10 \
    --rollout_width 1 \
    --algorithm mcts \
    --enable_fastchat_conv \
    --enable_seq_mode \
    --disable_early_stop \
    --conv_template llama-3 &
    echo $! >> logs/${exp_name}-eval_pid.txt
done