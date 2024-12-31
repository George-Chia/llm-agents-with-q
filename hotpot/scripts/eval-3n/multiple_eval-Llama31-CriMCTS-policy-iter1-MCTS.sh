#!/bin/bash
num_workers=3

explore_model_name=llama31-hotpot
exp_name=llama31-collection-KTO-critique

for ((j=0;j<${num_workers};j=j+1)); do
    part_idx=$((j))
    python run.py \
        --backend ${explore_model_name}-${j} \
        --data_split test \
        --part_num ${num_workers} \
        --part_idx ${part_idx} \
        --n_generate_sample 3 \
        --n_evaluate_sample 1 \
        --prompt_sample cot \
        --temperature 1 \
        --iterations 30 \
        --save_path trajectories-Critique-MCTS-3n-policy-iteration-huan \
        --log logs/llama31.log \
        --max_depth 7 \
        --algorithm mcts \
        --enable_fastchat_conv \
        --enable_seq_mode \
        --conv_template llama-3 \
        --expansion_sampling_method critique \
        --critique_backend Llama31 \
        --critique_prompt_template template_huan \
        --critique_temperature 1 &
    echo $! >> logs/${exp_name}-eval_pid.txt
done