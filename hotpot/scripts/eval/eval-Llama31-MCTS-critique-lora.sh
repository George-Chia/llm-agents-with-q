#!/bin/bash

python run.py \
    --backend llama31-0 \
    --data_split test \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1 \
    --iterations 30 \
    --save_path trajectories-MCTS-lora-critique \
    --log logs/llama31.log \
    --max_depth 7 \
    --algorithm mcts \
    --enable_fastchat_conv \
    --enable_seq_mode \
    --conv_template llama-3 \
    --expansion_sampling_method critique \
    --critique_backend llama31-0 \
    --critique_conv_template llama-3 &
echo $! >> logs/llama31-MCTS-lora-eval_pid.txt

