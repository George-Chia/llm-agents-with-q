#!/bin/bash

python run.py \
    --backend webshop-policy-llama31 \
    --data_split test \
    --n_generate_sample 3 \
    --temperature 1 \
    --iterations 30 \
    --log logs/eval_part.log \
    --save_path trajectories-MCTS-critique-round1 \
    --max_depth 10 \
    --rollout_width 1 \
    --algorithm mcts \
    --enable_fastchat_conv \
    --enable_seq_mode \
    --conv_template llama-3 \
    --expansion_sampling_method critique \
    --critique_backend webshop-critique-llama31 \
    --critique_prompt_template template_v1 \
