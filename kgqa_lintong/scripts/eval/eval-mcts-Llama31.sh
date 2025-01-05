#!/bin/bash

python run.py \
    --backend Llama31 \
    --n_generate_sample 5 \
    --save_path trajectories \
    --log logs/Llama31.log \
    --max_depth 5 \
    --algorithm mcts \
    --conv_template llama-3 \
    --iterations 20 \
    --temperature 0.9 \
    ${@}