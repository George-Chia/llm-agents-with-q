#!/bin/bash

python run.py \
    --backend gpt-3.5-turbo \
    --n_generate_sample 5 \
    --save_path trajectories \
    --log logs/gpt-4o-mini.log \
    --max_depth 5 \
    ${@}