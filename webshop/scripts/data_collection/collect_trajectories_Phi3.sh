python run.py \
    --backend Phi-3-0 \
    --data_split test \
    --n_generate_sample 5 \
    --temperature 1.0 \
    --iterations 30 \
    --log logs/collect_trajectories.log \
    --save_path trajectories \
    --max_depth 10 \
    --rollout_width 1 \
    --algorithm mcts \
    --enable_value_evaluation False \
    --enable_fastchat_conv True \
    --enable_seq_mode True \
    --conv_template phi3 \
    ${@}