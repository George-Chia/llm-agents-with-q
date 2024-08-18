python run.py \
    --backend Phi-3-0 \
    --data_split test \
    --n_generate_sample 1 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 1 \
    --save_path trajectories \
    --log logs/eval-Phi-3.log \
    --max_depth 10 \
    --algorithm simple \
    --enable_fastchat_conv \
    --enable_seq_mode \
    --conv_template phi3 \
    ${@}