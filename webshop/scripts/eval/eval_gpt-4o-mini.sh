

python run.py \
    --backend gpt-4o-mini \
    --data_split test \
    --n_generate_sample 1 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 1 \
    --save_path trajectories \
    --log logs/eval-original-gpt-4o-mini.log \
    --max_depth 10 \
    --algorithm simple \
    ${@}