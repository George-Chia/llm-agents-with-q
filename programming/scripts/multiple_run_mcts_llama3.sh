export OPENAI_KEY=sk-l8X7qhzVtthP3Hvd06F3Fb532dF044C09f351545C640F16f
export OPENAI_API_BASE=https://api.huiyan-ai.cn/v1
#!/bin/bash
num_workers=1
node_num=9
explore_model_name=llama31
exp_name=llama31-exploration

for ((j=0;j<${num_workers};j=j+1)); do
    part_idx=$((j))
    python main.py \
    --part_num ${num_workers} \
    --part_idx ${part_idx} \
    --run_name "mcts_llama31" \
    --root_dir "mcts_llama31" \
    --dataset_path ./benchmarks/humaneval-py.jsonl \
    --strategy "mcts" \
    --language "py" \
    --model ${explore_model_name}-${j} \
    --pass_at_k "1" \
    --max_iters "4" \
    --expansion_factor "3" \
    --number_of_tests "4" \
    --verbose &
    echo $! >> logs/${exp_name}-eval_pid.txt
done


#  --use_condition \