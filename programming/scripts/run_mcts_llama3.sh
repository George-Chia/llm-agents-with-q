export OPENAI_KEY=sk-l8X7qhzVtthP3Hvd06F3Fb532dF044C09f351545C640F16f
export OPENAI_API_BASE=https://api.huiyan-ai.cn/v1
python main.py \
  --run_name "mcts_llama31" \
  --root_dir "mcts_llama31" \
  --dataset_path ./benchmarks/humaneval-py.jsonl \
  --strategy "mcts" \
  --language "py" \
  --model "llama31-0" \
  --pass_at_k "1" \
  --max_iters "4" \
  --expansion_factor "3" \
  --number_of_tests "4" \
  --verbose &
pid=$!
wait $pid

python main.py \
  --use_condition \
  --run_name "cd_mcts_llama31" \
  --root_dir "cd_mcts_llama31" \
  --dataset_path ./benchmarks/humaneval-py.jsonl \
  --strategy "mcts" \
  --language "py" \
  --model "llama31-0" \
  --pass_at_k "1" \
  --max_iters "4" \
  --expansion_factor "3" \
  --number_of_tests "4" \
  --verbose &