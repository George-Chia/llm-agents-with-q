import os
import json
import argparse

from hotpotqa import HotPotQATask
from models import gpt_usage
from mcts import fschat_simple_search, fschat_mcts_search, fschat_beam_search, fschat_reflexion_search, fschat_lats_search
import logging

from typing import List, Tuple, Any
from tqdm import tqdm
import transformers
import math
import random

# Modified from ETO by 
def load_idxs(split: str, part_num: int, part_idx: int = -1, training_indices_path="data_split/train_indices.json") -> Tuple[int]:
    if split == 'train':
        idxs = json.load(open(training_indices_path))
    elif split == 'valid':
        idxs = json.load(open("data_split/valid_indices.json"))
    elif split == 'test':
        idxs = json.load(open("data_split/test_indices.json"))
    # random.shuffle(idxs)
    if part_num == 1:
        idxs = idxs
    else:
        assert part_idx != -1
        part_len = len(idxs) // part_num + 1
        idxs = idxs[part_len * part_idx: part_len * (part_idx + 1)]
    return idxs


def run(args):
    task = HotPotQATask()
    print(task)
    logs, cnt_avg, cnt_any = [], 0, 0

    # create log directories if they don't exist
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

    idx = load_idxs(args.data_split, args.part_num, args.part_idx, args.training_indices_path)
    if "Phi-3" in args.backend:
        trajectories_save_path = args.save_path+'_'+args.data_split+'_'+"Phi-3"+'_'+args.algorithm+'_'+str(args.iterations)+"iterations"
    else:
        trajectories_save_path = args.save_path+'_'+args.data_split+'_'+args.backend.split('-')[0]+'_'+args.algorithm+'_'+str(args.iterations)+"iterations"
    done_task_id = []
    if not os.path.exists(trajectories_save_path):
        os.makedirs(trajectories_save_path)
    else:
        for file in os.listdir(trajectories_save_path):
            if not file.endswith('json'):
                continue
            done_task_id.append(int(file.split('.')[0]))
        logging.info(f"Existing trajectories_save_path file found. {len(done_task_id)} tasks done.")
    idx = [x for x in idx if x not in done_task_id]


    if args.algorithm == 'beam' or args.using_puct:
        device = 'cuda'

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.policy_model_name_or_path,
            padding_side='right',
            use_fast=False,
        )
        
        config = transformers.AutoConfig.from_pretrained(
            args.policy_model_name_or_path,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        config.use_cache = False

        # Load model and tokenizer
        dpo_policy_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.policy_model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
            # attn_implementation="flash_attention_2",
        ).to(device)

        dpo_reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.reference_model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
            # attn_implementation="flash_attention_2",
        ).to(device)
    else:
        dpo_policy_model = None
        dpo_reference_model = None
        tokenizer = None

    count = 0
    task_accs = []
    info = []

    for i in tqdm(idx):
    # for i in range(args.task_start_index, args.task_end_index):
        # solve
        if args.enable_fastchat_conv:
            if args.algorithm == 'simple':
                state, value, reward, em = fschat_simple_search(args, task, i, args.iterations, True, trajectories_save_path, 
                                                              enable_reflection=args.enable_reflection)
            elif args.algorithm == 'beam':
                state, value, reward, em = fschat_beam_search(args, task, i, True, trajectories_save_path,
                                                              dpo_policy_model, dpo_reference_model, tokenizer, 
                                                              enable_reflection=args.enable_reflection)
            elif args.algorithm == 'mcts':
                state, value, reward, em = fschat_mcts_search(args, task, i, args.iterations, True, trajectories_save_path,
                                                              dpo_policy_model, dpo_reference_model, tokenizer, 
                                                              enable_reflection=args.enable_reflection)
            elif args.algorithm == 'reflexion':    # reflexion
                state, value, reward, em = fschat_reflexion_search(args, task, i, args.iterations, True,
                                                                trajectories_save_path, args.refine_num)
            elif args.algorithm == 'lats':
                state, value, reward, em = fschat_lats_search(args, task, i, args.iterations, True,
                                                              trajectories_save_path)

        else:
            if args.algorithm == 'mcts':
                state, value, all_nodes, reward, em = mcts_search(args, task, i, args.iterations, True)
            elif args.algorithm == 'tot':
                state, value, all_nodes, reward, em = dfs_search(args, task, i, args.iterations)
            elif args.algorithm == 'rap':
                state, value, all_nodes, reward, em = mcts_search(args, task, i, args.iterations)
            elif args.algorithm == 'simple':
                state, value, all_nodes, reward, em = simple_search(args, task, i, args.iterations)
            else:
                raise Exception("Search algorithm option not valid")
         # log main metric
        if em is None:
            em = 0
        task_accs.append(em)
        cnt_avg = sum(task_accs) / len(task_accs)
        print(i, 'len(task_accs)', len(task_accs), 'cnt_avg', cnt_avg, '\n')
        #all_nodes_dict = [(node.to_dict(), value) for node, value in all_nodes]
        
       
    # n = args.task_end_index - args.task_start_index
    print('usage_so_far', gpt_usage(args.backend))

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str,  default='gpt-4o-mini')
    args.add_argument('--temperature', type=float, default=1.0)
    args.add_argument('--data_split', type=str, default="test", help="Following ETO")
    args.add_argument('--training_indices_path', type=str, default="data_split/train_indices.json")
    args.add_argument(
        "--part_num",
        type=int,
        default=1,
    )
    args.add_argument(
        "--part_idx",
        type=int,
        default=-1,
    )
    # args.add_argument('--task_start_index', type=int, default=900)
    # args.add_argument('--task_end_index', type=int, default=1000)
    args.add_argument('--refine_num', type=int, default=1)
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])
    args.add_argument('--n_generate_sample', type=int, default=1)  
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--iterations', type=int, default=50)
    args.add_argument('--log', type=str)
    args.add_argument('--algorithm', type=str, choices=['mcts', 'simple', 'fschat_simple', 'beam', 'refine', 'lats'], default='mcts')

    args.add_argument('--max_depth', type=int, default=7)
    args.add_argument('--rollout_width', type=int, default=1)
    args.add_argument('--save_path', type=str)
    args.add_argument('--enable_value_evaluation', action='store_true')

    args.add_argument('--enable_fastchat_conv', action='store_true')
    args.add_argument('--enable_seq_mode', action='store_true')
    args.add_argument('--conv_template', type=str)
    args.add_argument('--critique_conv_template', type=str)
    args.add_argument('--q_model_conv_template', type=str)
    args.add_argument('--enable_reflection', action='store_true')

    # for MCTS
    args.add_argument('--disable_early_stop', action='store_true')
    args.add_argument('--enable_rollout_early_stop', action='store_true')


    # for calculating DPO logits
    args.add_argument(
        "--policy_model_name_or_path",
        type=str,
        help="Config path of tokenizer",
    )
    args.add_argument(
        "--reference_model_name_or_path",
        type=str,
        help="Config path of tokenizer",
    )
    args.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Config path of tokenizer",
    )
    args.add_argument(
        "--model_max_length",
        type=int,
        default=4096,
        help="Maximum sequence length. Sequences will be right padded (and possibly truncated).",
    )
    args.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
    )

    # for puct
    args.add_argument('--using_puct', action='store_true')
    args.add_argument('--puct_coeff', type=float, default=0.)
    args.add_argument('--enable_rollout_with_q', action='store_true')

    # various expansion_sampling_method for MCTS
    args.add_argument('--expansion_sampling_method', choices=['conditional', 'critique', 'vanilla', 'memory', 'lats'], default='vanilla')

    # for Critique
    args.add_argument('--critique_backend', type=str,  default=None)
    args.add_argument('--critique_prompt_template', type=str,  default=None)
    args.add_argument('--critique_temperature', type=float)
    args.add_argument('--enable_rollout_with_critique', action='store_true')

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)