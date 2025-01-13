import os
import json
import argparse
import logging

from models import gpt_usage
from mcts import fschat_simple_search, fschat_mcts_search, fschat_beam_search
from webshop import WebShopTask
from typing import List, Tuple, Any
from tqdm import tqdm
import transformers
import math
import random

# Modified from ETO by 
def load_idxs(split: str, part_num: int, part_idx: int = -1) -> Tuple[int]:
    if split == 'train':
        idxs = json.load(open("data_split/train_indices.json"))
    else:
        idxs = json.load(open("data_split/test_indices.json"))
    # idxs = idxs[:500]
    # random.shuffle(idxs)
    if part_num == 1:
        idxs = idxs
    else:
        assert part_idx != -1
        part_len = len(idxs) // part_num + 1
        idxs = idxs[part_len * part_idx: part_len * (part_idx + 1)]
    return idxs


# Configuring the logging
def run(args):
    task = WebShopTask()
    print(task)
    logs, cnt_avg, cnt_any = [], 0, 0
    
    logging.basicConfig(filename=args.log, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
    

    count = 0
    task_accs = []
    info = []
    # n = args.task_end_index
    idx = load_idxs(args.data_split, args.part_num, args.part_idx)
    if "Phi-3" in args.backend:
        trajectories_save_path = args.save_path+'_'+args.data_split+'_'+"Phi-3"+'_'+args.algorithm+'_'+str(args.iterations)+"iterations"
    else:
        trajectories_save_path = args.save_path+'_'+args.data_split+'_'+args.backend.split('-')[0]+'_T'+str(args.temperature)+'_'+args.algorithm+'_'+str(args.iterations)+"iterations"
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

    if args.algorithm == 'beam' or args.using_puct or args.enable_Q_value_model_for_critique:
        device = 'cuda:6'

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


    n=0
    # idx=[9680]
    for i in tqdm(idx):
    # for i in range(args.task_start_index, args.task_end_index):
        # solve
        if args.add_fixed_prefix:
            idx = f'fixed_{i}'
        else:
            idx = i

        if args.algorithm == 'simple':
            state, value, reward, em = fschat_simple_search(args, task, idx, args.iterations, True, trajectories_save_path,
                                                            dpo_policy_model, dpo_reference_model, tokenizer)
        elif args.algorithm == 'beam':
            state, value, reward, em = fschat_beam_search(args, task, idx, True, trajectories_save_path,
                                                            dpo_policy_model, dpo_reference_model, tokenizer)
        elif args.algorithm == 'mcts':
            state, value, reward, em = fschat_mcts_search(args, task, idx, args.iterations, True, trajectories_save_path,
                                                            dpo_policy_model, dpo_reference_model, tokenizer)
        elif args.algorithm == 'critique':  # step_level_critique
            state, value, reward, em = fschat_critique_search(args, task, idx, args.iterations, True, trajectories_save_path,
                                                            dpo_policy_model, dpo_reference_model, tokenizer)
            
         # log main metric
        # task_accs.append(em)
        print("best reward", reward)
        # cnt_avg = sum(task_accs) / len(task_accs)
        # print(i, 'len(task_accs)', len(task_accs), 'cnt_avg', cnt_avg, '\n')
        task_accs.append(reward)
        if (i+1) % 1 == 0:
            r, sr, fr = sum(task_accs) / len(task_accs), len([_ for _ in task_accs if _ == 1]) / len(task_accs), count / len(task_accs)
            print(i+1, r, sr, fr)
            print('-------------')
        n += 1
        r, sr, fr = sum(task_accs) / len(task_accs), len([_ for _ in task_accs if _ == 1]) / n, count / n
        print(r, sr, fr)

        logging.info(f"TASK RESULTS: {r}, {sr}, {fr}")
       
    # n = args.task_end_index - args.task_start_index
    print('usage_so_far', gpt_usage(args.backend))

def parse_args():
    args = argparse.ArgumentParser()
    # args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'llama2', "text-davinci-002",
    #                                                   'Phi-3-mini-4k-instruct-fastchat'], default='gpt-3.5-turbo-16k')
    args.add_argument('--backend', type=str, default='gpt-3.5-turbo-16k')
    args.add_argument('--temperature', type=float, default=1.0)
    args.add_argument('--data_split', type=str, default="test", help="Following ETO")
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
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--iterations', type=int, default=30)
    args.add_argument('--max_depth', type=int, default=15)
    args.add_argument('--rollout_width', type=int, default=5)
    args.add_argument('--log', type=str)
    args.add_argument('--save_path', type=str)
    args.add_argument('--algorithm', type=str, choices=['mcts', 'simple', 'fschat_simple', 'beam'], default='mcts')
    args.add_argument('--enable_value_evaluation', action='store_true')

    args.add_argument('--enable_fastchat_conv', action='store_true')
    args.add_argument('--enable_seq_mode', action='store_true')
    args.add_argument('--conv_template', type=str)
    args.add_argument('--q_model_conv_template', type=str)
    # args.add_argument('--agent_config_path', type=str)

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

    # for MCTS
    args.add_argument('--disable_early_stop', action='store_true')
    args.add_argument('--enable_rollout_early_stop', action='store_true')

    # various expansion_sampling_method for MCTS
    args.add_argument('--expansion_sampling_method', choices=['conditional', 'critique', 'vanilla'], default='vanilla')

    # for Critique
    args.add_argument('--critique_backend', type=str,  default=None)
    args.add_argument('--critique_prompt_template', type=str,  default=None)
    args.add_argument('--critique_temperature', type=float)
    args.add_argument('--enable_rollout_with_critique', action='store_true')
    args.add_argument('--enable_Q_value_model_for_critique', action='store_true')

    # webshop env
    args.add_argument('--add_fixed_prefix', action='store_true')


    args = args.parse_args()

    assert not (args.enable_fastchat_conv ^ args.enable_seq_mode) # 必须都开或者都不开
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)