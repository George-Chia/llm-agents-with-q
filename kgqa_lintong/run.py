import os
import json
import argparse
import concurrent.futures
from hotpotqa import HotPotQATask
from models import gpt_usage
from mcts import fschat_simple_search, fschat_mcts_search, fschat_beam_search
import logging

from typing import List, Tuple, Any
from tqdm import tqdm
import transformers
import math
import traceback
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


def run_search(args, task, i, trajectories_save_path, dpo_policy_model, dpo_reference_model, tokenizer):
    if args.enable_fastchat_conv:
        if args.algorithm == 'simple':
            state, value, reward, em = fschat_simple_search(args, task, i, args.iterations, True,
                                                            trajectories_save_path,
                                                            enable_reflection=args.enable_reflection)
        elif args.algorithm == 'beam':
            state, value, reward, em = fschat_beam_search(args, task, i, True, trajectories_save_path,
                                                          dpo_policy_model, dpo_reference_model, tokenizer,
                                                          enable_reflection=args.enable_reflection)
        elif args.algorithm == 'mcts':
            state, value, reward, em = fschat_mcts_search(args, task, i, args.iterations, True, trajectories_save_path,
                                                          dpo_policy_model, dpo_reference_model, tokenizer,
                                                          enable_reflection=args.enable_reflection)
    else:
        raise Exception("Fastchat conversation is required for this implementation.")

    return state, value, reward, em


def run(args):
    task = HotPotQATask()
    print(task)
    logs, cnt_avg, cnt_any = [], 0, 0

    # create log directories if they don't exist
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

    #idx = load_idxs(args.data_split, args.part_num, args.part_idx, args.training_indices_path)
    idx = random.sample(range(0, 1000), 100)
    #idx = [1,5,10,15,20,25,30,35,40,45,50,51,60,65,70,75,80,85,90,95,100]
    #idx =[51, 169, 597, 898, 608, 409, 47, 547, 683, 48, 543, 586, 605, 324, 361, 161, 17, 247, 280, 981, 647, 579, 125, 583, 664, 340, 938, 390, 506, 723, 871, 412, 99, 855, 174, 440, 68, 288, 341, 523, 761, 825, 522, 787, 102, 595, 811, 498, 606, 327, 533, 208, 864, 268, 828, 709, 824, 254, 493, 94, 604, 974, 578, 673, 32, 255, 742, 225, 135, 439, 520, 297, 707, 265, 74, 725, 750, 516, 567, 290, 211, 686, 580, 248, 995, 198, 482, 277, 257, 189, 885, 69, 15, 317, 76, 266, 719, 784, 758, 90]
    print(idx)
    if "Phi-3" in args.backend:
        trajectories_save_path = args.save_path+'_'+args.data_split+'_'+"Phi-3"+'_'+args.algorithm+'_'+str(args.iterations)+"iterations"
    else:
        trajectories_save_path = args.save_path+'_'+args.data_split+'_'+args.backend.split('-')[0]+'_'+args.algorithm+'_'+str(args.iterations)+'_'+str(args.kgqa_dataset)+"iterations"
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

    # 设置索引
    # idx = [1,5,10,15,20,25,30,35,40,45,50,51,60,65,70,75,80,85,90,95,100]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_search, args, task, i, trajectories_save_path, dpo_policy_model, dpo_reference_model, tokenizer) for i in idx]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                state, value, reward, em = future.result()
                # log main metric
                #if em is None:
                    #em = 0
                #cnt_avg = sum(task_accs) / len(task_accs)
                #print(i, 'len(task_accs)', len(task_accs), 'cnt_avg', cnt_avg, '\n')
            except Exception as e:
                logging.warning(f"Error processing task {e}")
                logging.error(traceback.format_exc())
                print(traceback.format_exc())

    # n = args.task_end_index - args.task_start_index
    print('usage_so_far', gpt_usage(args.backend))

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str,  default='gpt-3.5-turbo')
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
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'],default='cot')
    args.add_argument('--n_generate_sample', type=int, default=5)
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--iterations', type=int, default=5)
    args.add_argument('--log', type=str,default='logs/gpt-4.log')
    args.add_argument('--algorithm', type=str, choices=['mcts', 'rap', 'tot', 'simple', 'beam'],default='mcts')

    # 图谱中的搜索深度
    args.add_argument('--max_depth', type=int, default=3)
    #增加搜索宽度（没用到）
    args.add_argument('--search_width', type=int, default=3)
    #设置每次剪枝后的实体数（没用到）
    # args.add_argument('--width', type=int, default=3)
    args.add_argument('--rollout_width', type=int, default=1)
    args.add_argument('--save_path', type=str,default='trajectories')
    args.add_argument('--enable_value_evaluation', action='store_true')

    args.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    args.add_argument("--max_length", type=int,
                        default=2048, help="the max length of LLMs output.")
    args.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    args.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    args.add_argument("--width", type=int,    # --n_generate_sample
                        default=3, help="choose the search width of ToG.")
    # args.add_argument("--depth", type=int,    # --max_depth
    #                     default=1, help="choose the search depth of ToG.")
    args.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    args.add_argument("--LLM_type", type=str,
                        default="gpt-4o-mini", help="base LLM model.")
    args.add_argument("--opeani_api_keys", type=str,
                        default="sk-nPQVAFBDhZoMmYEnPPxYKk0p86jfCMxyQaqnCLV5qKq0XHxK",
                        help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    args.add_argument("--num_retain_entity", type=int,
                        default=1, help="Number of entities retained during entities search.")
    args.add_argument("--prune_tools", type=str,
                        default="bm25", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")

    args.add_argument('--enable_fastchat_conv', action='store_true',default=True)
    args.add_argument('--enable_seq_mode', action='store_true')
    args.add_argument('--conv_template', type=str)
    args.add_argument('--critique_conv_template', type=str)
    args.add_argument('--q_model_conv_template', type=str)
    args.add_argument('--enable_reflection', action='store_true')

    # for MCTS
    args.add_argument('--disable_early_stop',default=True)
    args.add_argument('--enable_rollout_early_stop', default=False)


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
    args.add_argument('--expansion_sampling_method', choices=['conditional', 'critique', 'vanilla'], default='vanilla')

    # for Critique
    args.add_argument('--critique_backend', type=str,  default=None)
    args.add_argument('--critique_prompt_template', type=str,  default=None)
    args.add_argument('--critique_temperature', type=float)
    args.add_argument('--enable_rollout_with_critique', default=False)

    # for dataset
    args.add_argument('--kgqa_dataset', choices=['cwq', 'web_qsp', 'SimpleQA','WebQuestions','grail_qa'], default='grailqa')
    args.add_argument('--enable_wiki_search',default=True)

    # max_workers
    args.add_argument('--max_workers', type=int, default=40, help="Number of parallel workers")

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)