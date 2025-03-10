# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from dataclasses import dataclass, field
import json
import re
import math
import pathlib
import sys
import os
import eval_agent.agents as agents
# import wandb
from torch.nn.modules.module import T

os.environ["WANDB_MODE"] = "offline"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
sys.path.append('/home/huan/works/ETO/')
from typing import Dict, Optional, Sequence

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training
)
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle, get_conv_template
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
local_rank = None

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)

        return (loss, outputs) if return_outputs else loss

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    reg_weight: float = field(
        default=0.1,
    )
    gamma: float = field(
        default=0.95,
    )
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )



def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()

def get_phi3_turns(text):
    positions = []
    for match in re.finditer(r'<\|end\|>\n', text):
        positions.append(match.start())
    assert len(positions) % 2 == 0
    turns = []
    cur_ind = 0
    for i in range(len(positions)):
        if i % 2 == 1:
            turns.append(text[cur_ind:positions[i]])
            cur_ind = positions[i] + 8
    return turns

def get_gemma2_turns(text):
    positions = []
    for match in re.finditer(r'<end_of_turn>\n', text):
        positions.append(match.start())
    assert len(positions) % 2 == 0
    turns = []
    cur_ind = 0
    for i in range(len(positions)):
        if i % 2 == 1:
            turns.append(text[cur_ind:positions[i]])
            cur_ind = positions[i] + 14
    return turns

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    model_path: str,
) -> Dict:

    if 'phi-3' in model_path.lower():
        conv = get_conv_template("phi-3")
    elif 'gemma-2' in model_path.lower():
        conv = get_conv_template("gemma-2")
    else:
        conv = get_model_adapter(model_path).get_default_conv_template(model_path)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # new_conversations = []
    # for i, source in enumerate(sources):
    #     tmp_list = []
    #     if roles[source[0]["from"]] != conv.roles[0]:
    #         # Skip the first one if it is not from human
    #         source = source[1:]
    #     for j, sentence in enumerate(source):
    #         role = sentence['from']
    #         if role == 'human':
    #             role = 'user'
    #         if role == 'gpt':
    #             role = 'assistant'
    #         tmp_list.append({
    #                 "role": role, "content": sentence['value']
    #         })
    #     new_conversations.append(tmp_list)

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # debug
    conversations = conversations[:60]

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    if conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
        sep = conv.sep + conv.roles[1] + ": "
    elif conv.sep_style == SeparatorStyle.LLAMA2:
        sep = conv.sep + conv.roles[1] + " "
    elif conv.sep_style == SeparatorStyle.PHI3:
        sep = conv.roles[1] + "\n"
    elif conv.sep_style == SeparatorStyle.GEMMA2:
        sep = conv.sep + "\n"
    else:
        raise NotImplementedError

    # Mask targets. Only compute loss on the assistant outputs.
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        if conv.sep_style == SeparatorStyle.PHI3:
            turns = get_phi3_turns(conversation)
        elif conv.sep_style == SeparatorStyle.GEMMA2:
            turns = get_gemma2_turns(conversation)
        else:
            turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            # remove <s>
            turn_len = len(tokenizer(turn).input_ids) - 1
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # remove <s> and the "_" in the end
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            if conv.sep_style == SeparatorStyle.PHI3:
                instruction_len += 1
            elif conv.sep_style == SeparatorStyle.GEMMA2:
                instruction_len += 1
            # magic number for vicuna, since different subtoken for "USER"
            if i != 0 and conv.roles[0] == 'USER':
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID

            # add the length of turn sep
            if conv.sep2 == '</s>':
                cur_len += turn_len + 1
            elif conv.sep2 == ' </s><s>':
                cur_len += turn_len + 3
            elif conv.sep_style == SeparatorStyle.PHI3:
                cur_len += turn_len + 1
            elif conv.sep_style == SeparatorStyle.GEMMA2:
                cur_len += turn_len + 2
            else:
                raise NotImplementedError

        if conv.sep_style == SeparatorStyle.PHI3:
            cur_len += 1
        elif conv.sep_style == SeparatorStyle.GEMMA2:
            cur_len += 1
        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(conversation)
            rank0_print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_path: str = None):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, model_path)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_path: str = None):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.model_path = model_path

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.model_path)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, model_path: str = None
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, model_path=model_path)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, model_path=model_path)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    with open(os.path.join(os.getcwd(), f"eval_agent/configs/model/fc_agent1.json")) as f:
        agent_config1: Dict[str, Any] = json.load(f)
    with open(os.path.join(os.getcwd(), f"eval_agent/configs/model/fc_agent2.json")) as f:
        agent_config2: Dict[str, Any] = json.load(f)
    # initialize the agent
    agent1: agents.LMAgent = getattr(agents, agent_config1["agent_class"])(
        agent_config1["config"]
    )
    agent2: agents.LMAgent = getattr(agents, agent_config2["agent_class"])(
        agent_config2["config"]
    )
    # wandb_config = {}
    # for attr, value in model_args.__dict__.items():
    #     if not attr.startswith('__') and 'deepspeed' not in attr.lower():
    #         wandb_config[attr] = value
    # for attr, value in training_args.__dict__.items():
    #     if not attr.startswith('__') and 'deepspeed' not in attr.lower():
    #         wandb_config[attr] = value
    # wandb.init(project='inverse Q',
    #            name='webshop',
    #            config=wandb_config,
    #            resume='None')
    local_rank = training_args.local_rank
    rank0_print(f'save output: {training_args.output_dir}.')

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False
    # Load model and tokenizer
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16
    )
    # if 'phi-3' in model_args.model_name_or_path.lower() and 'sft' in model_args.model_name_or_path.lower():
    #         model.lm_head.load_state_dict(torch.load(f'{model_args.model_name_or_path}/lm_head.pt'))
    #         model.base_model.norm.load_state_dict(torch.load(f'{model_args.model_name_or_path}/norm.pt'))
    #         print('Successful load lm_head and norm.')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_path=model_args.model_name_or_path)

    # Start trainner
    trainer = CustomTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.save_model()
    # trainer = Trainer(
    #     model=model, tokenizer=tokenizer, args=training_args, **data_module
    # )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    # # Save model
    model.config.use_cache = True
    trainer.save_state()
    trainer.save_model()
    torch.save(model.lm_head.state_dict(), f'{training_args.output_dir}/lm_head.pt')
    torch.save(model.base_model.norm.state_dict(), f'{training_args.output_dir}/norm.pt')

if __name__ == "__main__":
    train()
