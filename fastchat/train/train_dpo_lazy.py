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
import math
import pathlib
from typing import Dict, Optional, Sequence
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset

# from fastchat.train.llama2_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )

# replace_llama_attn_with_flash_attn()

import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
# from trl import DPOTrainer
from fastchat.train.dpo_trainer import DPOMultiTrainer
from datasets import load_dataset

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter, get_conv_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    ref_model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    conv_template: Optional[str] = field(default="")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})


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
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_prompt_length: int = field(
        default=512,
        metadata={
            "help": "Maximum target length."
        },
    )
    max_target_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum target length."
        },
    )


local_rank = None


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


def mask_labels(conversation, target, tokenizer, conv):
    if conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
        sep = conv.sep + conv.roles[1] + ": "
    elif conv.sep_style == SeparatorStyle.LLAMA2:
        sep = conv.sep + conv.roles[1] + " "
    elif conv.sep_style == SeparatorStyle.PHI3:
        sep = conv.sep + '\n' + conv.roles[1] + '\n'
    else:
        raise NotImplementedError
    
    total_len = int(target.ne(tokenizer.pad_token_id).sum())

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

        # magic number for vicuna, since different subtoken for "USER"
        if i != 0 and conv.roles[0] == 'USER':
            # The legacy and non-legacy modes handle special tokens differently
            instruction_len -= 1

        # Ignore the user instructions
        target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

        # add the length of turn sep
        if conv.sep2 == '</s>':
            cur_len += turn_len + 1 
        elif conv.sep2 == ' </s><s>':
            cur_len += turn_len + 3
        else:
            raise NotImplementedError
        
        # magic number for vicuna, since different subtoken for "USER"
        if i != 0 and conv.roles[0] == 'USER':
            # The legacy and non-legacy modes handle special tokens differently
            cur_len -= 1

    target[cur_len:] = IGNORE_TOKEN_ID

    if False:  # Inspect and check the correctness of masking
        z = target.clone()
        z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
        rank0_print(conversation)
        rank0_print(tokenizer.decode(z))
        exit()

    if cur_len < tokenizer.model_max_length:
        if cur_len != total_len:
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(conversation)
            print("#" * 50)
            rank0_print(tokenizer.decode(z))
            target[:] = IGNORE_TOKEN_ID
            rank0_print(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                f" #turn = {len(turns) - 1}. (ignored)"
            )

    return target


def preprocess_multi_turn(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    model_path: str,
    conv_template: str
) -> Dict:
    if conv_template is not None:
        conv = get_conv_template(conv_template)
    else:
        conv = get_model_adapter(model_path).get_default_conv_template(model_path)

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conv.messages = []
    for j, sentence in enumerate(source['prompt']):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2]
        conv.append_message(role, sentence["value"])
    prompt = conv.get_prompt()

    conv.messages = []
    for j, sentence in enumerate(source['prompt'] + source['chosen']):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2]
        conv.append_message(role, sentence["value"])
    chosen = conv.get_prompt()

    conv.messages = []
    for j, sentence in enumerate(source['prompt'] + source['rejected']):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2]
        conv.append_message(role, sentence["value"])
    rejected = conv.get_prompt()

    # Tokenize conversations
    prompt_tokens = tokenizer(prompt, return_tensors="pt")

    chosen_tokens = tokenizer(chosen, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True)
    # chosen_tokens = tokenizer(chosen, return_tensors="pt")
    chosen_labels = chosen_tokens.input_ids[0].clone()
    chosen_labels = mask_labels(chosen, chosen_labels, tokenizer, conv)
    chosen_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

    rejected_tokens = tokenizer(rejected, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True)
    rejected_labels = rejected_tokens.input_ids[0].clone()
    rejected_labels = mask_labels(rejected, rejected_labels, tokenizer, conv)
    rejected_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

    if False:  # Inspect and check the correctness of masking
        z = chosen_labels.clone()
        z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
        rank0_print(chosen)
        rank0_print(tokenizer.decode(z))
        z = rejected_labels.clone()
        z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
        rank0_print(rejected)
        rank0_print(tokenizer.decode(z))
        exit()

    return dict(
        chosen_input_ids=chosen_tokens['input_ids'][0].tolist(),
        chosen_attention_mask=chosen_tokens['attention_mask'][0].tolist(),
        chosen_labels=chosen_labels.tolist(),
        rejected_input_ids=rejected_tokens['input_ids'][0].tolist(),
        rejected_attention_mask=rejected_tokens['attention_mask'][0].tolist(),
        rejected_labels=rejected_labels.tolist(),
        prompt_input_ids=prompt_tokens['input_ids'][0].tolist(),
        prompt_attention_mask=prompt_tokens['attention_mask'][0].tolist(),
    )


class DPODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_path, conv_template):
        super(DPODataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = raw_data
        # chosens = [example["chosen"] for example in raw_data]
        # rejecteds = [example["rejected"] for example in raw_data]
        data_dict = preprocess_multi_turn(sources, tokenizer, model_path, conv_template)

        self.chosen_input_ids = data_dict["chosen_input_ids"]
        self.chosen_labels = data_dict["chosen_labels"]
        self.chosen_attention_mask = data_dict["chosen_attention_mask"]
        self.rejected_input_ids = data_dict["rejected_input_ids"]
        self.rejected_labels = data_dict["rejected_labels"]
        self.rejected_attention_mask = data_dict["rejected_attention_mask"]
        self.rejected_input_ids = data_dict["rejected_input_ids"]
        self.prompt_input_ids = data_dict["prompt_input_ids"]
        self.prompt_attention_mask = data_dict["prompt_attention_mask"]

    def __len__(self):
        return len(self.prompt_input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            chosen_input_ids=self.chosen_input_ids[i],
            chosen_labels=self.chosen_labels[i],
            chosen_attention_mask=self.chosen_attention_mask[i],
            rejected_input_ids=self.rejected_input_ids[i],
            rejected_labels=self.rejected_labels[i],
            rejected_attention_mask=self.rejected_attention_mask[i],
            prompt_input_ids=self.prompt_input_ids[i],
            prompt_attention_mask=self.prompt_attention_mask[i],
        )



class LazyDPODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_path, conv_template):
        super(LazyDPODataset, self).__init__()

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.model_path=model_path
        self.conv_template=conv_template


    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess_multi_turn(self.raw_data[i], self.tokenizer, self.model_path, self.conv_template)
        ret = dict(
            chosen_input_ids=ret["chosen_input_ids"][0],
            chosen_labels=ret["chosen_labels"][0],
            chosen_attention_mask=ret["chosen_attention_mask"][0],
            rejected_input_ids=ret["rejected_input_ids"][0],
            rejected_labels=ret["rejected_labels"][0],
            rejected_attention_mask=ret["rejected_attention_mask"][0],
            prompt_input_ids=ret["prompt_input_ids"][0],
            prompt_attention_mask=ret["prompt_attention_mask"][0],
        )
        self.cached_data_dict[i] = ret
        return ret

def make_dpo_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, model_path, conv_template
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazyDPODataset if data_args.lazy_preprocess else DPODataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, model_path=model_path, conv_template=conv_template)

    # if data_args.eval_data_path:
    #     eval_json = json.load(open(data_args.eval_data_path, "r"))
    #     eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    # else:
    #     eval_dataset = None

    # return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
    return dict(train_dataset=train_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="flash_attention_2",
    )

    model_ref = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.ref_model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="flash_attention_2",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    # Load data
    # dataset = load_dataset("json", data_files=data_args.data_path)
    # preprocess = partial(preprocess_multi_turn, tokenizer=tokenizer, model_path=model_args.model_name_or_path, conv_template=model_args.conv_template)
    # train_dataset = dataset["train"].map(preprocess)

    # Load data
    data_module = make_dpo_data_module(tokenizer=tokenizer, data_args=data_args, 
                                       model_path=model_args.model_name_or_path, conv_template=model_args.conv_template)

    # Start trainner
    trainer = DPOMultiTrainer(
        model,
        model_ref,
        args=training_args,
        beta=model_args.beta,
        # train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=training_args.model_max_length,
        max_target_length=training_args.max_target_length,
        max_prompt_length=training_args.max_prompt_length,
        generate_during_eval=True,
        **data_module
    )

    # trainer.ref_model = trainer.accelerator.prepare(trainer.ref_model)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()
