# Import necessary libraries
import torch
import os
import logging
from tqdm.notebook import tqdm  # Use notebook version of tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Manually set your parameters here
base_model_name_or_path = "/home/zhaiyuanzhao/llm/Meta-Llama-3.1-8B-Instruct"
peft_model_path = "webshop/trained_models/webshop-all_traj-step-16000"
output_dir = peft_model_path+"/merged_model"
device = "cuda:7"  # set to 'auto' or specify device like 'cuda:0'
push_to_hub = False  # set to True if you want to push to the Hugging Face Model Hub

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    if device == 'auto':
        device_arg = {'device_map': 'auto'}
    else:
        device_arg = {'device_map': {"": device}}

    logger.info(f"Loading base model: {base_model_name_or_path}")
    with tqdm(total=1, desc="Loading base model") as pbar:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            return_dict=True,
            torch_dtype=torch.float16,
            **device_arg
        )
        pbar.update(1)

    logger.info(f"Loading Peft: {peft_model_path}")
    with tqdm(total=1, desc="Loading Peft model") as pbar:
        model = PeftModel.from_pretrained(base_model, peft_model_path)
        pbar.update(1)

    logger.info("Running merge_and_unload")
    with tqdm(total=1, desc="Merge and Unload") as pbar:
        model = model.merge_and_unload()
        pbar.update(1)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    model.save_pretrained(f"{output_dir}")
    tokenizer.save_pretrained(f"{output_dir}")
    logger.info(f"Model saved to {output_dir}")

except Exception as e:
    logger.exception("An error occurred:")
    raise