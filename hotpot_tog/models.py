import os
import openai
import backoff 
from transformers import GPT2Tokenizer
import warnings

from models_fastchat import fschat_instruct_conv

completion_tokens = prompt_tokens = 0
MAX_TOKENS = 4000

# 修改了地址
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

def tokens_in_text(text):
    """
    Accurately count the number of tokens in a string using the GPT-2 tokenizer.
    
    :param text: The input text.
    :return: The exact number of tokens in the text.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        tokenizer = GPT2Tokenizer.from_pretrained(LOCAL_LLM_PATH+'/gpt2-medium')
        tokens = tokenizer.encode(text)
    return len(tokens)

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base
# key 读取失败，显式设置
api_key= "sk-F29EYhchwPz1aEmoRu0U7W2IQL7vRxjHyFKkPV9irCOteTeB"
openai.api_base= "https://api.huiyan-ai.cn/v1"
@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-3.5-turbo-16k", temperature=1.0, max_tokens=128, n=1, stop=None, enable_fastchat_conv=False) -> list:
    if enable_fastchat_conv:
        if "gpt-" in model:
            return chatgpt(prompt, model, temperature, max_tokens, n, stop)
        else:
            return fschat_instruct_conv(prompt, model, temperature, max_tokens, n, stop) # max_tokens=100 already means max_new_tokens

    else:
        if model == "test-davinci-002":
            return gpt3(prompt, model, temperature, max_tokens, n, stop)
        elif "Phi-3" in model:
            return phi3_instruct(prompt, model, temperature, max_tokens, n, stop) # max_tokens=100 already means max_new_tokens   
        else:
            messages = [{"role": "user", "content": prompt}]
            return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    result = os.system('proxy_on')
     
def chatgpt(messages, model="gpt-3.5-turbo", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    # print('Model: ', model)
    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-3.5-turbo-16k":
        cost = completion_tokens / 1000 * 0.004 + prompt_tokens / 1000 * 0.003
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens}
