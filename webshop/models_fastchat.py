import argparse
import json
import time
import requests

from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import get_conv_template

from requests.exceptions import Timeout, ConnectionError


import openai
import backoff


def get_worker_address(model_name, controller_address="http://0.0.0.0:21001"):
    controller_addr = controller_address
    ret = requests.post(controller_addr + "/refresh_all_workers")
    ret = requests.post(controller_addr + "/list_models")
    models = ret.json()["models"]
    models.sort()
    # print(f"Use Model: {model_name} from {models}")

    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]

    if worker_addr == "":
        print(f"No available workers for {model_name}")
        raise ValueError
    return worker_addr

def get_response(worker_addr, gen_params):
    headers = {"User-Agent": "FastChat Client"}
    for _ in range(3):
        try:
            response = requests.post(
                worker_addr + "/worker_generate_stream",
                headers=headers,
                json=gen_params,
                stream=True,
                timeout=120,
            )
            text = ""
            for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if line:
                    data = json.loads(line)
                    if data["error_code"] != 0:
                        assert False, data["text"]
                    text = data["text"]
            return text
        # if timeout or connection error, retry
        except Timeout:
            print("Timeout, retrying...")
        except ConnectionError:
            print("Connection error, retrying...")
        time.sleep(5)
    else:
        raise Exception("Timeout after 3 retries.")

def phi3_instruct(prompt, model, temperature, max_new_tokens, n, stop)  -> list :
    worker_addr = get_worker_address(model)
    conv = get_conv_template('phi3')
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    gen_params = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": 0.9,
        "max_new_tokens": max_new_tokens,
        "stop": stop,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
    response_list = []
    for _ in range(n):
        response = get_response(worker_addr, gen_params)
        response = response.replace('\nAction: ', "")
        response = response.replace('Action: ', "")
        response = response.replace('\n', "")
        response_list.append(response)
    return response_list


def fschat_instruct_conv(conv, model, temperature, max_new_tokens, n, stop)  -> list :
    worker_addr = get_worker_address(model)
    # conv = get_conv_template('phi3')
    # conv.append_message(conv.roles[0], prompt)
    # conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    gen_params = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": 0.9,
        "max_new_tokens": max_new_tokens,
        "stop": stop,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
    response_list = []
    for _ in range(n):
        response = get_response(worker_addr, gen_params)
        # response = response.replace('\nAction: ', "")
        # response = response.replace('Action: ', "")
        # response = response.replace('\n', "")

        response_list.append(response)
    return response_list

@backoff.on_exception(
        backoff.fibo,
        # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        (
            openai.error.APIError,
            openai.error.Timeout,
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.APIConnectionError,
        ),
    )
def gpt_instruct_conv(messages, model, temperature, max_new_tokens, n, stop)  -> list :
    openai.api_base = "https://api.huiyan-ai.cn/v1"
    openai.api_key = "sk-sWtiCQni9ZuDezwF863aC4C42b6a461884Fe54B9Ee8dD3Fa"
    response_list = []
    for _ in range(n):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=stop,
        )
        return response.choices[0].message["content"]
    return response_list