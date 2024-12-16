import openai
from ast import literal_eval
import json

def openai_chat_completion(model_engine="gpt-3.5-turbo", in_context=False, temperature = 1):
    chatLog = []
    save_ornot = ""
    if model_engine.startswith("gpt") or model_engine.startswith("text-davinci"):
        openai.api_base = "https://api.huiyan-ai.cn/v1"
        openai.api_key = "sk-wWuauIF02aZbs8tnFaF448D518Ac43149d09287147534e6a"
    elif model_engine.startswith("Wizard") or model_engine.startswith("llama") or model_engine.startswith("vicuna"):
        openai.api_base = "http://175.6.27.233:8000/v1"
        openai.api_key = ""
    else:
        print(f"[-]Model do not support: {model_engine}")
        return

    while True:
        try:
            user_input = input(">>>User: ")
            if in_context:
                chatLog.append({"role": "user", "content": user_input})
            else:
                chatLog = [{"role": "user", "content": user_input}]

            if model_engine.startswith("text-davinci"):
                prompt = "\n".join([f"{i['role']}: {i['content']}" for i in chatLog]) + "\nassistant: "
                chat = openai.Completion.create(model = model_engine, prompt = prompt, temperature = temperature)
            else:
                chat = openai.ChatCompletion.create(model = model_engine, messages = chatLog, temperature = temperature)
            if isinstance(chat, str):
                #print(type(chat))
                chat = json.loads(chat)
                #print(chat)
                #nchat = literal_eval(chat)
                #print(type(nchat))
                #print(nchat)
            if chat.get("error"):
                print("Error: ", chat["error"]['message'])
                # chatLog.append({"role": "error", "content": chat["error"]['message']})
                break
            elif chat.get("choices"):
                print("here")
                answer = chat["choices"][0]['message']['content'].lstrip()
                chatLog.append({"role": "assistant", "content": answer})
                print(f"\n>>>Assistant:\n{answer}\n{'-'*99}")
                print(f"{chat['usage']}\n{'='*99}")
        except KeyboardInterrupt:
            print("\n[*]Keyboard Interrupt")
            save_ornot = input(">>>Save conversation? (y/n): ")
            break
        except Exception as e:
            print(e)
            if chatLog:
                save_ornot = ""

if __name__ == '__main__':
    openai_chat_completion(model_engine="gpt-4o", in_context=False, temperature = 1)