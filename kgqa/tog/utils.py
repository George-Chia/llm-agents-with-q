
import json
import time
import openai
import re
from tog.prompt_list import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from models_fastchat import llama31_instruct

openai.api_base = "https://api.huiyan.chat/v1"

def retrieve_top_docs(query, docs, model, width=3):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query (str): The input query.
    - docs (list of str): The list of documents to search from.
    - model_name (str): The name of the SentenceTransformer model to use.
    - width (int): The number of top documents to return.

    Returns:
    - list of float: A list of scores for the topn documents.
    - list of str: A list of the topn documents.
    """

    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores


def compute_bm25_similarity(query, corpus, width=3):
    """
    Computes the BM25 similarity between a question and a list of relations,
    and returns the topn relations with the highest similarity along with their scores.

    Args:
    - question (str): Input question.
    - relations_list (list): List of relations.
    - width (int): Number of top relations to return.

    Returns:
    - list, list: topn relations with the highest similarity and their respective scores.
    """

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)
    # 修改打分，直接返回全部结构
    '''
    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    doc_scores = sorted(doc_scores, reverse=True)[:width]
    return relations, doc_scores
    '''
    scored_relations = list(zip(corpus, doc_scores))
    scored_relations.sort(key=lambda x: x[1], reverse=True)
    relations, scores = zip(*scored_relations)
    return list(relations), list(scores)


def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations

#原来的runllm运行失败，暂时没找到原因，换一个自己写的

def openai_chat_completion(model_engine="gpt-3.5-turbo", in_context=False, temperature = 1,prompt=""):
    chatLog = []
    save_ornot = ""
    if model_engine.startswith("gpt") or model_engine.startswith("text-davinci"):
        openai.api_base = "https://api.huiyan.chat/v1"
        openai.api_key = "sk-F29EYhchwPz1aEmoRu0U7W2IQL7vRxjHyFKkPV9irCOteTeB"
    elif model_engine.startswith("Wizard") or model_engine.startswith("llama") or model_engine.startswith("vicuna"):
        openai.api_base = "http://175.6.27.233:8000/v1"
        openai.api_key = "sk-F29EYhchwPz1aEmoRu0U7W2IQL7vRxjHyFKkPV9irCOteTeB"
    else:
        print(f"[-]Model do not support: {model_engine}")
        return

    user_input = prompt
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

        chat = json.loads(chat)

    if chat.get("error"):
        print("Error: ", chat["error"]['message'])


    elif chat.get("choices"):
        answer = chat["choices"][0]['message']['content'].lstrip()
        chatLog.append({"role": "assistant", "content": answer})

    return answer
def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):

    if "llama" in engine.lower():
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"  # your local llama server port
        engine = openai.Model.list()["data"][0]["id"]
    else:
        openai.api_key = opeani_api_keys

    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    f = 0


    while(f == 0):
        try:

            api_key=opeani_api_keys,
            base_url = 'https://api.huiyan.chat/v1',
            response = openai.Completion.create(
                    model=engine,
                    messages = messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    )
            result = response.choices[0].message.content

            #result = openai_chat_completion(model_engine=engine, in_context=False, temperature=temperature, prompt=messages)
            f = 1
        except:
            print("openai error, retry")
            time.sleep(2)
    return result

    
def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates


def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)
    

def save_2_jsonl(question, answer, cluster_chain_of_entities, retrivainfo,file_name):
    dict = {"question":question, "results": answer, "reasoning_chains": cluster_chain_of_entities, "retrieval_info": retrivainfo}
    with open("ToG_{}.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")

    
def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    

def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False


def generate_without_explored_paths(question,cluster_chain_of_entities, args, retrivainfo):
    prompt = answer_prompt + question + '\n'
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + '\nthe associated retrieved knowledge: ' + retrivainfo + 'A: '
    if args.enable_fastchat_conv and 'lama' in args.backend:
        result = llama31_instruct(prompt, model=args.backend, n=1)[0]
    else:
    # 调用 GPT 模型获取评分
        # result = gpt(prompt, n=1, stop=None)[0].strip()
        result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    # result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return result


def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    elif dataset_name == 'hotpot_e':
        with open('../data/hotpotadv_entities_azure.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string
