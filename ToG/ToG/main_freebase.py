from tqdm import tqdm
import argparse
from utils import *
from freebase_func import *
import random
from client import *
from concurrent.futures import ThreadPoolExecutor, as_completed

# 并行处理
def process_data(data, args):
    question = data[question_string]
    topic_entity = data['topic_entity']
    #给qald数据集特殊处理
    #topic_entity = data['qid_topic_entity']
    #给hotpotadv数据集特殊处理
    #topic_entity = data['entities']
    cluster_chain_of_entities = []
    retriveinformation = ''


    if len(topic_entity) == 0:
        retriveinfo = process_search(question, cluster_chain_of_entities, retriveinformation, args)
        retriveinformation += retriveinfo
        results = generate_without_explored_paths(question, cluster_chain_of_entities, args,retriveinformation)
        save_2_jsonl(question, results, [], retriveinformation, file_name=args.dataset)
        return

    pre_relations = []
    pre_heads = [-1] * len(topic_entity)
    flag_printed = False
    search_depth = 1

    for depth in range(1, args.depth + 1):
        current_entity_relations_list = []
        i = 0
        for entity in topic_entity:
            if entity != "[FINISH_ID]":
                retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i], question, args)
                current_entity_relations_list.extend(retrieve_relations_with_scores)
            i += 1
        total_candidates = []
        total_scores = []
        total_relations = []
        total_entities_id = []
        total_topic_entities = []
        total_head = []


        for entity in current_entity_relations_list:
            if entity['head']:
                entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
            else:
                entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)

            if args.prune_tools == "llm":
                if len(entity_candidates_id) >= 20:
                    entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)

            if len(entity_candidates_id) == 0:
                continue
            scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity['score'], entity['relation'], args)

            total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)

        if len(total_candidates) == 0:
            if retriveinformation == '':
                retriveinformation = process_search(question, cluster_chain_of_entities, retriveinformation, args)
            half_stop(question, cluster_chain_of_entities, depth, args, retriveinformation)
            flag_printed = True
            break

        flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args)
        cluster_chain_of_entities.append(chain_of_entities)
        if flag:
            stop, results = reasoning(question, cluster_chain_of_entities, args,retriveinformation)
            if stop:
                print("ToG stopped at depth %d." % depth)
                results = generate_answer(question, cluster_chain_of_entities, args, retriveinformation)
                save_2_jsonl(question, results, cluster_chain_of_entities, retriveinformation, file_name=args.dataset)
                flag_printed = True
                break
            else:
                print("depth %d still not find the answer." % depth)
                flag_finish, entities_id = if_finish_list(entities_id)
                if flag_finish:
                   half_stop(question, cluster_chain_of_entities, depth, args, retriveinformation)
                   flag_printed = True
                else:
                   #调用搜索补充信息
                    for i in range(search_depth):
                        retriveinfo = process_search(question, cluster_chain_of_entities, retriveinformation, args)
                        retriveinformation += retriveinfo
                        stop, results = reasoning(question, cluster_chain_of_entities, args,retriveinformation)
                        if stop:
                            print("ToG stopped at depth %d." % depth)
                            results = generate_answer(question, cluster_chain_of_entities, args, retriveinformation)
                            save_2_jsonl(question, results, cluster_chain_of_entities, retriveinformation, file_name=args.dataset)
                            flag_printed = True
                            break

                    topic_entity = {entity: id2entity_name_or_type(entity) for entity in entities_id}
                    continue
        else:
            half_stop(question, cluster_chain_of_entities, depth, args, retriveinformation)
            flag_printed = True
    search_depth += 2

    if not flag_printed:
        if retriveinformation == '':
            retriveinformation = process_search(question, cluster_chain_of_entities, retriveinformation, args)
        results = generate_without_explored_paths(question, cluster_chain_of_entities, args, retriveinformation)
        save_2_jsonl(question, results, [], retriveinformation, file_name=args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    args = parser.parse_args()

    datas, question_string = prepare_dataset(args.dataset)
    print("Start Running ToG on %s dataset." % args.dataset)
    # 改并行
    with ThreadPoolExecutor(max_workers=40) as executor:  # 根据需要调整 max_workers
        futures = [executor.submit(process_data, data, args) for data in datas]
        for future in as_completed(futures):
           try:
               future.result()  # 获取结果，如果有异常会在这里抛出
           except Exception as e:
               print(f"An error occurred: {e}")

    '''
    for data in tqdm(datas):
        question = data[question_string]
        topic_entity = data['topic_entity']
        cluster_chain_of_entities = []
        if len(topic_entity) == 0:
            results = generate_without_explored_paths(question, args)
            save_2_jsonl(question, results, [], file_name=args.dataset)
            continue
        pre_relations = []
        pre_heads= [-1] * len(topic_entity)
        flag_printed = False
        for depth in range(1, args.depth+1):
            current_entity_relations_list = []
            i=0
            for entity in topic_entity:
                if entity!="[FINISH_ID]":
                    retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i], question, args)  # best entity triplet, entitiy_id
                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                i+=1
            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities_id = []
            total_topic_entities = []
            total_head = []
            retriveinformation = ''

            for entity in current_entity_relations_list:
                if entity['head']:
                    entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
                else:
                    entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)
                
                if args.prune_tools == "llm":
                    if len(entity_candidates_id) >=20:
                        entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)

                if len(entity_candidates_id) ==0:
                    continue
                scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity['score'], entity['relation'], args)
                
                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)
            
            if len(total_candidates) ==0:
                half_stop(question, cluster_chain_of_entities, depth, args,retriveinformation)
                flag_printed = True
                break
                
            flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args)
            cluster_chain_of_entities.append(chain_of_entities)
            if flag:
                # 修改，增加推理过程后调用搜索
                stop, results,retriveinformation = reasoning(question, cluster_chain_of_entities, args)
                if stop:
                    print("ToG stoped at depth %d." % depth)
                    save_2_jsonl(question, results, cluster_chain_of_entities,retriveinformation, file_name=args.dataset)
                    flag_printed = True
                    break
                else:
                    print("depth %d still not find the answer." % depth)
                    flag_finish, entities_id = if_finish_list(entities_id)
                    if flag_finish:
                        half_stop(question, cluster_chain_of_entities, depth, args,retriveinformation)
                        flag_printed = True
                    else:
                        topic_entity = {entity: id2entity_name_or_type(entity) for entity in entities_id}
                        continue
            else:
                half_stop(question, cluster_chain_of_entities, depth, args,retriveinformation)
                flag_printed = True
        
        if not flag_printed:
            results = generate_without_explored_paths(question, cluster_chain_of_entities, args,retriveinformation)
            save_2_jsonl(question, results, [], retriveinformation, file_name=args.dataset)
        '''