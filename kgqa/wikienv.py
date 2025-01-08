import ast
import json
import time
import gym
import requests
from bs4 import BeautifulSoup

import backoff
from requests.exceptions import RequestException
import os

#图谱相关库
from tqdm import tqdm
import argparse
from utils import *
from tog.freebase_func import *
import random
from tog.client import *
# import wikipedia

def clean_str(p):
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except UnicodeDecodeError:
        return p


class textSpace(gym.spaces.Space):
  def contains(self, x) -> bool:
    """Return boolean specifying if x is a valid member of this space."""
    return isinstance(x, str)


class WikiEnv(gym.Env):

  def __init__(self):
    """
      Initialize the environment.
    """
    super().__init__()
    self.page = None  # current Wikipedia page
    self.obs = None  # current observation
    self.lookup_keyword = None  # current lookup keyword
    self.lookup_list = None  # list of paragraphs containing current lookup keyword
    self.lookup_cnt = None  # current lookup index
    self.steps = 0  # current number of steps
    self.answer = None  # current answer from the agent
    self.observation_space = self.action_space = textSpace()
    self.search_time = 0
    self.num_searches = 0
    # 增加节点
    self.node = None
    self.cluster_chain_of_entities = []
    
  def _get_obs(self):
    return self.obs

  def _get_info(self, answer):
    return {"steps": self.steps, "answer": answer}

  def reset(self, seed=None, return_info=False, options=None):
    # We need the following line to seed self.np_random
    # super().reset(seed=seed)
    self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                "finish[].\n")
    self.page = None
    self.lookup_keyword = None
    self.lookup_list = None
    self.lookup_cnt = None
    self.steps = 0
    self.answer = None
    observation = self._get_obs()
    info = self._get_info(answer=None)
    return (observation, info) if return_info else observation

  def construct_lookup_list(self, keyword):
    # find all paragraphs
    if self.page is None:
      return []
    paragraphs = self.page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]

    parts = sentences
    parts = [p for p in parts if keyword.lower() in p.lower()]
    return parts

  @staticmethod
  def get_page_obs(page):
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    return ' '.join(sentences[:5])

    # ps = page.split("\n")
    # ret = ps[0]
    # for i in range(1, len(ps)):
    #   if len((ret + ps[i]).split(" ")) <= 50:
    #     ret += ps[i]
    #   else:
    #     break
    # return ret
  @backoff.on_exception(backoff.expo,
                      requests.exceptions.RequestException,
                      max_tries=10)
  def search_step(self, entity):
    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    old_time = time.time()

    os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    response_text = requests.get(search_url).text
    del os.environ['https_proxy']
    del os.environ['HTTPS_PROXY']

    self.search_time += time.time() - old_time
    self.num_searches += 1
    soup = BeautifulSoup(response_text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    if result_divs:  # mismatch
      self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
      self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
    else:
      page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
      if any("may refer to:" in p for p in page):
        self.search_step("[" + entity + "]")
      else:
        self.page = ""
        for p in page:
          if len(p.split(" ")) > 2:
            self.page += clean_str(p)
            if not p.endswith("\n"):
              self.page += "\n"
        self.obs = self.get_page_obs(self.page)
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
  
  def select(self,node):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=4096, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=1, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=1, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="sk-F29EYhchwPz1aEmoRu0U7W2IQL7vRxjHyFKkPV9irCOteTeB",
                        help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    args = parser.parse_args()
    question = node.question
    topic_entity = node.topic_entity


    pre_relations = []
    pre_heads = [-1] * len(topic_entity)
    flag_printed = False
    search_depth = 1

    current_entity_relations_list = []
    i = 0
    for entity in topic_entity:
      if entity != "[FINISH_ID]":
        retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations,
                                                               pre_heads[i], question, args)
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
      scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity['score'],
                                                                     entity['relation'], args)

      total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(
        entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations,
        total_entities_id, total_topic_entities, total_head)


    flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations,
                                                                                  total_candidates,
                                                                                  total_topic_entities, total_head,
                                                                                  total_scores, args)
    self.cluster_chain_of_entities.append(chain_of_entities)
    node.entity_list = entities_id
    new_topic_entity = {}
    for i in range(len(total_candidates)):
      new_topic_entity[total_entities_id[i]] = total_candidates[i]
    node.topic_entity = new_topic_entity
    return chain_of_entities,total_scores[0]




  def step(self, action):
    reward = 0
    done = False
    action = action.strip()
    answer = None
    # if self.answer is not None:  # already finished
    #   done = True
    #   return self.obs, reward, done, self._get_info()
    
    if action.startswith("search[") and action.endswith("]"):
      entity = action[len("search["):-1]
      # entity_ = entity.replace(" ", "_")
      # search_url = f"https://en.wikipedia.org/wiki/{entity_}"
      self.search_step(entity)
    elif action.startswith("lookup[") and action.endswith("]"):
      keyword = action[len("lookup["):-1]
      if self.lookup_keyword != keyword:  # reset lookup
        self.lookup_keyword = keyword
        self.lookup_list = self.construct_lookup_list(keyword)
        self.lookup_cnt = 0
      if self.lookup_cnt >= len(self.lookup_list):
        self.obs = "No more results.\n"
      else:
        self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
        self.lookup_cnt += 1
    # 剪枝获得下一节点的实体
    elif action.startswith("select[") and action.endswith("]"):
      raise NotImplementedError
      '''
      r,current_triple = self.select(self.node)
      self.obs = f"Knowledge Triplets:  {current_triple}\n"
      reward = r
      '''
    elif action.startswith("finish[") and action.endswith("]"):
      answer = action[len("finish["):-1]
      # self.answer = answer
      done = True
      # 找到答案
      reward = 1
      self.obs = f"Episode finished, reward = {reward}\n"
    elif action.startswith("think[") and action.endswith("]"):
      self.obs = "Nice thought."
    else:
      self.obs = "Invalid action: {}".format(action)

    self.steps += 1

    return self.obs, reward, done, self._get_info(answer)
  
  def get_time_info(self):
    speed = self.search_time / self.num_searches if self.num_searches else 0
    return {
        "call_speed": speed,
        "call_time": self.search_time,
        "num_calls": self.num_searches,
    }
