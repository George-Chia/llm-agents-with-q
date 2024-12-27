import numpy as np
from models import gpt
import logging
import numpy as np
from fastchat.conversation import get_conv_template

import torch

from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def softmax(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=0)

def select_node_softmax(node, temperature=1.0):
    while node and node.children:
        uct_values = [child.uct() for child in node.children if not child.is_terminal]
        if not uct_values:
            return None  # All children are terminal

        probabilities = softmax(np.array(uct_values), temperature)
        selected_child = np.random.choice([child for child in node.children if not child.is_terminal], p=probabilities)
        
        node = selected_child
    return node


class Node:
    def __init__(self, state, question, topic_entity={}, env_state=None, parent=None):
        self.state = {'action': '', 'observation': topic_entity} if state is None else state
        self.parent = parent
        self.question = question
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0
        self.exhausted = False # If all children are terminal
        self.em = 0  # Exact match, evaluation metric
        self.env_state = env_state

        self.messages = None # Added by , only for root node
        self.puct_coeff = None # Added by 
        self.q_value = None


        # 增加中心词
        self.topic_entity = topic_entity
        # 增加当前实体列表
        self.entity_list = []
        # 增加当前轨迹
        self.triple = ""
        # 增加当前对wiki的搜索
        self.wikiinformation = ""
    def uct(self):
        if self.visits == 0 and self.value >= 0:
            return float('inf')
            #return self.value * 2
        elif self.visits == 0 and self.value < 0:
            return self.value
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)

    def puct(self, policy_model, reference_model, conv_template, tokenizer, puct_coeff):
        # TODO: tune puct_c
        # puct_coeff = 1e5
        if self.visits == 0 and self.value >= 0:
            return float('inf')
            #return self.value * 2
        elif self.visits == 0 and self.value < 0:
            return self.value
        q_value = get_q_value(self, policy_model, reference_model, conv_template, tokenizer)
        # q_value_sum = sum(q_value_list)
        # action_probs = [q_value/q_value_sum for q_value in q_value_list]
        return self.value / self.visits + puct_coeff * q_value * (np.sqrt(self.parent.visits) / (1+self.visits))
        # return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    
    def uct_with_depth(self, C1=1, C2=1):
        if self.visits == 0:
            return self.value
        exploitation_term = self.value / self.visits
        exploration_term = np.sqrt(2 * np.log(self.parent.visits) / self.visits)
        depth_term = self.depth
        return exploitation_term + C1 * exploration_term + C2 * depth_term

    def __str__(self):
        return f"Node(depth={self.depth}, value={self.value:.2f}, visits={self.visits}, action={self.state['action']}, observation={self.state['observation']})"
    
    def to_dict(self):
        return {
            'state': self.state,
            'question': self.question,
            # 'parent': self.parent.to_dict() if self.parent else None,
            'children': [child.to_dict() for child in self.children] if len(self.children)>0 else None,
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'reward': self.reward,
            'em': self.em,
            'messages': self.messages,
        }
    # def to_dict(self):
    #     if len(self.children)>0:
    #         return {
    #             'state': self.state,
    #             'question': self.question,
    #             'parent': self.parent.to_dict() if self.parent else None,
    #             'children': [child.to_dict() for child in self.children],
    #             'visits': self.visits,
    #             'value': self.value,
    #             'depth': self.depth,
    #             'is_terminal': self.is_terminal,
    #             'reward': self.reward,
    #             'em': self.em,
    #         }
    #     else:
    #         return {
    #             'state': self.state,
    #             'question': self.question,
    #             'parent': self.parent.to_dict() if self.parent else None,
    #             'children': None,
    #             'visits': self.visits,
    #             'value': self.value,
    #             'depth': self.depth,
    #             'is_terminal': self.is_terminal,
    #             'reward': self.reward,
    #             'em': self.em,
    #         }
    
def node_trajectory_to_text(node_string):
    lines = node_string.split('\n')
    formatted_lines = []
    for line in lines:
        try:
            depth = int(line.split(",")[0].split("=")[1].strip())
            action = line.split(", action=")[1].split(", observation=")[0].strip()
            observation = line.split(", observation=")[1].split(")")[0].strip()
        except IndexError:
            continue
        
        if depth != 0:
            if action:
                formatted_lines.append(f"Action {depth}: {action}")
            if observation:
                formatted_lines.append(f"Observation {depth}: {observation}")
    
    return '\n'.join(formatted_lines)

def collect_actions_to_node(node):
    actions = []
    while node:
        if node.state['action']:
            actions.append(node.state['action'])
        node = node.parent
    return list(reversed(actions))


def collect_all_nodes(node):
        """Recursively collect all nodes starting from the given node."""
        nodes = [node]
        for child in node.children:
            nodes.extend(collect_all_nodes(child))
        return nodes

def collect_trajectory_from_bottom(node):
    trajectory = []
    #print("collecting traj", node)
    
    # Append the question from the root node
    trajectory.append(node.question)
    
    # Collect action and observation from each node till the root
    while node:
        if node.state and 'action' in node.state and node.state['action'] and node.parent:
            trajectory.insert(1, f"Action: {node.state['action']}")
        else:
            logging.warning(f"Missing or empty action in node at depth {node.depth}")
            
        if node.state and 'observation' in node.state and node.state['observation'] and node.parent:
            trajectory.insert(1, f"Observation: {node.state['observation']}\n")
        else:
            logging.warning(f"Missing or empty observation in node at depth {node.depth}")
        # 将可能的列表转成字符串
        if isinstance(node.state, list):
            # 如果 state 是列表，则将其转换为字符串
            trajectory.append('\n'.join(map(str, node.state)))
        else:
            trajectory.append(str(node.state))
        node = node.parent
        return '\n'.join(trajectory)

def get_conv_from_bottom(node, conv_template):
    conv = get_conv_template(conv_template)
    # Apply prompt templates
    messages = []
    while node.parent:
        # if 'regenerate_prompt' in node.state.keys():
        #     critique = f"{node.state['regenerate_prompt']}"
        #     messages.insert(0, [conv.roles[0], f"{node.state['observation']}"+critique+node.state['observation']])
        # else:
        messages.insert(0, [conv.roles[0], f"{node.state['observation']}"])
        messages.insert(0, [conv.roles[1], f"{node.state['action']}"])
        node = node.parent
    # root node
    for j, sentence in enumerate(node.messages):
        role = sentence['role']
        assert role in conv.roles[j % 2]
        conv.append_message(conv.roles[j % 2], sentence["content"])
    if len(messages)>0:
        conv.messages.extend(messages)
    return conv


def get_messages_from_bottom(node):
    messages = []
    while node.parent:
        # if 'regenerate_prompt' in node.state.keys():
        #     critique = f"{node.state['regenerate_prompt']}"
        #     messages.insert(0,{'role':'user', 'content': f"{node.state['observation']}"+critique+node.state['observation']})
        # else:
        # 增加 wiki 信息
        messages.insert(0,{'role':'user', 'content': f"{node.state['observation']}" + f"  {node.wikiinformation}"})
        messages.insert(0,{'role':'assistant', 'content': f"{node.state['action']}"})
        # if 'regenerate_prompt' in node.state.keys():
        #     critique = f"{node.state['regenerate_prompt']}"
        #     action = f"{node.state['action']}"
        #     messages.insert(0,{'role':'assistant', 'content': critique+"\n"+action})
        # else:
        #     messages.insert(0,{'role':'assistant', 'content': f"{node.state['action']}"})
        # messages.insert(0, [conv.roles[0], f"{node.state['observation']}"])
        # messages.insert(0, [conv.roles[1], f"{node.state['action']}"])
        node = node.parent
    # root node
    for message in node.messages[::-1]:
        messages.insert(0, message)
    return messages


def get_conv_from_bottom_critique(node, conv_template):
    conv = get_conv_template(conv_template)
    # Apply prompt templates
    messages = []
    while node.parent:
        if node.state['regenerate_prompt'] is not None:
            critique = f"{node.state['regenerate_prompt']}"
            messages.insert(0, [conv.roles[0], f"{node.state['observation']}"+critique+node.state['observation']])
        else:
            messages.insert(0, [conv.roles[0], f"{node.state['observation']}"])
        messages.insert(0, [conv.roles[1], f"{node.state['action']}"])
        node = node.parent
    # root node
    for j, sentence in enumerate(node.messages):
        role = sentence['role']
        assert role in conv.roles[j % 2]
        conv.append_message(conv.roles[j % 2], sentence["content"])
    if len(messages)>0:
        conv.messages.extend(messages)
    return conv


def get_messages_from_bottom_critique(node):
    messages = []
    while node.parent:
        if node.state['regenerate_prompt'] is not None:
            critique = f"{node.state['regenerate_prompt']}"
            messages.insert(0,{'role':'user', 'content': f"{node.state['observation']}"+critique+node.state['observation']})
        else:
            messages.insert(0,{'role':'user', 'content': f"{node.state['observation']}"})
        messages.insert(0,{'role':'assistant', 'content': f"{node.state['action']}"})
        # if 'regenerate_prompt' in node.state.keys():
        #     critique = f"{node.state['regenerate_prompt']}"
        #     action = f"{node.state['action']}"
        #     messages.insert(0,{'role':'assistant', 'content': critique+"\n"+action})
        # else:
        #     messages.insert(0,{'role':'assistant', 'content': f"{node.state['action']}"})
        # messages.insert(0, [conv.roles[0], f"{node.state['observation']}"])
        # messages.insert(0, [conv.roles[1], f"{node.state['action']}"])
        node = node.parent
    # root node
    for message in node.messages[::-1]:
        messages.insert(0, message)
    return messages


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def get_q_value(node, policy_model, reference_model, conv_template, tokenizer) -> list:
    conv = get_conv_from_bottom(node.parent, conv_template)
    prompt = conv.get_prompt()
    prompt_tokens = tokenizer(prompt, return_tensors="pt")

    conv.append_message(conv.roles[1], node.state['action'])
    prompt_response = conv.get_prompt()
    prompt_response_tokens = tokenizer(prompt_response, return_tensors="pt")
    labels = prompt_response_tokens.input_ids.clone()
    labels[0][:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

    with torch.no_grad():
        policy_logits = policy_model(
                prompt_response_tokens["input_ids"].to(policy_model.device),
                attention_mask=prompt_response_tokens["attention_mask"].to(policy_model.device),
            ).logits
        reference_logits = reference_model(
                prompt_response_tokens["input_ids"].to(policy_model.device),
                attention_mask=prompt_response_tokens["attention_mask"].to(policy_model.device),
            ).logits
    
        logits = policy_logits - reference_logits
    
        logps_value = get_batch_logps(
            logits,
            labels.to(policy_model.device),
            average_log_prob=True,
            is_encoder_decoder=False,
            label_pad_token_id=IGNORE_TOKEN_ID,
        )
        q_value = logps_value.exp()

    return q_value

def get_child_q_value_list(node, policy_model, reference_model, conv_template, tokenizer) -> list:
    conv = get_conv_from_bottom(node, conv_template)
    prompt = conv.get_prompt()
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    prompt_tokens_length = len(prompt_tokens['input_ids'][0])

    q_value_list = []
    for child in node.children:
        conv.append_message(conv.roles[1], child.state['action'])
        prompt_response = conv.get_prompt()
        prompt_response_tokens = tokenizer(prompt_response, return_tensors="pt")
        labels = prompt_response_tokens.input_ids.clone()
        labels[0][:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

        with torch.no_grad():
            policy_logits = policy_model(
                    prompt_response_tokens["input_ids"].to(policy_model.device),
                    attention_mask=prompt_response_tokens["attention_mask"].to(policy_model.device),
                ).logits
            reference_logits = reference_model(
                    prompt_response_tokens["input_ids"].to(policy_model.device),
                    attention_mask=prompt_response_tokens["attention_mask"].to(policy_model.device),
                ).logits
        
            logits = policy_logits - reference_logits
        
            logps_value = get_batch_logps(
                logits,
                labels.to(policy_model.device),
                average_log_prob=True,
                is_encoder_decoder=False,
                label_pad_token_id=IGNORE_TOKEN_ID,
            )
            q_value_list.append(logps_value)
        conv.pop_message()
    return q_value_list


def get_node_children_q_value_list(node,children, policy_model, reference_model, conv_template, tokenizer) -> list:
    conv = get_conv_from_bottom(node, conv_template)
    prompt = conv.get_prompt()
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    prompt_tokens_length = len(prompt_tokens['input_ids'][0])

    q_value_list = []
    for child in children:
        conv.append_message(conv.roles[1], child.state['action'])
        prompt_response = conv.get_prompt()
        prompt_response_tokens = tokenizer(prompt_response, return_tensors="pt")
        labels = prompt_response_tokens.input_ids.clone()
        labels[0][:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

        with torch.no_grad():
            policy_logits = policy_model(
                    prompt_response_tokens["input_ids"].to(policy_model.device),
                    attention_mask=prompt_response_tokens["attention_mask"].to(policy_model.device),
                ).logits
            reference_logits = reference_model(
                    prompt_response_tokens["input_ids"].to(policy_model.device),
                    attention_mask=prompt_response_tokens["attention_mask"].to(policy_model.device),
                ).logits
        
            logits = policy_logits - reference_logits
        
            logps_value = get_batch_logps(
                logits,
                labels.to(policy_model.device),
                average_log_prob=True,
                is_encoder_decoder=False,
                label_pad_token_id=IGNORE_TOKEN_ID,
            )
            q_value_list.append(logps_value.exp())
        conv.pop_message()
    return q_value_list


def beam_search(node, policy_model, reference_model, conv_template, tokenizer):
    q_value_list = get_child_q_value_list(node, policy_model, reference_model, conv_template, tokenizer)
    for index, q_value in enumerate(q_value_list):
        node.children[index].q_value = q_value
    max_q_value_index = q_value_list.index(max(q_value_list))
    return node.children[max_q_value_index]



def collect_fschat_trajectory(node, conv):
    trajectory = []
    #print("collecting traj", node)
    
    # Append the question from the root node
    trajectory.append(node.question)
    
    # Collect action and observation from each node till the root
    while node:
        if node.state and 'action' in node.state and node.state['action'] and node.parent:
            trajectory.append(f"Action: {node.state['action']}")
        else:
            logging.warning(f"Missing or empty action in node at depth {node.depth}")
            
        if node.state and 'observation' in node.state and node.state['observation'] and node.parent:
            trajectory.append(f"Observation: {node.state['observation']}\n")
        else:
            logging.warning(f"Missing or empty observation in node at depth {node.depth}")
            
        node = node.parent
    return '\n'.join(trajectory)

def collect_trajectory_index(node):
    trajectory_index = []
    while node.parent:
        trajectory_index.append(node.parent.children.index(node))
        node = node.parent
    return trajectory_index[::-1]
        

def collect_trajectory_fschat(node):
    trajectory = []
    #print("collecting traj", node)
    
    # Append the question from the root node
    trajectory.append(node.question)
    
    # Collect action and observation from each node till the root
    while node.parent:
        if node.state and 'action' in node.state and node.state['action'] and node.parent:
            trajectory.append(f"Action: {node.state['action']}")
        else:
            logging.warning(f"Missing or empty action in node at depth {node.depth}")
            
        if node.state and 'observation' in node.state and node.state['observation'] and node.parent:
            trajectory.append(f"Observation: {node.state['observation']}\n")
        else:
            logging.warning(f"Missing or empty observation in node at depth {node.depth}")
            
        node = node.parent
    return '\n'.join(trajectory)
