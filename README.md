# Enhancing Decision-Making for LLM Agents via Step-Level Q-Value Models

<p>
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Python-3.9+-1f425f.svg?color=purple">
    </a>
    <a href="https://copyright.illinois.edu/">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
    </a>
</p>

[**🤗 Dataset**]() 


Official repo for Enhancing Decision-Making for LLM Agents via Step-Level Q-Value Models.

We propose training a Q-value model to guide action selection for LLM agents in each decision-making step.
Our method comprises both training and inference stages. During the training stage, we first use Monte Carlo Tree Search (MCTS) to explore high-quality trajectories, annotating the actions in each step with Q-values. We then construct preference data and train the Q-value model using step-level Direct Policy Optimization (DPO). During inference, the trained Q-value model guides action selection at each decision-making step.


<p align="center">
<img src=assets/architecture.png width=700/>
</p>




Our method has following features:
| Approach                                      | Step Level | Applicable to API-based LLMs | Single Trial | Task Experience Accumulation |
|-----------------------------------------------|:----------:|:----------------------------:|:------------:|:----------------------------:|
| **Prompt Strategies:** Reflection, Reflexion  | ❌         | <span style="color:green;">✔</span> | <span style="color:green;">✔</span> or ❌ | ❌ |
| **Tree Search:** LATS, Search-agent           | <span style="color:green;">✔</span> | <span style="color:green;">✔</span> | ❌ | ❌ |
| **Fine-tuning:** Agent-FLAN, AgentEvol, ETO   | ❌         | ❌                            | <span style="color:green;">✔</span> | <span style="color:green;">✔</span> |
| **Q-value model enhanced (Ours)**             | <span style="color:green;">✔</span> | <span style="color:green;">✔</span> | <span style="color:green;">✔</span> | <span style="color:green;">✔</span> |






## 🛠️ Environments Setup
#### WebShop


1. Move to the WebShop directory:
```bash
cd LLM-Agents-with-Q/webshop
```

2. Install WebShop from source and run environment instance locally. Follow the instructions here (https://github.com/princeton-nlp/WebShop)

3. Install the module dependencies into your environment:
```bash
pip install -r requirements.txt
```
#### HotPotQA

1. Move to the HotPotQA directory:
```bash
git clone https://github.com/andyz245/LLM-Agents-with-Q && cd LLM-Agents-with-Q/hotpot
```

2. Install the module dependencies into your environment:
```bash
pip install -r requirements.txt
```



## 🎎Multi-type Agents Support
### API-based LLM agents
Set `OPENAI_API_KEY` environment variable to your OpenAI API key:
```bash
export OPENAI_API_KEY=<your key>
```


### Open-source LLM agents
For open-source LLM agents, we adopt the OpenAI-compatible APIs provided by [FastChat](https://github.com/lm-sys/FastChat).

1. Move to the fastchat directory:
```bash
cd fastchat
```

2. Launch the controller of FastChat
```bash
python -m fastchat.serve.controller
```

3. Launch the model worker of FastChat
```bash
bash fastchat/start_multiple_vllm_server_from0_Phi3.sh
```










## 🚀 Training
We use the HotPotQA task as an example, which can be directly transferred to the webshop task.    
1. Collect trajectories with MCTS

```bash
cd hotpot
bash scripts/data_collection/collect_trajectories-phi3.sh
```


2. Construct step-level preference data

```bash
cd hotpot
python hotpot/scripts/construct_preference_data.py
```

3. Train Q-value models

```bash
bash run-hotpot-Q-value-model.sh
```


## 🎮 Inference



Finally, evaluate the agent
```bash
cd hotpot
bash scripts/eval/eval-Phi-3_with-Q-epoch1.sh
```
- ``--algorithm``: "simple" refers to Greedy Decision-making, while "beam" refers to Guiding Action Selection with Q-value model.