auto_j_single_template = """

Write critiques for the previous Thought and Action you generated along with their corresponding Observation:
  
[BEGIN DATA]
***
[Previous Thought and Action]: {previous_response}
***
[Corresponding Observation]: {previous_obs}
***
[END DATA]

Write critiques for the previous Thought and Action.
Format
Feedback:[[Feedback]]"""



template_v1 = """

Below are the previous Thought and Action you generated along with their corresponding Observation: 

{previous_response}
{previous_obs}

Review the previous Thought, Action, and Observation. Your role is to determine whether the action is effective for completing the task, and provide specific and constructive feedback. Please output feedback directly. 
Format
Feedback:[[Feedback]]"""


template_v2 = """

Below are the previous Thought and Action you generated along with their corresponding Observation: 

<previous Thought, Action, and Observation>
{previous_response}
{previous_obs}
</previous Thought, Action, and Observation>

Review the previous Thought, Action, and Observation. Your role is to determine whether the action is effective for completing the task, and provide specific and constructive feedback. Please output feedback directly. 
Format
Feedback:[[Feedback]]"""

template_huan = """
You are a critic responsible for assessing the effectiveness of a single step (including thought, action, observation) in completing user instructions and explaining the high-level reasons behind it.
I will provide you with the task scenario description, user instruction, historical context, and the current step. The historical context includes previous multiple steps, and the current step is executed based on the interaction history. 
Your task is to identify the reason for the inefficiency of the current step. Please remember not to provide any specific suggestions or a next step regarding the current step.
Give your concise critique after 'Critique:', limited to no more than 100 words.
</system>
Task Scenario Description:
{scenario_description}

User Instruction:
{user_inst}

Historical Context:
{historical_context}

Current Step: 
{current_state}

Critique:
"""
# todo
hotpot_description = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: \n(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.\n(3) Finish[answer], which returns the answer and finishes the task.\nAfter each observation, provide the next Thought and next Action.
"""