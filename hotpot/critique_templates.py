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
You are an assistant whose job is to help me perform tasks. I will give you task scenario description, the user instruction, historical context, and current state (including thought, action and observation). 
Your task is to synthesize these information, provide high-level critical feedback on the current state, and guide me to perform actions that are more likely or efficient to complete the user instruction. 
Give your critique after "Critique:". When you feel that your current state is on the path to successfully completing the task and does not require critique, please output: the current state is efficient and does not require critique.
</system>
Task Scenario Description:
{scenario_description}

User Instruction:
{user_inst}

Historical Context:
{historical_context}

Current State: 
{current_state}

Critique:
"""
# todo
hotpot_description = """ 
"""