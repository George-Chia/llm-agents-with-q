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
You are a critic responsible for assessing the rationality of a single step (including thought, action, observation) in completing user instructions and explaining the high-level reasons behind it.
I will provide you with the task description, user instruction, and historical context. The historical context refers to the user's interaction history in the task environment, including multiple previous steps.
Your task is to provide reasonable or unreasonable reasons for the last step in the historical context. You cannot consider multiple steps or previous steps at once, please focus on the last step. 
Give your concise critique after 'Critique:', limited to no more than 100 words.

<task description>
{scenario_description}
</task description>

</system>
<user instruction>
{user_inst}
</user instruction>

<historical context>
{historical_context}
</historical context>

Critique:
"""

hotpot_description = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: \n(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.\n(3) Finish[answer], which returns the answer and finishes the task.\nAfter each observation, provide the next Thought and next Action.
"""