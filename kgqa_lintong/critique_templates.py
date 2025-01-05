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

hotpot_description = """
To address a question-answering task, follow a sequence of Thought, Action, and Observation steps. Here's how each step works:

1. Thought: This is where you analyze the current state of information and determine which triplet you need to choose. It involves reasoning about which triplet is most helpful to answer the question.

2. Action: Your action must be in the form of Choose[head_entity, relation, tail_entity], with no other words. This step involves taking an action based on the Thought step:
   - Choose[head_entity, relation, tail_entity]: This action is choosing a triplet in the observation that is most helpful to answer the question. Even if there is no direct information in these triples to answer the question, please still choose one that might be the most helpful. It requires choosing an exact and complete triplet from the list provided in the last observation. The chosen triplet must match exactly with one of the triplets in the observation, with no variations allowed.

3. Observation: After each action, the next triplets will be provided, which will be used to inform the next Thought step.

After each observation, you should have a Thought about which to choose next and then an Action that follows from the Thought.
"""