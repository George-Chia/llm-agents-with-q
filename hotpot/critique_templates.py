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



critique_prompt_template = """

Below are the previous Thought and Action you generated along with their corresponding Observation: 

{previous_response}
{previous_obs}

Review the previous Thought, Action, and Observation. Your role is to determine whether the action is effective for completing the task, and provide specific and constructive feedback. Please output feedback directly. 
Format
Feedback:[[Feedback]]"""