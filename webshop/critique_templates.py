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

# When you feel that your state is on the path to successfully completing the task and does not require critique, please output: the state is efficient and does not require critique.
template_huan = """
You are a critic, and your responsibility is to criticize the effectiveness of a single step (including thought, action, observation) in completing user instructions and explain the high level reasons behind it.
I will give you task scenario description, the user instruction, historical context and the current step. The historical context includes previous mutilples step, and the current step is executed based on the interaction history.
You task is to only the reason for the inefficiency of the current step. Please remember that you should not provide any specific suggestions or a next step regarding the current step.
Give your concise critique after "Critique:", no more than 100 words.
</system>
Task Scenario Description:
{scenario_description}

User Instruction:
{user_inst}

Historical Context:
{historical_context}

Current Step: 
{substep}

Critique:
"""

webshop_description = """You are web shopping.
I will give you instructions about what to do.
You have to follow the instructions.
Every round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.
You can use search action if search is available.
You can click one of the buttons in clickables.
An action should be of the following structure:
search[keywords]
click[value]
If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.
Your response should use the following format:

Thought: I think ...
Action: click[something]
"""

# icl='''
# WebShop [SEP] Instruction: [SEP] i am looking for a queen sized bed that is black, and price lower than 130.00 dollars [SEP] Search
# Thought: I think I should use the search bar to look for the product I need.
# Action: search[black queen bed price < 130.00]
# Observation:
# Instruction: [SEP] i am looking for a queen sized bed that is black, and price lower than 130.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B09P1RYKNN [SEP] Best Bed Frame, 7" Adjustable Bed Frame from Twin to Queen, Black Metal Bed Frame - Solid Construction, Convenient, Affordable Price and Easy Assembly (Black) [SEP] $54.99 [SEP] B0959B21HW [SEP] Signature Design by Ashley Finch Modern Platform Bed Frame, Queen, Black [SEP] $161.45 [SEP] B07CVQ1V27 [SEP] Best Price Mattress 18 Inch Metal Platform Beds w/Heavy Duty Steel Slat Mattress Foundation (No Box Spring Needed), Queen, Black [SEP] $135.06 [SEP] B078Y7J2RY [SEP] Queen Size Montreal Espresso Futon Frame Innerspring Mattress Sofa Bed Wood Futons (Grey Mattress, Frame, Drawers (Queen Size)) [SEP] $799.95 [SEP] B089ZQPZLH [SEP] Queen Size Pillow Cases Set of 2 – Soft, Premium Quality Pillowcase Covers – Machine Washable Protectors – 20x26 & 20x30 Pillows for Sleeping 2 Piece - Queen Size Pillow Case Set [SEP] $11.89 [SEP] B07CVQNKP5 [SEP] Best Price Mattress 14 Inch Metal Platform Beds w/ Heavy Duty Steel Slat Mattress Foundation (No Box Spring Needed), Black [SEP] $100.0 [SEP] B00LLQK6BK [SEP] TRIBECCA home Wrought Iron Bed Frame Dark Bronze Metal Queen Size USA Vintage Look Shabby Chic French Country (Queen) [SEP] $340.75 [SEP] B07CVP3TB1 [SEP] Best Price -Mattress 7 Inch Metal Platform Beds w/ Heavy Duty Steel Construction Compatible with Twin, Full, and Queen Size [SEP] $71.52 [SEP] B07MC93ZPS [SEP] Mellow Mission - 10 Inch Metal Platform Bed with Headboard, Patented Wide Steel Slats, Easy Assembly, Queen, Black [SEP] $258.94 [SEP] B07TC5W9PB [SEP] 4 Piece Bed Sheets Set, Hotel Luxury 600 TC Platinum Collection Bedding Set, 15" Deep Pockets, Wrinkle & Fade Resistant, Hypoallergenic Sheet & Pillow Case Set (Calking, Wine) [SEP] $66.7
# critique:
# The current step is overly broad and likely to yield many irrelevant results, as it includes all beds in all price ranges, not just queen-sized beds with a price lower than 130.00 dollars.
# '''