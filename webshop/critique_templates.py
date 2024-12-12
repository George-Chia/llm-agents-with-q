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

# Please remember not to provide any specific suggestions or next steps.
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

rewrite_prompt = """
Below are the previous Thought and Action you generated along with their corresponding Observation:

<previous Thought, Action, and Observation>
{previous_response}
{previous_obs}
</previous Thought, Action, and Observation>

Critique:
<critique>
{critique}
</critique>

Based on the critique, generate a new Thought and Action with as much distinctiveness as possible for the current Observation:
<current observation>
Observation:
{original_observation}
</current observation>

<constraint>
Remember that the action - search[keywords] can only be performed on the initial interface. In other cases, to search, you need to first perform action - click[Back to Search] to return to the initial interface.
</constraint>

Consider the feasibility of new actions based on the current observation, rather than relying on the previous observation.
"""


icl_prompt=[
['user',
"""
<user instruction>
WebShop [SEP] Instruction: [SEP] i looking gluten free italian ground sausage, and price lower than 70.00 dollars [SEP] Search
</user instruction>

<historical context>
Thought: I think I should use the search bar to look for gluten-free Italian ground sausage.
Action: search[itlian gluten free ground sausage price < 70.00]
Observation: 
Instruction: [SEP] i looking gluten free italian ground sausage, and price lower than 70.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B08GY7G2JL [SEP] Pederson’s Farms, Spicy Breakfast Sausage, Whole30 (5 Pack, Use / Freeze) 16oz – No Sugar Added, Keto and Paleo Diet Friendly, No Nitrates or Nitrites, Gluten Free, Uncured Sausage, Made in the US [SEP] $37.99 [SEP] B08M26278X [SEP] Classic Smoked Salami, Beef & Pork Cervelat Sausage 1.4 lb [SEP] $24.99 [SEP] B08TKNJL31 [SEP] SANRIKU FISH SAUSAGE, SANRIKU (CHILLI PEPPER) WITH OMEGA 3, BREAKFAST SAUSAGE, WHOLE FOOD, GLUTEN-FREE WITH NO PRESERVATIVES [SEP] $7.99 [SEP] B08TKVZRQC [SEP] SANRIKU FISH SAUSAGE, SANRIKU SHRIMP WITH OMEGA 3, BREAKFAST SAUSAGE, WHOLE FOOD, GLUTEN-FREE WITH NO PRESERVATIVES [SEP] $7.99 [SEP] B08TL22YH3 [SEP] SANRIKU FISH SAUSAGE, SANRIKU BASIL AND CHEESE WITH OMEGA 3, BREAKFAST SAUSAGE, WHOLE FOOD, GLUTEN-FREE WITH NO PRESERVATIVES [SEP] $7.99 [SEP] B007WTSUYK [SEP] Fortuna's Hot SOUPY® Salami 2- 8 oz. sticks Calabria Style Dry Salami Stick Hand Made, spicy Hot traditionally cured Salami Sausage, Nitrate Free and Gluten Free [SEP] $100.0 [SEP] B01J48OHQQ [SEP] Banquet Brown ‘N Serve Fully Cooked Turkey Sausage Links, Frozen Meat, 10 Count, 6.4 OZ [SEP] $100.0 [SEP] B0898JBY63 [SEP] Banquet Brown ‘N Serve Original Fully Cooked Sausage Links, Frozen Meat, About 50 Count, 32 OZ [SEP] $100.0 [SEP] B08GYHRS28 [SEP] Pederson’s Farms, No Sugar Added Chorizo, Whole30 (5 Pack, Use / Freeze) 16oz ea – No Sugar Added, Keto, Paleo Diet Friendly, No Nitrates Nitrites, Gluten Free, Made in the USA [SEP] $37.99 [SEP] B088X6LPYD [SEP] Pederson’s Farms, Old World Kielbasa Rope Sausage, Whole30 (10 Pack, Use / Freeze) 14oz – No Sugar Added, Uncured, Keto Paleo Diet Friendly, No Nitrates Nitrites, Gluten Free, Made in the US [SEP] $59.99
</historical context>

Critique:"""],

['assistant',
 """Critique: The previous action was effective in narrowing down a list of potential products that match the search criteria. However, there is a typographical error in the search terms with "itlian" instead of "italian," which might reduce the effectiveness of the search results. Additionally, the search terms could be more specific to ensure results for "ground sausage" rather than other types of sausage. Therefore, the action was mostly effective, but attention to detail in search keywords could yield even more relevant results.
 """],

['user',
"""
<user instruction>
WebShop [SEP] Instruction: [SEP] i would like a non gmo container of artichoke hearts, and price lower than 40.00 dollars [SEP] Search
</user instruction>

<historical context>
Thought: I think I should use more specific keywords to narrow down the search results and focus on known brands or labels that certify non-GMO.
Action: search[nongmo certified artichoke hearts container < $40]
Observation: 
Instruction: [SEP] i would like a non gmo container of artichoke hearts, and price lower than 40.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B093JS8RFR [SEP] Reese Chopped Artichoke Hearts, 14 Ounces (Pack of 3) [SEP] $14.99 [SEP] B093JLXRB6 [SEP] Reese Quartered Artichoke Hearts, 14 Ounces (Pack of 3) [SEP] $100.0 [SEP] B000FDCS84 [SEP] Reese Petite Artichoke Hearts, 7 Ounces (Pack of 12) [SEP] $29.2 [SEP] B000SR600O [SEP] Reese Quartered Artichoke Hearts, 14 Ounces [SEP] $100.0 [SEP] B08796DFGR [SEP] Artichoke Lemon Pesto, 6 oz. (4 pack) [SEP] $23.98 [SEP] B01BGQ1HNA [SEP] More Than Fair Whole Artichoke Hearts Jar, 9.8 Ounce [SEP] $6.99 [SEP] B000FDB3XK [SEP] Reese Artichoke Hearts, Small Size, 14-Ounce Cans (Pack of 12) [SEP] $100.0 [SEP] B079H5B8DQ [SEP] Idaho Spuds Premium Hashbrown Potatoes 4.2 oz, 8 Pack, Made from 100 Potatoes No Artificial Colors or Flavors NonGMO Certified Gluten Free Koshe, 33.6 Ounce [SEP] $14.99 [SEP] B00A3IFR5C [SEP] Living Intentions Organic Superfood Cereal – Blueberry Blast – NonGMO – Gluten Free – Vegan Paleo – Kosher – 9 Ounce Unit [SEP] $6.49 [SEP] B0779DBB1B [SEP] Living Intentions Organic Superfood Cereal - Radiant Raspberry – NonGMO – Gluten Free – Vegan – Paleo – Kosher – 9 Oz [SEP] $10.39
Thought: I think I should choose new search keywords that mentions non-GMO.
Action: search[non-GMO certified artichoke hearts container]
Observation: 
Invalid action!
</historical context>

Critique:"""],
['assistant',
 """Critique: The action to search[non-GMO certified artichoke hearts container] was incorrect as it was not available in current web interface. If you want to search again, please click the button - "Back to Search" first, which is click[Back to Search]."""]
]