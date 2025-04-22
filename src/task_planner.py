import torch
from torch.nn import CrossEntropyLoss
import guidance
import logging
import openai
import base64
from PIL import Image
from io import BytesIO
import re
from openai import OpenAI
import unicodedata
import time
import random
from openai import RateLimitError

def call_api_with_retry(model, messages, max_tokens, temperature, retries=5, backoff_factor=2):
    """
    Makes an API call and handles errors with retries and exponential backoff.

    Args:
        client: API client instance.
        model: Model name to use for the call.
        messages: List of input messages.
        max_tokens: Maximum number of tokens to generate.
        temperature: Temperature setting for the model.
        retries: Number of retries on failure (default: 3).
        backoff_factor: Factor for exponential backoff (default: 2).

    Returns:
        Response from the API call, or None if all retries fail.
    """

    api_key_list = [
        "YOUR_API_KEY_HERE"
    ]
    
    
    attempt = 0

    while attempt <= retries:
        try:
            # Attempt the API call
            client = OpenAI(api_key=random.choice(api_key_list), base_url="https://api.openai.com/v1")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            # If successful, return the response
            return response

        except RateLimitError as e:
            # Handle rate limit errors
            print(f"[ERROR] Rate limit error: {e}. Retrying in {backoff_factor ** attempt} seconds...")
            attempt += 1
            if attempt > retries:
                print("[ERROR] Exceeded maximum retries. Please try again later.")
                return None
            time.sleep(backoff_factor ** attempt)  # Exponential backoff

        except Exception as e:
            # Handle other potential exceptions
            print(f"[ERROR] An unexpected error occurred: {e}")
            time.sleep(backoff_factor ** attempt)
            if attempt > retries:
                print("[ERROR] Exceeded maximum retries. Please try again later.")
                return None

            # return None

def pil_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

def process_messages(messages):
    result = []
    for message in messages:
        if message['type'] == 'text':
            result.append(message['text'])
        elif message['type'] == 'image_url':
            result.append("{image}")
    return "\n".join(result)



def extract_action_object(content):
    action_pattern = r'Action:\s*(?:\[(.*?)\]|(.*?))(?=\n|$)'
    object_pattern = r'Object:\s*(?:\[(.*?)\]|(.*?))(?=\n|$)'
    
    action_match = re.search(action_pattern, content)
    action = (action_match.group(1) or action_match.group(2)).strip() if action_match else None
    
    object_match = re.search(object_pattern, content)
    obj = (object_match.group(1) or object_match.group(2)).strip() if object_match else None
    
    return action, obj


def validate_and_retry(content, action_list, object_list, client=None, model=None, max_retries=3):

    for attempt in range(max_retries):
        action, obj = extract_action_object(content)
        if action and obj:
            if 'done' in action:
                return 'done'

            action = action.lower()
            if obj != 'CD':
                obj = obj.lower()

            if (action in action_list) and (obj in object_list):
                return f"{action} {obj}"
            else:
                # print(f"{action} {obj}")
                mapping_prompt = f"""Please map the following action and object to the closest valid options:
                Original Action: {action}
                Original Object: {obj}
                ---
                Valid Actions: {action_list}
                Valid Objects: {object_list}
                
                Respond in format:
                Action: [mapped_action]
                Object: [mapped_object]"""

                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": mapping_prompt},
                    ],
                    stream=False
                )
                print(mapping_prompt)
                if response:
                    content = response.choices[0].message.content
                else:
                    content = ""
                print(f"REMATCH: {content}")

        else:
            mapping_prompt = f"""Please map the following content to the closest valid options:
            Content: {content}
            Valid Actions: {action_list}
            Valid Objects: {object_list}
            
            Respond in format:
            Action: [mapped_action]
            Object: [mapped_object]"""
        
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": mapping_prompt},
                ],
                stream=False
            )
    
            if response:
                content = response.choices[0].message.content
            else:
                content = ""
            print(f"REGEN: {content}")

                
    return ""  # Return empty string if all attempts fail

def extract_results(content):
    action_list = ["find", "pick up", "put down", "open", "close", "slice", "turn on", "turn off", "done"]
    action_pattern = r'Action:\s*(.*?)(?=\n|$)'  # 匹配 Action 后面的内容直到换行或结束
    object_pattern = r'Object:\s*(.*?)(?=\n|$)'  # 匹配 Object 后面的内容直到换行或结束

    # Extract Action
    action_match = re.search(action_pattern, content)
    action = action_match.group(1).strip() if action_match else None

    # Extract Object
    object_match = re.search(object_pattern, content)
    obj = object_match.group(1).strip() if object_match else None

    if action and obj and action in action_list:
        best_step = f"{action} {obj}"
    else:
        best_step = ""

    return best_step


def extract_or_average_score(content):
    """
    从给定内容中提取方括号内的数值。
    如果未找到方括号内的数值，则提取所有数字并求平均。
    :param content: 包含 [数字] 或其他数字的文本
    :return: 提取到的方括号内的数值（整数），或所有数字的平均值（浮点数）
    """
    # 尝试提取方括号内的数字
    match = re.search(r"\[(\d+)\]", content)
    if match:
        return int(match.group(1))
    
    # 如果没有找到方括号内的数字，提取所有数字
    all_numbers = re.findall(r"\d+", content)
    if all_numbers:
        numbers = [int(num) for num in all_numbers]
        return sum(numbers) / len(numbers)  # 返回平均值
    return None  # 如果没有任何数字，返回 None



class TaskPlanner:
    def __init__(self, cfg):

        self.cfg = cfg
        self.model_name = cfg.planner.model_name

        # Load pre-trained model
        print(f"Initializing Policy Model: {cfg.planner.model_name}")

        self.base_url = cfg.planner.base_url
        self.api_key = cfg.planner.api_key

        self.client = openai.Client(base_url=self.base_url, api_key=self.api_key)
        self.max_steps = cfg.planner.max_steps
        self.icl = cfg.planner.icl
        self.sft = cfg.planner.sft


    # def reset(self, nl_act_list, nl_obj_list):
    #     self.nl_obj_list = nl_obj_list
    #     self.skill_set = self.init_skill_set(nl_act_list, nl_obj_list)

    # def reset(self):
    #     self.skill_set = self.init_skill_set()

    # def init_prompt(self, cfg):
    #     raise NotImplementedError()

    # def init_skill_set(self, nl_act_list, nl_obj_list):
    #     raise NotImplementedError()

    # def update_skill_set(self, previous_step, nl_obj_list):
    #     raise NotImplementedError()



    def generate_step_details_sft(self, query, prev_steps=(), prev_msgs=(), prev_images=None, visible_objs_str=None, n=1):
        action_list = ["find", "pick up", "put down", "open", "close", "slice", "turn on", "turn off", "done"]

        if len(prev_steps) >= self.max_steps:
            return None, None, None
        # import pdb; pdb.set_trace()

        base64_images = []
        for img in prev_images:
            img_base64 = pil_to_base64(img)
            base64_images.append(img_base64)

        # client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
        client = self.client


        # Start building the messages list
        messages = []
        
        messages.append({"type": "image_url", "image_url":  {"url": f"data:image/jpeg;base64,{base64_images[-1]}"}})
       
        previous_step = ""
        # Add each step with corresponding result (Success/Failed)
        for i, (step, status) in enumerate(zip(prev_steps, prev_msgs)):
            # Text for the step
            previous_step += f"#Step {i + 1}: {step} (this action {status}\n"

        messages.append({"type": "text", "text": f"Please generate the plan for the next step based on the given Goal, Previous Steps, and Images. The plan should select one action and one object from the provided lists.\n\n### Goal: {query}\n### Previous Steps: {previous_step}\n\n### Action List: ['find', 'pick up', 'put down', 'open', 'close', 'slice', 'turn on', 'turn off', 'done']\n### Object List: {visible_objs_str}\n\nGenerate the next step in the format:\nReasoning:\nAction:\nObject:",})


        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": messages
                }
            ],
            max_tokens=1024,
            temperature=0,
        )

        content = response.choices[0].message.content

        best_step = validate_and_retry(content, action_list, visible_objs_str, client, model=self.model_name)

        messages_info = process_messages(messages)


        return best_step, messages_info, content


    def generate_step_details(self, query, prev_steps=(), prev_msgs=(), prev_images=None, visible_objs_str=None, n=1):
        action_list = ["find", "pick up", "put down", "open", "close", "slice", "turn on", "turn off", "done"]
        example = "## Example Plan:\nMain Goal: Place a cold potato on the table.\nComplete Plan:\nstep 1: action: find\nobject: potato\nstep 2: action: pick up\nobject: potato\nstep 3: action: find\nobject: fridge\nstep 4: action: open\nobject: fridge\nstep 5: action: put down\nobject: potato\nstep 6: action: close\nobject: fridge\nstep 7: action: open\nobject: fridge\nstep 8: action: find\nobject: potato\nstep 9: action: pick up\nobject: potato\nstep 10: action: close\nobject: fridge\nstep 11: action: find\nobject: counter top\nstep 12: action: put down\nobject: potato\nstep 13: action: done\nobject: -\n\nMain Goal: Place a heated plate on the round table.\nComplete Plan:\nstep 1: action: find\nobject: plate\nstep 2: action: pick up\nobject: plate\nstep 3: action: find\nobject: microwave\nstep 4: action: open\nobject: microwave\nstep 5: action: put down\nobject: plate\nstep 6: action: close\nobject: microwave\nstep 7: action: turn on\nobject: microwave\nstep 8: action: turn off\nobject: microwave\nstep 9: action: open\nobject: microwave\nstep 10: action: find\nobject: plate\nstep 11: action: pick up\nobject: plate\nstep 12: action: close\nobject: microwave\nstep 13: action: find\nobject: dining table\nstep 14: action: put down\nobject: plate\nstep 15: action: done\nobject:"
        examplev2='''## Example Plan:\nMain Goal: Pick up the alarm clock and turn on the lamp.\nAll Steps: 1. find an alarm clock, 2. pick up the alarm clock, 3. find a desk lamp, 4. turn on the desk lamp 5. done\nMain Goal: Put a cardboard box with a phone in it on the seat of a chair.\nAll Steps: 1. find a cell phone, 2. pick up the cell phone, 3. find a box, 4. put down the cell phone, 5. pick up the box, 6. find an arm chair, 7. put down the box, 8. done.\nMain Goal: Put a clean bowl in a microwave.\nAll Steps: 1. find a bowl, 2. pick up the bowl, 3. find a sink, 4. put down the bowl, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a bowl, 9.pick up the bowl, 10. find a microwave, 11. open the microwave, 12. put down the bowl, 13. close the microwave, 14. done.\nMain Goal: put cooked apple in the sink.\nAll Steps: 1. find an apple, 2. pick up the apple, 3. find a microwave, 4. open the microwave, 5. put down the apple, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find an apple, 11. pick up the apple, 12. close the microwave, 13. find a sink, 14. put down the apple, 15. done.\nMain Goal: Put a chilled potato in the microwave.\nAll Steps: 1. find a microwave, 2. open the microwave, 3. find a potato, 4. pick up the potato, 5. close the microwave, 6. find a fridge, 7. open the fridge, 8. put down the potato, 9. close the fridge, 10. open the fridge, 11. find a potato, 12. pick up the potato, 13. close the fridge, 14. find a microwave, 15. open the microwave, 16. put down the potato, 17. close the microwave, 18. done.\n'''


        if len(prev_steps) >= self.max_steps:
            return None, None, None

        
        base64_images = []
        for img in prev_images:
            img_base64 = pil_to_base64(img)
            base64_images.append(img_base64)

        # client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
        client = self.client

        # Start building the messages list
        messages = []
        
        # Add the Current Image as the input
        messages.append({"type": "image_url", "image_url":  {"url": f"data:image/jpeg;base64,{base64_images[-1]}"}})
        
        # Add each previous step with corresponding result (Success/Failed)
        previous_step = ""
        for i, (step, status) in enumerate(zip(prev_steps, prev_msgs)):
            # Text for the step
            previous_step += f"#Step {i + 1}: {step} (this action {status})\n"

        
        if self.icl:
            messages.append({"type": "text", "text": f"You are given a main goal, and your task is to plan and execute steps step-by-step based on the current environment and previous steps. After each step, you will receive feedback on whether it was successful or not. If a step fails, re-plan and consider the historical steps and progress to adjust your next step accordingly. If a step succeeds, continue executing the next logical step toward completing the main goal.\n\n{examplev2}\n\n## Current Task:\n### Main Goal: {query}\n\n### Previous Step Details: {previous_step}\n\n### Action List: ['find', 'pick up', 'put down', 'open', 'close', 'slice', 'turn on', 'turn off', 'done']\n### Object List: {visible_objs_str}\nGiven the previous steps and your current observation, generate the next step to implement the goal. The next step should include:\nAction: Choose one action from the given **Action List**.\nObject: Choose one object from the **Object List** that can be visible, based on the current observation.\n### Rule:\n1. If an object is visible in your current observation, you can choose to interact with it directly.\n2. If an object is not visible, you must first use the find action to locate it.\n3. If the planning is completed, the action output is 'done', indicating that the task is completed.\n### Output: First, provide your reasoning, then specify the action and object for the next step. The output format should be:\nReasoning:...\nAction: (only action chosen from the action list)\nObject: (only object chosen from the object list)\n",})
        else:
            messages.append({"type": "text", "text": f"You are given a main goal, and your task is to plan and execute steps step-by-step based on the current environment and previous steps. After each step, you will receive feedback on whether it was successful or not. If a step fails, re-plan and consider the historical steps and progress to adjust your next step accordingly. If a step succeeds, continue executing the next logical step toward completing the main goal.\n\n## Current Task:\n### Main Goal: {query}\n\n### Previous Step Details: {previous_step}\n\n### Action List: ['find', 'pick up', 'put down', 'open', 'close', 'slice', 'turn on', 'turn off', 'done']\n### Object List: {visible_objs_str}\nGiven the previous steps and your current observation, generate the next step to implement the goal. The next step should include:\nAction: Choose one action from the given **Action List**.\nObject: Choose one object from the **Object List** that can be visible, based on the current observation.\n### Rule:\n1. If an object is visible in your current observation, you can choose to interact with it directly.\n2. If an object is not visible, you must first use the find action to locate it.\n3. If the planning is completed, the action output is 'done', indicating that the task is completed.\n### Output: First, provide your reasoning, then specify the action and object for the next step. The output format should be:\nReasoning:...\nAction: (only action chosen from the action list)\nObject: (only object chosen from the object list)\n",})

        
        if 'gpt' in self.model_name or 'gemini' in self.model_name:
            while True:
                try:
                    completion = client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": messages}
                        ]
                    )
                    message = completion.choices[0].message
                    content = unicodedata.normalize('NFKC', message.content)
                    break
                except Exception as e:
                    print(e)
        else:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": messages
                    }
                ],
                max_tokens=1024,
                temperature=0,
            )
            content = response.choices[0].message.content
       
        # print(response)

        # Get the response
        best_step = validate_and_retry(content, action_list, visible_objs_str, client, model=self.model_name)
        
        # Process the input message without image token for log
        messages_info = process_messages(messages)

        return best_step, messages_info, content

    
    def generate_step_candidates(self, query, prev_steps=(), prev_msgs=(), prev_images=None, visible_objs_str=None, n=1, task_type=''):
        action_list = ["find", "pick up", "put down", "open", "close", "slice", "turn on", "turn off", "done"]
        # example = "## Example Plan:\nMain Goal: Place a cold potato on the table.\nComplete Plan:\nstep 1: action: find\nobject: potato\nstep 2: action: pick up\nobject: potato\nstep 3: action: find\nobject: fridge\nstep 4: action: open\nobject: fridge\nstep 5: action: put down\nobject: potato\nstep 6: action: close\nobject: fridge\nstep 7: action: open\nobject: fridge\nstep 8: action: find\nobject: potato\nstep 9: action: pick up\nobject: potato\nstep 10: action: close\nobject: fridge\nstep 11: action: find\nobject: counter top\nstep 12: action: put down\nobject: potato\nstep 13: action: done\nobject: -\n\nMain Goal: Place a heated plate on the round table.\nComplete Plan:\nstep 1: action: find\nobject: plate\nstep 2: action: pick up\nobject: plate\nstep 3: action: find\nobject: microwave\nstep 4: action: open\nobject: microwave\nstep 5: action: put down\nobject: plate\nstep 6: action: close\nobject: microwave\nstep 7: action: turn on\nobject: microwave\nstep 8: action: turn off\nobject: microwave\nstep 9: action: open\nobject: microwave\nstep 10: action: find\nobject: plate\nstep 11: action: pick up\nobject: plate\nstep 12: action: close\nobject: microwave\nstep 13: action: find\nobject: dining table\nstep 14: action: put down\nobject: plate\nstep 15: action: done\nobject:"



        if len(prev_steps) >= self.max_steps:
            return None, None, None
        # import pdb; pdb.set_trace()

        

        # 示例使用
        # 假设 prev_images 是一个包含 PIL 图像的列表
        base64_images = []
        for img in prev_images:
            img_base64 = pil_to_base64(img)
            base64_images.append(img_base64)

        # client = OpenAI(api_key="sk-M2n9zRPMOSmtZxgT7d6e185a88D64d44Ad49D0A130635e94", base_url="http://15.204.101.64:4000/v1")
        # client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
        client = OpenAI(api_key="sk-chipkupebaysbnakbeovnodzbwtamctkaqpmdzzekuzvrbos", base_url="https://api.siliconflow.cn/v1")

        # import pdb; pdb.set_trace()

        # Start building the messages list
        messages = []

        messages.append({"type": "image_url", "image_url":  {"url": f"data:image/jpeg;base64,{base64_images[-1]}"}})
        
        # Add the main goal as the first message
        messages.append({"type": "text", "text": "You are given a main goal, and your task is to plan and execute steps step-by-step based on the current environment and previous steps. After each step, you will receive feedback on whether it was successful or not. If a step fails, re-plan and consider the historical steps and progress to adjust your next step accordingly. If a step succeeds, continue executing the next logical step toward completing the main goal.",})
        if 'movable' in task_type:
            messages.append({"type": "text", "text": "\n## Example Complete Plan:\nGoal: Put a cardboard box with a phone in it on the seat of a chair.\nAll Steps: 1. find a cell phone, 2. pick up the cell phone, 3. find a box, 4. put down the cell phone, 5. pick up the box, 6. find an arm chair, 7. put down the box, 8. done.\nGoal: place a sponge in a glass bowl on an overhead drawer\nAll Steps: 1. find a cabinet, 2. open the cabinet, 3. find a dish sponge, 4. pick up the dish sponge, 5. close the cabinet, 6. find a bowl, 7. put down the dish sponge, 8. pick up the bowl, 9. find a cabinet, 10. open the cabinet, 11. put down the bowl, 12. close the cabinet, 13. done.\nGoal: Move a sponge and pan to the counter.\nAll Steps: 1. find a dish sponge, 2. pick up the dish sponge, 3. find a pan, 4. put down the dish sponge, 5. pick up the pan, 6. find a dining table, 7. put down the pan, 8. done.\nGoal: Place a pan with a piece of sliced lettuce plus a metal knife in it on a round black table.\nAll Steps: 1. find a knife, 2. pick up the knife, 3. find a lettuce, 4. slice the lettuce, 5. find a pan, 6. put down the knife, 7. find a lettuce, 8. pick up the lettuce, 9. find a pan, 10. put down the lettuce, 11. pick up the pan, 12. find a dining table, 13. put down the pan, 14. done."})
        elif 'clean' in task_type:
            messages.append({"type": "text", "text": "\n## Example Complete Plan:\nGoal: Put a clean bowl in a microwave.\nAll Steps: 1. find a bowl, 2. pick up the bowl, 3. find a sink, 4. put down the bowl, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a bowl, 9.pick up the bowl, 10. find a microwave, 11. open the microwave, 12. put down the bowl, 13. close the microwave, 14. done.\nGoal: wash a bowl from the counter then put it away\nAll Steps: 1. find a bowl, 2. pick up the bowl, 3. find a sink, 4. put down the bowl, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a bowl, 9. pick up the bowl, 10. find a shelf, 11. put down the bowl, 12. done.\nGoal: Place a rinsed knife on a counter.\nAll Steps: 1. find a butter knife, 2. pick up the butter knife, 3. find a sink, 4. put down the butter knife, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a butter knife, 9. pick up the butter knife, 10. find a counter top, 11. put down the butter knife, 12. done.\nGoal: pick up the knife, wash it off, place it on the table.\nAll Steps: 1. find a knife, 2. pick up the knife, 3. find a sink, 4. put down the knife, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a knife, 9. pick up the knife, 10. find a dining table, 11. put down the knife, 12. done."})
        elif 'heat' in task_type:
            messages.append({"type": "text", "text": "\n## Example Complete Plan:\nGoal: put cooked apple in the sink.\nAll Steps: 1. find an apple, 2. pick up the apple, 3. find a microwave, 4. open the microwave, 5. put down the apple, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find an apple, 11. pick up the apple, 12. close the microwave, 13. find a sink, 14. put down the apple, 15. done.\nGoal: Put a heated egg on a table.\nAll Steps: 1. find an egg, 2. pick up the egg, 3. find a microwave, 4. open the microwave, 5. put down the egg, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find an egg, 11. pick up the egg, 12. close the microwave, 13. find a dining table, 14. put down the egg, 15. done.\nGoal: Heat up coffee.\nAll Steps: 1. find a mug, 2. pick up the mug, 3. find a microwave, 4. open the microwave, 5. put down the mug, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find a mug, 11. pick up the mug, 12. close the microwave, 13. find a coffee machine, 14. put down the mug, 15. done.\nGoal: Put cooked apple slice on a table.\nAll Steps: 1. find a knife, 2. pick up the knife, 3. find an apple, 4. slice the apple, 5. find a microwave, 6. open the microwave, 7. put down the knife, 8. close the microwave, 9. find an apple, 10. pick up the apple, 11. find a microwave, 12. open the microwave, 13. put down the apple, 14. close the microwave, 15. turn on the microwave, 16. turn off the microwave, 17. open the microwave, 18. find an apple, 19. pick up the apple, 20. close the microwave, 21. find a side table, 22. put down the apple, 23. done.\nGoal: Cooking and cooling an egg.\nAll Steps: 1. find an egg, 2. pick up the egg, 3. find a microwave, 4. open the microwave, 5. put down the egg, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find an egg, 11. pick up the egg, 12. close the microwave, 13. find a fridge, 14. open the fridge, 15. put down the egg, 16. close the fridge, 17. done"})
        elif 'cool' in task_type:
            messages.append({"type": "text", "text": "\n## Example Complete Plan:\nGoal: Put a chilled potato in the microwave.\nAll Steps: 1. find a microwave, 2. open the microwave, 3. find a potato, 4. pick up the potato, 5. close the microwave, 6. find a fridge, 7. open the fridge, 8. put down the potato, 9. close the fridge, 10. open the fridge, 11. find a potato, 12. pick up the potato, 13. close the fridge, 14. find a microwave, 15. open the microwave, 16. put down the potato, 17. close the microwave, 18. done.\nGoal: Put a cool potato in the sink.\nAll Steps: 1. find a potato, 2. pick up the potato, 3. find a fridge, 4. open the fridge, 5. put down the potato, 6. close the fridge, 7. open the fridge, 8. find a potato, 9. pick up the potato, 10. close the fridge, 11. find a sink, 12. put down the potato, 13. done.\nGoal: Put a chilled mug on the coffee machine.\nAll Steps: 1. find a mug, 2. pick up the mug, 3. find a fridge, 4. open the fridge, 5. put down the mug, 6. close the fridge, 7. open the fridge, 8. find a mug, 9. pick up the mug, 10. close the fridge, 11. find a coffee machine, 12. put down the mug, 13. done.\nGoal: Place a chilled lettuce slice on the table.\nAll Steps: 1. find a butter knife, 2. pick up the butter knife, 3. find a lettuce, 4. slice the lettuce, 5. find a fridge, 6. open the fridge, 7. put down the butter knife, 8. close the fridge, 9. find a lettuce, 10. pick up the lettuce, 11. find a fridge, 12. open the fridge, 13. put down the lettuce, 14. close the fridge, 15. open the fridge, 16. find a lettuce, 17. pick up the lettuce, 18. close the fridge, 19. find a dining table, 20. put down the lettuce, 21. done.\nGoal: Cooking and cooling an egg.\nAll Steps: 1. find an egg, 2. pick up the egg, 3. find a microwave, 4. open the microwave, 5. put down the egg, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find an egg, 11. pick up the egg, 12. close the microwave, 13. find a fridge, 14. open the fridge, 15. put down the egg, 16. close the fridge, 17. done"})
        messages.append({"type": "text", "text": f"\n\n\n##Current Task:\n### Main Goal: {query}\n\n### Previous Step Details: "})

        # Add the initial image (if any)
        # if len(base64_images)==1:
        #     messages.append({"type": "image_url", "image_url":  {"url": f"data:image/jpeg;base64,{base64_images[0]}"}})
        # else:
        #     messages.append({"type": "image_url", "image_url":  {"url": f"data:image/jpeg;base64,{base64_images[0]}"},"modalities": "multi-images",})

        # Add each step with corresponding image and result (Success/Failed)
        for i, (step, status) in enumerate(zip(prev_steps, prev_msgs)):
            # Text for the step
            step_text = f"#Step {i + 1}: {step}"
            messages.append({"type": "text", "text": f"{step_text} (this action {status})"})
            
            # Add the corresponding image
            # if i + 1 < len(prev_images):
            #     messages.append({"type": "image_url", "image_url":  {"url": f"data:image/jpeg;base64,{base64_images[i + 1]}"},"modalities": "multi-images"})
        
        
        # Add the "Current Step" as the final message
        # ### Object List: {visible_objs_str}
        messages.append({"type": "text", "text": f"\n### Action List: {action_list}\n### Object List: {visible_objs_str}\nGiven the previous steps and your current observation, only generate the next step to implement the goal. The next step should include:\nAction: Choose one action from the given **Action List**.\nObject: Choose one object from the **Object List** that can be visible, based on the current observation.\n### Rule:\n1. If an object is visible in your current observation, you can choose to interact with it directly.\n2. If an object is not visible, you must first use the find action to locate it.\n### Output: First, provide your reasoning, then specify the action and object for only the next step. The output format should be:\nReasoning:...\nAction: (only action chosen from the action list)\nObject: (only object chosen from the object list)\n"})
        
        all_contents = []
        for i in range(n):
            response = call_api_with_retry(
                model="Pro/Qwen/Qwen2-VL-7B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": messages
                    }
                ],
                max_tokens=1024,
                temperature=1.0,
                # n=1
            )
       
            # print(response)

        # Get the response
        # best_step = response['choices'][0]['message']['content'].strip()
        # best_step = response.choices[0].message.content.strip().lower()
            if response:
                content = response.choices[0].message.content
            else:
                content = ""
            print(f"First Gen: {content}")
        # all_contents = [choice.message.content for choice in response.choices]
            all_contents.append(content)


        # 初始化结果列表
        final_results = []
        final_content = []
        # print(process_messages(messages))
        # 处理每个内容以提取 Action 和 Object
        for content in all_contents:
            # best_step = extract_results(content)
            best_step = validate_and_retry(content, action_list, visible_objs_str)
            
            # print(content)
            
            messages_info = process_messages(messages)
            
            # final_results.append((best_step, messages_info, content))
            final_results.append(best_step) 
            final_content.append(content.split("Action:")[0])
        

        return final_results, final_content


    def generate_step_candidates_hint(self, query, prev_steps=(), prev_msgs=(), prev_images=None, visible_objs_str=None, n=1, hint=None, task_type=''):
        action_list = ["find", "pick up", "put down", "open", "close", "slice", "turn on", "turn off", "done"]

        if len(prev_steps) >= self.max_steps:
            return None, None, None
        # import pdb; pdb.set_trace()

        

        # 示例使用
        # 假设 prev_images 是一个包含 PIL 图像的列表
        base64_images = []
        for img in prev_images:
            img_base64 = pil_to_base64(img)
            base64_images.append(img_base64)

        # client = OpenAI(api_key="sk-M2n9zRPMOSmtZxgT7d6e185a88D64d44Ad49D0A130635e94", base_url="http://15.204.101.64:4000/v1")
        # client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
        client = OpenAI(api_key="sk-chipkupebaysbnakbeovnodzbwtamctkaqpmdzzekuzvrbos", base_url="https://api.siliconflow.cn/v1")


        # import pdb; pdb.set_trace()

        # Start building the messages list
        messages = []
        
        messages.append({"type": "image_url", "image_url":  {"url": f"data:image/jpeg;base64,{base64_images[-1]}"}})

        # Add the main goal as the first message
        messages.append({"type": "text", "text": "You are given a main goal, and your task is to plan and execute steps step-by-step based on the current environment and previous steps. After each step, you will receive feedback on whether it was successful or not. If a step fails, re-plan and consider the historical steps and progress to adjust your next step accordingly. If a step succeeds, continue executing the next logical step toward completing the main goal.",})
        # messages.append({"type": "text", "text": "\n## Example Complete Plan:\nGoal: Put a cardboard box with a phone in it on the seat of a chair.\nAll Steps: 1. find a cell phone, 2. pick up the cell phone, 3. find a box, 4. put down the cell phone, 5. pick up the box, 6. find an arm chair, 7. put down the box, 8. done.\nGoal: place a sponge in a glass bowl on an overhead drawer\nAll Steps: 1. find a cabinet, 2. open the cabinet, 3. find a dish sponge, 4. pick up the dish sponge, 5. close the cabinet, 6. find a bowl, 7. put down the dish sponge, 8. pick up the bowl, 9. find a cabinet, 10. open the cabinet, 11. put down the bowl, 12. close the cabinet, 13. done.\nGoal: Move a sponge and pan to the counter.\nAll Steps: 1. find a dish sponge, 2. pick up the dish sponge, 3. find a pan, 4. put down the dish sponge, 5. pick up the pan, 6. find a dining table, 7. put down the pan, 8. done.\nGoal: Place a pan with a piece of sliced lettuce plus a metal knife in it on a round black table.\nAll Steps: 1. find a knife, 2. pick up the knife, 3. find a lettuce, 4. slice the lettuce, 5. find a pan, 6. put down the knife, 7. find a lettuce, 8. pick up the lettuce, 9. find a pan, 10. put down the lettuce, 11. pick up the pan, 12. find a dining table, 13. put down the pan, 14. done."})
        # messages.append({"type": "text", "text": "\n## Example Complete Plan:\nGoal: Put a clean bowl in a microwave.\nAll Steps: 1. find a bowl, 2. pick up the bowl, 3. find a sink, 4. put down the bowl, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a bowl, 9.pick up the bowl, 10. find a microwave, 11. open the microwave, 12. put down the bowl, 13. close the microwave, 14. done.\nGoal: wash a bowl from the counter then put it away\nAll Steps: 1. find a bowl, 2. pick up the bowl, 3. find a sink, 4. put down the bowl, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a bowl, 9. pick up the bowl, 10. find a shelf, 11. put down the bowl, 12. done.\nGoal: Place a rinsed knife on a counter.\nAll Steps: 1. find a butter knife, 2. pick up the butter knife, 3. find a sink, 4. put down the butter knife, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a butter knife, 9. pick up the butter knife, 10. find a counter top, 11. put down the butter knife, 12. done.\nGoal: pick up the knife, wash it off, place it on the table.\nAll Steps: 1. find a knife, 2. pick up the knife, 3. find a sink, 4. put down the knife, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a knife, 9. pick up the knife, 10. find a dining table, 11. put down the knife, 12. done."})
        if 'movable' in task_type:
            messages.append({"type": "text", "text": "\n## Example Complete Plan:\nGoal: Put a cardboard box with a phone in it on the seat of a chair.\nAll Steps: 1. find a cell phone, 2. pick up the cell phone, 3. find a box, 4. put down the cell phone, 5. pick up the box, 6. find an arm chair, 7. put down the box, 8. done.\nGoal: place a sponge in a glass bowl on an overhead drawer\nAll Steps: 1. find a cabinet, 2. open the cabinet, 3. find a dish sponge, 4. pick up the dish sponge, 5. close the cabinet, 6. find a bowl, 7. put down the dish sponge, 8. pick up the bowl, 9. find a cabinet, 10. open the cabinet, 11. put down the bowl, 12. close the cabinet, 13. done.\nGoal: Move a sponge and pan to the counter.\nAll Steps: 1. find a dish sponge, 2. pick up the dish sponge, 3. find a pan, 4. put down the dish sponge, 5. pick up the pan, 6. find a dining table, 7. put down the pan, 8. done.\nGoal: Place a pan with a piece of sliced lettuce plus a metal knife in it on a round black table.\nAll Steps: 1. find a knife, 2. pick up the knife, 3. find a lettuce, 4. slice the lettuce, 5. find a pan, 6. put down the knife, 7. find a lettuce, 8. pick up the lettuce, 9. find a pan, 10. put down the lettuce, 11. pick up the pan, 12. find a dining table, 13. put down the pan, 14. done."})
        elif 'clean' in task_type:
            messages.append({"type": "text", "text": "\n## Example Complete Plan:\nGoal: Put a clean bowl in a microwave.\nAll Steps: 1. find a bowl, 2. pick up the bowl, 3. find a sink, 4. put down the bowl, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a bowl, 9.pick up the bowl, 10. find a microwave, 11. open the microwave, 12. put down the bowl, 13. close the microwave, 14. done.\nGoal: wash a bowl from the counter then put it away\nAll Steps: 1. find a bowl, 2. pick up the bowl, 3. find a sink, 4. put down the bowl, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a bowl, 9. pick up the bowl, 10. find a shelf, 11. put down the bowl, 12. done.\nGoal: Place a rinsed knife on a counter.\nAll Steps: 1. find a butter knife, 2. pick up the butter knife, 3. find a sink, 4. put down the butter knife, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a butter knife, 9. pick up the butter knife, 10. find a counter top, 11. put down the butter knife, 12. done.\nGoal: pick up the knife, wash it off, place it on the table.\nAll Steps: 1. find a knife, 2. pick up the knife, 3. find a sink, 4. put down the knife, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a knife, 9. pick up the knife, 10. find a dining table, 11. put down the knife, 12. done."})
        elif 'heat' in task_type:
            messages.append({"type": "text", "text": "\n## Example Complete Plan:\nGoal: put cooked apple in the sink.\nAll Steps: 1. find an apple, 2. pick up the apple, 3. find a microwave, 4. open the microwave, 5. put down the apple, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find an apple, 11. pick up the apple, 12. close the microwave, 13. find a sink, 14. put down the apple, 15. done.\nGoal: Put a heated egg on a table.\nAll Steps: 1. find an egg, 2. pick up the egg, 3. find a microwave, 4. open the microwave, 5. put down the egg, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find an egg, 11. pick up the egg, 12. close the microwave, 13. find a dining table, 14. put down the egg, 15. done.\nGoal: Heat up coffee.\nAll Steps: 1. find a mug, 2. pick up the mug, 3. find a microwave, 4. open the microwave, 5. put down the mug, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find a mug, 11. pick up the mug, 12. close the microwave, 13. find a coffee machine, 14. put down the mug, 15. done.\nGoal: Put cooked apple slice on a table.\nAll Steps: 1. find a knife, 2. pick up the knife, 3. find an apple, 4. slice the apple, 5. find a microwave, 6. open the microwave, 7. put down the knife, 8. close the microwave, 9. find an apple, 10. pick up the apple, 11. find a microwave, 12. open the microwave, 13. put down the apple, 14. close the microwave, 15. turn on the microwave, 16. turn off the microwave, 17. open the microwave, 18. find an apple, 19. pick up the apple, 20. close the microwave, 21. find a side table, 22. put down the apple, 23. done.\nGoal: Cooking and cooling an egg.\nAll Steps: 1. find an egg, 2. pick up the egg, 3. find a microwave, 4. open the microwave, 5. put down the egg, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find an egg, 11. pick up the egg, 12. close the microwave, 13. find a fridge, 14. open the fridge, 15. put down the egg, 16. close the fridge, 17. done"})
        elif 'cool' in task_type:
            messages.append({"type": "text", "text": "\n## Example Complete Plan:\nGoal: Put a chilled potato in the microwave.\nAll Steps: 1. find a microwave, 2. open the microwave, 3. find a potato, 4. pick up the potato, 5. close the microwave, 6. find a fridge, 7. open the fridge, 8. put down the potato, 9. close the fridge, 10. open the fridge, 11. find a potato, 12. pick up the potato, 13. close the fridge, 14. find a microwave, 15. open the microwave, 16. put down the potato, 17. close the microwave, 18. done.\nGoal: Put a cool potato in the sink.\nAll Steps: 1. find a potato, 2. pick up the potato, 3. find a fridge, 4. open the fridge, 5. put down the potato, 6. close the fridge, 7. open the fridge, 8. find a potato, 9. pick up the potato, 10. close the fridge, 11. find a sink, 12. put down the potato, 13. done.\nGoal: Put a chilled mug on the coffee machine.\nAll Steps: 1. find a mug, 2. pick up the mug, 3. find a fridge, 4. open the fridge, 5. put down the mug, 6. close the fridge, 7. open the fridge, 8. find a mug, 9. pick up the mug, 10. close the fridge, 11. find a coffee machine, 12. put down the mug, 13. done.\nGoal: Place a chilled lettuce slice on the table.\nAll Steps: 1. find a butter knife, 2. pick up the butter knife, 3. find a lettuce, 4. slice the lettuce, 5. find a fridge, 6. open the fridge, 7. put down the butter knife, 8. close the fridge, 9. find a lettuce, 10. pick up the lettuce, 11. find a fridge, 12. open the fridge, 13. put down the lettuce, 14. close the fridge, 15. open the fridge, 16. find a lettuce, 17. pick up the lettuce, 18. close the fridge, 19. find a dining table, 20. put down the lettuce, 21. done.\nGoal: Cooking and cooling an egg.\nAll Steps: 1. find an egg, 2. pick up the egg, 3. find a microwave, 4. open the microwave, 5. put down the egg, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find an egg, 11. pick up the egg, 12. close the microwave, 13. find a fridge, 14. open the fridge, 15. put down the egg, 16. close the fridge, 17. done"})
        
        messages.append({"type": "text", "text": f"\n\n##Current Task:\n### Main Goal: {query}\n\n### Previous Step Details: "})

        # Add the initial image (if any)
        # if len(base64_images)==1:
        #     messages.append({"type": "image_url", "image_url":  {"url": f"data:image/jpeg;base64,{base64_images[0]}"}})
        # else:
        #     messages.append({"type": "image_url", "image_url":  {"url": f"data:image/jpeg;base64,{base64_images[0]}"},"modalities": "multi-images",})

        # Add each step with corresponding image and result (Success/Failed)
        for i, (step, status) in enumerate(zip(prev_steps, prev_msgs)):
            # Text for the step
            step_text = f"#Step {i + 1}: {step}"
            messages.append({"type": "text", "text": f"{step_text} (this action {status})"})
            
            # Add the corresponding image
            # if i + 1 < len(prev_images):
            #     messages.append({"type": "image_url", "image_url":  {"url": f"data:image/jpeg;base64,{base64_images[i + 1]}"},"modalities": "multi-images"})
        
        
        # Add the "Current Step" as the final message
        # ### Object List: {visible_objs_str}
        messages.append({"type": "text", "text": f"\n### Action List: {action_list}\n### Object List: {visible_objs_str}\nGiven the previous steps and your current observation, generate the next step to implement the goal. The next step should include:\nAction: Choose one action from the given **Action List**.\nObject: Choose one object from the **Object List** that can be visible, based on the current observation.\n### Rule:\n1. If an object is visible in your current observation, you can choose to interact with it directly.\n2. If an object is not visible, you must first use the find action to locate it.\n### Output: First, provide your reasoning, then specify the action and object for the next step. The output format should be:\nReasoning:...\nAction: (only action chosen from the action list)\nObject: (only object chosen from the object list)\n---\n### Hint: {hint}"})
        
        all_contents = []
        for i in range(n):
            response = call_api_with_retry(
                model="Pro/Qwen/Qwen2-VL-7B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": messages
                    }
                ],
                max_tokens=1024,
                temperature=0.2,
                # n=1
            )
       
            # print(response)

        # Get the response
        # best_step = response['choices'][0]['message']['content'].strip()
        # best_step = response.choices[0].message.content.strip().lower()
            if response:
                content = response.choices[0].message.content
            else:
                content = ""
            # content = response.choices[0].message.content
            print(f"First Gen HINT: {content}")
        # all_contents = [choice.message.content for choice in response.choices]
            all_contents.append(content)


        # 初始化结果列表
        final_results = []
        final_content = []
        # print(process_messages(messages))
        # 处理每个内容以提取 Action 和 Object
        # messages_info = process_messages(messages)
        # print(messages_info)
        for content in all_contents:
            # best_step = extract_results(content)
            best_step = validate_and_retry(content, action_list, visible_objs_str)
            
            # print(content)
            
            
            
            # final_results.append((best_step, messages_info, content))
            final_results.append(best_step) 
            final_content.append(content.split("Action:")[0])
        

        return final_results, final_content

    def evaluate_score(self, goal, prev_steps, prev_action_msg, imgs, step, action_ret):
        client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
        base64_images = []
        for img in imgs:
            img_base64 = pil_to_base64(img)
            base64_images.append(img_base64)

        messages = []
        messages.append({
            "type": "text",
            "text": f"Please serve as an unbiased evaluator for the AI assistant's next step in the embodied planning task, focusing on the following two aspects:\n- Executable Action: Can the step be executed in the current environment and context?\n- Closer to the Goal: Does this step effectively bring the task closer to the completion of the goal, or is it irrelevant?\n\n### Input Data:\ngoal: {goal}"
        })
        for i, (prev_step, feedback, img) in enumerate(zip(prev_steps, prev_action_msg, base64_images[1:])):
            
            # 添加文本步骤和反馈信息
            messages.append({
                "type": "text",
                "text": f"Step {i + 1}: {prev_step}\nFeedback: {feedback}"
            })

            # 添加图片信息
            messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                "modalities": "multi-images"
            })

        # 添加当前步骤的信息和反馈
        current_step = step
        current_feedback = action_ret
        messages.append({
            "type": "text",
            "text": f"\n\n###AI-generated next step:\nStep: {current_step}\nFeedback: {current_feedback}"
        })

        # base64_image = encode_image(imgs[-1])
        # print(data['imgs'][-1])
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_images[-1]}"},
            "modalities": "multi-images"
        })

        messages.append({
            "type": "text",
            "text": "### Evaluation Criteria:\nPlease assess whether the step meets the following criteria:\n-Executable Action: Can the step be executed in the current environment? Specifically: For actions that operate on objects (e.g., “pick up”, “put down”), the object must be visible in the current environment. For objects that are not visible, an appropriate action (e.g., “find”) should be used to locate the object first before attempting operations on it.\n-Closer to the Goal: Does the step move the task closer to completion? Specifically: The action should directly contribute to completing the goal by interacting with relevant objects or preparing the environment. Actions involving unrelated objects or irrelevant tasks that do not help achieve the goal are not considered closer to the goal.\n\n### Scoring System:\nPoor (1): The step is not executable in the current environment and is unrelated to achieving the goal.\nFair (2): The step can be executed to some extent, but it is not closely related to completing the goal.\nAverage (3): The step is executable and somewhat moves the task closer to the goal, but lacks full alignment or optimal strategy.\nGood (4): The step is executable and significantly advances the goal with a high degree of relevance and feasibility.\nExcellent (5): The step is fully executable in the current environment and greatly advances the goal, showing excellent relevance and execution.\n\n### Output Format:\nAnalysis: Provide a brief evaluation based on the criteria above, explaining if the step is executable and how it contributes to or detracts from achieving the goal.\nJudgement: Provide a final evaluation score based on your analysis, ranging from 1 to 5. Use the following scale format: [1], [2], [3], [4], [5]."
        })

        response = client.chat.completions.create(
                model="Qwen2-VL-7B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": messages
                    }
                ],
                max_tokens=1024,
                temperature=0,
                # n=1
            )
        
        content = response.choices[0].message.content
        score = extract_or_average_score(content)
        # print(process_messages(messages))

        return score, content


    def evaluate_score_img(self, goal, goal_image, prev_steps, prev_action_msg, imgs, step, action_ret, env_visible_objects):
        client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
        base64_images = []
        for img in imgs:
            img_base64 = pil_to_base64(img)
            base64_images.append(img_base64)

        messages = []
        # messages.append({
        #     "type": "text",
        #     "text": f"Please serve as an unbiased evaluator for the AI assistant's next step in the embodied planning task, focusing on the following three aspects:\n- Executable Action: Can the step be executed in the current environment and context?\n- Closer to the Goal: Does this step effectively bring the task closer to the completion of the goal, or is it irrelevant?\n- Visual Alignment: Does the scene after executing this step more closely match the target goal completion image?\n\n### Input Data:\n### Goal: {goal}\n"
        # })

        # messages.append({
        #     "type": "text", 
        #     "text": f"""Please evaluate the AI assistant's next step in the task planning according to two key metrics:\n\n### Input Data:\n### Goal: {goal}\n"""
        # })

        messages.append({
            "type": "text", 
            "text": f"""Please serve as an unbiased evaluator for the AI-generated next step in the task planning according to goal progress. The task involves pick & place actions that typically follow this sequence:\n1. Find target object (may require opening containers like drawers/fridge if object is inside)\n2. Pick up the object \n3. Find target location\n4. Place object at target location\n\n## Input Data:\n### Goal: {goal}\n"""
        })

        # messages.append({
        #         "type": "text",
        #         "text": "### Goal completion reference image:"
        #     })

        # goalimage_base64 = pil_to_base64(goal_image)

        # messages.append({
        #         "type": "image_url",
        #         "image_url": {"url": f"data:image/jpeg;base64,{goalimage_base64}"},
        #         "modalities": "multi-images"
        #     })


        messages.append({
            "type": "text",
            "text": "### Previous Steps:"
                })

        
        for i, (prev_step, feedback, img) in enumerate(zip(prev_steps, prev_action_msg, base64_images[1:])):
            
            # 添加文本步骤和反馈信息
            messages.append({
                "type": "text",
                "text": f"Step {i + 1}: {prev_step}\nFeedback: {feedback}"
            })

            # # 添加图片信息
            # messages.append({
            #     "type": "image_url",
            #     "image_url": {"url": f"data:image/jpeg;base64,{img}"},
            #     "modalities": "multi-images"
            # })

        

        


        # 添加当前步骤的信息和反馈
        current_step = step
        current_feedback = action_ret
        # messages.append({
        #     "type": "text",
        #     "text": f"---\n\n###AI-generated next step:\nStep: {current_step}\nFeedback: {current_feedback}\nIn this Frame, you can see the following objects: {env_visible_objects}"
        # })

        messages.append({
            "type": "text",
            "text": f"---\n\n## AI-generated Next Step to Evaluate:\nStep: {current_step}\nExecution Result: {current_feedback}\nAfter execute the step, you can see the following objects: {env_visible_objects}\n"
        })

        # base64_image = encode_image(imgs[-1])
        # print(data['imgs'][-1])
        # messages.append({
        #     "type": "image_url",
        #     "image_url": {"url": f"data:image/jpeg;base64,{base64_images[-1]}"},
        #     "modalities": "multi-images"
        # })

        # messages.append({
        #     "type": "text",
        #     "text": "### Evaluation Criteria:\nPlease assess whether the step meets the following criteria:\n-Executable Action: Can the step be executed in the current environment? Specifically: For actions that operate on objects (e.g., \"pick up\", \"put down\"), the object must be visible in the current environment. For objects that are not visible, an appropriate action (e.g., \"find\") should be used to locate the object first before attempting operations on it.\n-Closer to the Goal: Does the step move the task closer to completion? Specifically: The action should directly contribute to completing the goal by interacting with relevant objects or preparing the environment. Actions involving unrelated objects or irrelevant tasks that do not help achieve the goal are not considered closer to the goal.\n-Visual Progress: Compare the scene after this step with the goal completion image. Does the current state of the environment more closely resemble the target state shown in the goal completion image?\n\n### Scoring System:\nPoor (1): The step is not executable in the current environment, is unrelated to achieving the goal, and does not bring the scene closer to the goal completion image.\nFair (2): The step can be executed to some extent, but makes minimal progress toward the goal state shown in the completion image.\nAverage (3): The step is executable and somewhat moves the task closer to the goal state, showing some visual alignment with the target image.\nGood (4): The step is executable and significantly advances toward the goal state, with clear visual progress toward the completion image.\nExcellent (5): The step is fully executable and greatly advances the goal, with the resulting scene closely matching or moving directly toward the goal completion image.\n\n### Output Format:\nAnalysis: Provide a brief evaluation based on the criteria above, explaining:\n1. If the step is executable\n2. How it contributes to or detracts from achieving the goal\n3. How the resulting scene compares to the goal completion image\n\nJudgement: Provide a final evaluation score based on your analysis, ranging from 1 to 5. Use the following scale format: [1], [2], [3], [4], [5]."
        # })

        # messages.append({
        #     "type": "text",
        #     "text": """### Evaluation Criteria:\n\n1. Action Executability (0 or 5 points):\n- [0]: The action failed to execute (indicated by failure feedback)\n- [5]: The action successfully executed (indicated by success feedback)\n\n2. Goal Progress (1-5 points):\nEvaluate how effectively the step moves toward completing the task by considering:\n- If the generated next step is repetitive compared to previous steps, give low score since it does not make the progress\n- The goal description and target completion image\n- The current environment state after executing the step.\n\nScoring for Goal Progress:\n[1]: Step moves away from the goal or makes goal completion more difficult\n[2]: Step exactly repeats previous actions without new progress\n[3]: Step makes moderate progress toward the goal\n[4]: Step makes significant progress toward the goal\n[5]: Step makes excellent progress, directly advancing toward goal completion\n\n### Output Format:\nAnalysis: brief explain whether the generated step meets the metics of Executability and Goal Progress\nExecutability Score: Use the following scale format: [0] or [5]\nGoal Progress Score: Use the following scale format: [1], [2], [3], [4], [5]\n\n"""})

        messages.append({
            "type": "text",
            "text": """## Evaluation Criteria:\n\nGoal Progress (1-5 points):\nEvaluate how effectively the step moves toward completing the task by considering:\n1. Action Sequence - Does it follow the standard find→pick→find→place flow?\n2. Previous Actions - How does it build on prior steps? Does it use the same words as the immediate previous step?\n3. Goal State - Does it advance toward the target condition?\n4. Environment State - Do visible objects match expected progress?\n\nScoring for Goal Progress:\n[1]: Step moves away from the goal or makes goal completion more difficult\n[2]: Step uses EXACTLY the same action word(s) as the immediate previous step\n[3]: Step makes moderate progress toward the goal\n[4]: Step makes significant progress toward the goal\n[5]: Step makes excellent progress, directly advancing toward goal completion\n\nExamples:\n- "find knife" followed by "find knife" = [2] (exact same action)\n- "find knife" followed by "pick up knife" = [4] (different action, follows sequence)\n- "pick up knife" followed by "open drawer" = [1] (moves away from sequence)\n- "pick up knife" followed by "find microwave table" = [4] (follows sequence)\n- "pick up knife" followed by "put down knife" = [1] (The environment state after execution does not match the target location)\n\n### Output Format:\nAnalysis: Brief explain how the step compares to previous step and advances goal progress\nGoal Progress Score: Use the following scale format: [1], [2], [3], [4], [5]\n\n"""})

        response = client.chat.completions.create(
                model="Qwen2-VL-7B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": messages
                    }
                ],
                max_tokens=1024,
                temperature=0,
                # n=1
            )
        
        content = response.choices[0].message.content
        score = extract_or_average_score(content)
        # print(process_messages(messages))

        return score, content

    def evaluate_score_gpt(self, goal, goal_image, prev_steps, prev_action_msg, imgs, step, action_ret, env_visible_objects, task_type=''):
        # client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
        client = OpenAI(api_key="sk-32krvxpzmYp3g4ar88D9490cEe524dA087F45bBa9745DbBa", base_url="https://api3.apifans.com/v1")

        base64_images = []
        for img in imgs:
            img_base64 = pil_to_base64(img)
            base64_images.append(img_base64)


        previous_step = ""

        for i, (prev_step, feedback, img) in enumerate(zip(prev_steps, prev_action_msg, base64_images[1:])):
                    
            previous_step = previous_step + f"Step {i + 1}: {prev_step}\nFeedback: {feedback}\n"

        
        if 'movable' in task_type:
            example = "\n## Example Complete Plan:\nGoal: Put a cardboard box with a phone in it on the seat of a chair.\nAll Steps: 1. find a cell phone, 2. pick up the cell phone, 3. find a box, 4. put down the cell phone, 5. pick up the box, 6. find an arm chair, 7. put down the box, 8. done.\nGoal: place a sponge in a glass bowl on an overhead drawer\nAll Steps: 1. find a cabinet, 2. open the cabinet, 3. find a dish sponge, 4. pick up the dish sponge, 5. close the cabinet, 6. find a bowl, 7. put down the dish sponge, 8. pick up the bowl, 9. find a cabinet, 10. open the cabinet, 11. put down the bowl, 12. close the cabinet, 13. done.\nGoal: Move a sponge and pan to the counter.\nAll Steps: 1. find a dish sponge, 2. pick up the dish sponge, 3. find a pan, 4. put down the dish sponge, 5. pick up the pan, 6. find a dining table, 7. put down the pan, 8. done.\nGoal: Place a pan with a piece of sliced lettuce plus a metal knife in it on a round black table.\nAll Steps: 1. find a knife, 2. pick up the knife, 3. find a lettuce, 4. slice the lettuce, 5. find a pan, 6. put down the knife, 7. find a lettuce, 8. pick up the lettuce, 9. find a pan, 10. put down the lettuce, 11. pick up the pan, 12. find a dining table, 13. put down the pan, 14. done."
        elif 'clean' in task_type:
            example =  "\n## Example Complete Plan:\nGoal: Put a clean bowl in a microwave.\nAll Steps: 1. find a bowl, 2. pick up the bowl, 3. find a sink, 4. put down the bowl, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a bowl, 9.pick up the bowl, 10. find a microwave, 11. open the microwave, 12. put down the bowl, 13. close the microwave, 14. done.\nGoal: wash a bowl from the counter then put it away\nAll Steps: 1. find a bowl, 2. pick up the bowl, 3. find a sink, 4. put down the bowl, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a bowl, 9. pick up the bowl, 10. find a shelf, 11. put down the bowl, 12. done.\nGoal: Place a rinsed knife on a counter.\nAll Steps: 1. find a butter knife, 2. pick up the butter knife, 3. find a sink, 4. put down the butter knife, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a butter knife, 9. pick up the butter knife, 10. find a counter top, 11. put down the butter knife, 12. done.\nGoal: pick up the knife, wash it off, place it on the table.\nAll Steps: 1. find a knife, 2. pick up the knife, 3. find a sink, 4. put down the knife, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a knife, 9. pick up the knife, 10. find a dining table, 11. put down the knife, 12. done."
        elif 'heat' in task_type:
            example =  "\n## Example Complete Plan:\nGoal: put cooked apple in the sink.\nAll Steps: 1. find an apple, 2. pick up the apple, 3. find a microwave, 4. open the microwave, 5. put down the apple, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find an apple, 11. pick up the apple, 12. close the microwave, 13. find a sink, 14. put down the apple, 15. done.\nGoal: Put a heated egg on a table.\nAll Steps: 1. find an egg, 2. pick up the egg, 3. find a microwave, 4. open the microwave, 5. put down the egg, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find an egg, 11. pick up the egg, 12. close the microwave, 13. find a dining table, 14. put down the egg, 15. done.\nGoal: Heat up coffee.\nAll Steps: 1. find a mug, 2. pick up the mug, 3. find a microwave, 4. open the microwave, 5. put down the mug, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find a mug, 11. pick up the mug, 12. close the microwave, 13. find a coffee machine, 14. put down the mug, 15. done.\nGoal: Put cooked apple slice on a table.\nAll Steps: 1. find a knife, 2. pick up the knife, 3. find an apple, 4. slice the apple, 5. find a microwave, 6. open the microwave, 7. put down the knife, 8. close the microwave, 9. find an apple, 10. pick up the apple, 11. find a microwave, 12. open the microwave, 13. put down the apple, 14. close the microwave, 15. turn on the microwave, 16. turn off the microwave, 17. open the microwave, 18. find an apple, 19. pick up the apple, 20. close the microwave, 21. find a side table, 22. put down the apple, 23. done.\nGoal: Cooking and cooling an egg.\nAll Steps: 1. find an egg, 2. pick up the egg, 3. find a microwave, 4. open the microwave, 5. put down the egg, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find an egg, 11. pick up the egg, 12. close the microwave, 13. find a fridge, 14. open the fridge, 15. put down the egg, 16. close the fridge, 17. done"
        elif 'cool' in task_type:
            example = "\n## Example Complete Plan:\nGoal: Put a chilled potato in the microwave.\nAll Steps: 1. find a microwave, 2. open the microwave, 3. find a potato, 4. pick up the potato, 5. close the microwave, 6. find a fridge, 7. open the fridge, 8. put down the potato, 9. close the fridge, 10. open the fridge, 11. find a potato, 12. pick up the potato, 13. close the fridge, 14. find a microwave, 15. open the microwave, 16. put down the potato, 17. close the microwave, 18. done.\nGoal: Put a cool potato in the sink.\nAll Steps: 1. find a potato, 2. pick up the potato, 3. find a fridge, 4. open the fridge, 5. put down the potato, 6. close the fridge, 7. open the fridge, 8. find a potato, 9. pick up the potato, 10. close the fridge, 11. find a sink, 12. put down the potato, 13. done.\nGoal: Put a chilled mug on the coffee machine.\nAll Steps: 1. find a mug, 2. pick up the mug, 3. find a fridge, 4. open the fridge, 5. put down the mug, 6. close the fridge, 7. open the fridge, 8. find a mug, 9. pick up the mug, 10. close the fridge, 11. find a coffee machine, 12. put down the mug, 13. done.\nGoal: Place a chilled lettuce slice on the table.\nAll Steps: 1. find a butter knife, 2. pick up the butter knife, 3. find a lettuce, 4. slice the lettuce, 5. find a fridge, 6. open the fridge, 7. put down the butter knife, 8. close the fridge, 9. find a lettuce, 10. pick up the lettuce, 11. find a fridge, 12. open the fridge, 13. put down the lettuce, 14. close the fridge, 15. open the fridge, 16. find a lettuce, 17. pick up the lettuce, 18. close the fridge, 19. find a dining table, 20. put down the lettuce, 21. done.\nGoal: Cooking and cooling an egg.\nAll Steps: 1. find an egg, 2. pick up the egg, 3. find a microwave, 4. open the microwave, 5. put down the egg, 6. close the microwave, 7. turn on the microwave, 8. turn off the microwave, 9. open the microwave, 10. find an egg, 11. pick up the egg, 12. close the microwave, 13. find a fridge, 14. open the fridge, 15. put down the egg, 16. close the fridge, 17. done"

        # example = "## Example Complete Plan:\nGoal: Put a clean bowl in a microwave.\nAll Steps: 1. find a bowl, 2. pick up the bowl, 3. find a sink, 4. put down the bowl, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a bowl, 9.pick up the bowl, 10. find a microwave, 11. open the microwave, 12. put down the bowl, 13. close the microwave, 14. done.\nGoal: wash a bowl from the counter then put it away\nAll Steps: 1. find a bowl, 2. pick up the bowl, 3. find a sink, 4. put down the bowl, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a bowl, 9. pick up the bowl, 10. find a shelf, 11. put down the bowl, 12. done.\nGoal: Place a rinsed knife on a counter.\nAll Steps: 1. find a butter knife, 2. pick up the butter knife, 3. find a sink, 4. put down the butter knife, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a butter knife, 9. pick up the butter knife, 10. find a counter top, 11. put down the butter knife, 12. done.\nGoal: pick up the knife, wash it off, place it on the table.\nAll Steps: 1. find a knife, 2. pick up the knife, 3. find a sink, 4. put down the knife, 5. find a faucet, 6. turn on the faucet, 7. turn off the faucet, 8. find a knife, 9. pick up the knife, 10. find a dining table, 11. put down the knife, 12. done."

        

        
        prompt_template = f'''Please serve as an unbiased evaluator for the AI-generated next step in the task planning according to the goal progress. The task involves robotic actions that typically follow a logical sequence of steps to achieve a defined goal.

{example}

## Input Data:
### Goal: {goal}

### Previous Steps:
{previous_step}
---

## AI-generated Next Step to Evaluate:
Step: {step}
Execution Result: {action_ret}
After executing the step, you can see the following objects/environment state: {env_visible_objects}

## Evaluation Criteria:

### Goal Progress (1-5 points):
Evaluate how effectively the step moves toward completing the task by considering:
1. **Action Sequence** - Does it follow a logical progression of actions based on the task requirements? (e.g., preparation → execution → refinement → goal completion)
2. **Previous Actions** - How does it build on prior steps? Does it avoid unnecessary repetition or conflicting actions?
3. **Goal State** - Does the step advance the task toward achieving the defined goal or final condition?
4. **Environment State** - Does the environment state after executing the step align with the expected progress toward the goal?

Scoring for Goal Progress:
- **[1]:** Step moves away from the goal or makes goal completion more difficult.
- **[2]:** Step is redundant or repeats the exact same action as the immediate previous step without progress.
- **[3]:** Step makes moderate progress toward the goal.
- **[4]:** Step makes significant progress toward the goal, aligning well with the task sequence.
- **[5]:** Step makes excellent progress, directly advancing toward goal completion.

### Examples:
- A step that repeats an action unnecessarily (e.g., "find object" followed by "find object") = [2].
- A step that logically follows the sequence (e.g., "find object" before "pick up object") = [4].
- A step that conflicts with the goal (e.g., "pick up object" followed by "put down object" without correct location) = [1].

---

## Output Format:
### Evaluation:
Analysis: Briefly explain how the step compares to prior actions, whether it follows a logical sequence, and how it advances the goal.
Goal Progress Score: Use the following scale format: [1], [2], [3], [4], [5].

### Hint:
Based on the evaluation, give some suggestion or hint of next step that aligns with the goal and current environment state.'''

        
        while True:
            try:
                response = client.chat.completions.create(
                        # model="Qwen2-VL-7B-Instruct",
                        model = "gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {
                                "role": "user",
                                "content": prompt_template
                            }
                        ],
                        max_tokens=1024,
                        temperature=0,
                        # n=1
                    )
                
                content = response.choices[0].message.content
                break
            except Exception as e:
                print(e)



        goal_score = extract_or_average_score(content)
        # print(prompt_template)

        executable_score = 5 if 'success' in action_ret else 0

        score = (goal_score + executable_score) /2

        # hint = content.split("### Hint:")[1]
        try:
            hint = content.split("### Hint:")[1]
        except IndexError:
            hint = None

        return score, content, hint
    
    

    def duplicate_past_key_values(self, past_key_values, batch_size):
        batch_past_key_values = []
        for layer in range(len(past_key_values)):
            batch_past_key_values_layer = []
            for kv in range(len(past_key_values[layer])):
                batch_past_key_values_layer.append(past_key_values[layer][kv].repeat(batch_size, 1, 1, 1))
            batch_past_key_values_layer = tuple(batch_past_key_values_layer)
            batch_past_key_values.append(batch_past_key_values_layer)
        batch_past_key_values = tuple(batch_past_key_values)
        return batch_past_key_values
