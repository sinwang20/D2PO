import time

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, 'src')
sys.path.insert(0, './alfred')

from src.alfred.thor_connector import ThorConnector
from src.alfred.utils import load_task_json, dotdict,ithor_name_to_natural_word
import os
from PIL import Image, ImageDraw, ImageFont
import json


# 1. 读取原始任务映射和完整的原始数据
def read_task_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 创建trail_id到完整task的映射
    task_mapping = {}
    # 创建task到原始数据项的映射
    # task_to_items = {}
    
    for item in data["valid_seen"]:
        task = item['task']
        trail_id = task.split('/')[-1]
        task_mapping[trail_id] = task
        
        # if task not in task_to_items:
        #     task_to_items[task] = []
        # task_to_items[task].append(item)

    print(f"Before: {len(data['train'])}")
    
    return task_mapping



def main(task_name):
    
    
    args_dict = {'data': 'alfred/data/json_2.1.0', 'pframe': 1000, 'fast_epoch': False,
                    'use_templated_goals': False, 'dout': 'exp/model', 'pp_folder': 'pp',
                    'reward_config': 'alfred/models/config/rewards.json', 'max_steps': 1000}
    model_args = dotdict(args_dict)

    task = {'task': task_name, 'repeat_idx': 1}
    traj_data = load_task_json(task)
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']

    env = ThorConnector(x_display='1')
    event = env.reset('FloorPlan%d' % scene_num)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    env.step(dict(traj_data['scene']['init_action']))
    env.set_task(traj_data, model_args, reward_type='dense')

    env_objects = list(set(ithor_name_to_natural_word(item['name'].split('_')[0]) for item in event.metadata["objects"]))
    print(env_objects)

    env.step(dict(action='ToggleMapView'))

    map_view = event.frame


    instructions = [
        # 'find kettle', 'pick up kettle', 'find kettle', 'find sink', 'put down kettle', 'turn on faucet', 'turn off faucet', 'pick up kettle', 'find cabinet','open cabinet',  'put down kettle', 'close cabinet'
# "find a dinner table",
"find a microwave",
            "open the microwave",
            "find an egg",
            "pick up the egg",
            "close the microwave",
            "find a sink",
            "put down the egg",
            "find a faucet",
            "turn on the faucet",
            "turn off the faucet",
            "find an egg",
            "pick up the egg",
            "find a fridge",
            "open the fridge",
            "put down the egg",
            "close the fridge"
            # "pick up the key chain",
            # "find a floor lamp",
            # "turn on the floor lamp"
            # "close the cabinet",
            # "find a bathtub",
            # "put down the cloth"
        # 'find cloth',
        # 'pick up cloth',
        # 'find a sink',
        # 'put down cloth',
        # # 'find a faucet',
        # 'turn on the faucet',
        # 'turn off the faucet',
        # 'find cloth',
        # 'pick up cloth',
        # 'find bathtub',
        # 'put down cloth',

        # 'find a soap bar',
        # 'find a sink',
        # 'find a sink',
        # 'open the fridge',
        # 'put down the plate',
        # 'put down the plate',
        # 'put down the plate',
    ]
    image_save_dir = os.path.join("output/test/clean-25", 'images-test')
    os.makedirs(image_save_dir, exist_ok=True)
    image_state = Image.fromarray(env.last_event.frame)
    image_state.save(os.path.join(image_save_dir, f'frame_map_{scene_num}.png'))

    

    # exit()
    return

    instruction_text = traj_data['turk_annotations']['anns'][0]['task_desc']
    print(f"Task: {instruction_text}")

    imgs = []
    for i, instruction in enumerate(instructions):
        ret_dict = env.llm_skill_interact(instruction)
        if not ret_dict['success']:
            print(ret_dict['message'])
        print(ret_dict)
        # imgs.append(env.write_step_on_img(False, i, instruction))
        image_state = Image.fromarray(env.last_event.frame)
        image_state.save(os.path.join(image_save_dir, f'frame_{i+1}.png')) 

    # goal_satisfied = env.get_goal_satisfied()
    # print(goal_satisfied)
    # print('Target goal: ' + json.dumps(env.task.get_targets()))
    # print('Success: ' + str(goal_satisfied))


    

        

    # save_result(None, imgs, 'results_test')


def minimal_test():
    import ai2thor.controller

    controller = ai2thor.controller.Controller()
    controller.start()

    # controller.reset('FloorPlan28')
    # controller.step(dict(action='Initialize', gridSize=0.25))

    for i in range(10):
        controller.step(dict(action='RotateRight'))
        print('rotate')
        time.sleep(0.1)

def all_test():
    task_mapping = read_task_data('/data-mnt/data/sywang/LLMTaskPlanning/alfred/data/splits/oct21.json')
    args_dict = {'data': 'alfred/data/json_2.1.0', 'pframe': 300, 'fast_epoch': False,
                    'use_templated_goals': False, 'dout': 'exp/model', 'pp_folder': 'pp',
                    'reward_config': 'alfred/models/config/rewards.json', 'max_steps': 1000}
    model_args = dotdict(args_dict)

    # 读取所有任务的json文件
    with open('./resource/alfred_examples_for_prompt-valid_seen.json', 'r') as f:
        all_tasks = json.load(f)

    # 创建结果输出文件
    output_file = "results_valid_seen.jsonl"
    success_count = 0
    total_count = 0

    env = ThorConnector(x_display='1')

    for task_data in all_tasks[158:]:
        total_count += 1
        result = {}
        
        # 构造任务ID
        task_com = task_mapping[task_data['task id']]
        task = {'task': task_com, 'repeat_idx': 1}
        
        try:
            traj_data = load_task_json(task)
            scene_num = traj_data['scene']['scene_num']
            object_poses = traj_data['scene']['object_poses']
            dirty_and_empty = traj_data['scene']['dirty_and_empty']
            object_toggles = traj_data['scene']['object_toggles']

            event = env.reset('FloorPlan%d' % scene_num)
            env.restore_scene(object_poses, object_toggles, dirty_and_empty)

            env.step(dict(traj_data['scene']['init_action']))
            env.set_task(traj_data, model_args, reward_type='dense')

            env_objects = list(set(ithor_name_to_natural_word(item['name'].split('_')[0]) for item in event.metadata["objects"]))
    # print(env_objects)

            # 使用任务中的NL steps
            instructions = task_data['NL steps']
            
            # 创建该任务的图片保存目录
            image_save_dir = os.path.join("output/test/valid-seen", task_data['task id'], 'images')
            os.makedirs(image_save_dir, exist_ok=True)

            step_results = []
            for i, instruction in enumerate(instructions):
                ret_dict = env.llm_skill_interact(instruction)
                step_results.append({
                    'instruction': instruction,
                    'success': ret_dict['success'],
                    'message': ret_dict.get('message', '')
                })
                
                # 保存每步的图片
                image_state = Image.fromarray(env.last_event.frame)
                image_state.save(os.path.join(image_save_dir, f'frame_{i}.png'))

            goal_satisfied = env.get_goal_satisfied()
            if goal_satisfied:
                success_count += 1

            # 记录该任务的结果
            result = {
                'task_id': task_data['task id'],
                'task_type': task_data['task type'],
                'task_description': task_data['task description'],
                'env_objects': env_objects,
                'step_results': step_results,
                'goal_satisfied': goal_satisfied,
                'targets': env.task.get_targets()
            }

        except Exception as e:
            result = {
                'task_id': task_data['task id'],
                'error': str(e),
                'goal_satisfied': False
            }

        # 将结果写入jsonl文件
        with open(output_file, 'a') as f:
            f.write(json.dumps(result) + '\n')

    # 输出总体成功率
    success_rate = success_count / total_count if total_count > 0 else 0
    print(f"Overall Success Rate: {success_rate:.2%} ({success_count}/{total_count})")

    env.stop()


if __name__ == '__main__':
    # minimal_test()
    task_list = ["pick_heat_then_place_in_recep-Tomato-None-CounterTop-20/trial_T20190909_041126_508914", "look_at_obj_in_light-CD-None-DeskLamp-314/trial_T20190907_114323_767231"]
    for task_name in task_list:
        main(task_name)
    # all_test()