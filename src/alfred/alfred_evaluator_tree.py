import math
import os, json
import pprint
import random
import textwrap
import time, datetime
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '')
sys.path.insert(0, './alfred')

from PIL import Image, ImageDraw, ImageFont

from alfred.data.preprocess import Dataset
from src.alfred.alfred_task_planner import AlfredTaskPlanner
from src.alfred.thor_connector import ThorConnector
from src.alfred.utils import dotdict, load_task_json, ithor_name_to_natural_word
from tqdm import tqdm

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import heapq
import copy
from io import BytesIO

from src.evaluator import Evaluator
from collections import OrderedDict


splits = 'alfred/data/splits/dec14.json'
font = ImageFont.truetype("UbuntuMono-B.ttf", 24)
log = logging.getLogger(__name__)




import hashlib
# from PIL import Image
# import os

# 全局缓存字典，存储图像哈希值与路径的映射
image_cache = {}

def hash_image(img):
    """
    计算 PIL Image 对象的哈希值（MD5）
    """
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return hashlib.md5(buffer.getvalue()).hexdigest()

def save_image_optimized(img, task, step_idx, level_idx, candidate_idx, image_dir="saved_images"):
    """
    优化存储图像，避免重复保存
    """
    global image_cache
    # os.makedirs(image_dir, exist_ok=True)

    # 计算图像哈希值
    img_hash = hash_image(img)

    # 如果图像已被保存过，直接返回路径
    if img_hash in image_cache:
        return image_cache[img_hash]

    # 否则保存图像并记录路径
    image_path = os.path.join(image_dir, f"task_{task}_step_{step_idx}_img_{level_idx}-{candidate_idx}.png")
    img.save(image_path)
    image_cache[img_hash] = image_path

    return image_path




def store_candidate_to_jsonl(file_path, task, goal, step_idx, level_idx, candidate_idx, prev_steps, prev_action_msg, imgs, step, action_ret, score, image_dir, env_visible_objects, hint, reasoning):
    """
    存储候选数据为 JSONL 格式，优化图像存储
    """
    global image_cache
    # import pdb; pdb.set_trace()
    # 优化图像存储，避免重复保存
    image_paths = []
    for idx, img in enumerate(imgs):
        # if isinstance(img, Image.Image):  # 检查是否为 PIL Image 对象
        image_path = save_image_optimized(img, task, step_idx, level_idx, candidate_idx, image_dir)
        image_paths.append(image_path)
        # else:
        #     image_paths.append(img)  # 如果不是 Image 对象，直接保存原数据

    data = {
        "task": task,
        "goal": goal,
        "step_idx": step_idx,
        "prev_steps": prev_steps,
        "prev_action_msg": prev_action_msg,
        "step": step,
        "hint": hint,
        "reasoning": reasoning,
        "action_ret": action_ret,
        "score": score,
        "env_visible_objects": env_visible_objects,
        "imgs": image_paths
    }
    with open(file_path, 'a') as f:
        f.write(json.dumps(data) + "\n")





class AlfredEvaluator(Evaluator):
    def __init__(self, hparams):
        self.cfg = hparams

    def evaluate(self):
        cfg = self.cfg

        log.info(OmegaConf.to_yaml(cfg))
        global splits

        # llm planner
        if len(cfg.planner.model_name) > 0:
            planner = AlfredTaskPlanner(cfg)
            planner.reset()  # init skill set
        else:
            planner = None

        # prepare
        args_dict = {'data': 'alfred/data/json_2.1.0', 'pframe': 300, 'fast_epoch': False,
                    'use_templated_goals': False, 'dout': 'exp/model', 'pp_folder': 'pp',
                    'reward_config': 'alfred/models/config/rewards.json', 'max_steps': 1000}

        with open(splits) as f:
            splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in splits.items()})

        # preprocessing
        number_of_dirs = len(list(os.listdir(args_dict['data'])))
        do_preprocessing = number_of_dirs < 50  # one-time process
        if do_preprocessing:
            log.info("\nPreprocessing dataset... Do this once as required:")
            vocab = None  # todo
            dataset = Dataset(dotdict(args_dict), vocab)
            dataset.preprocess_splits(splits)

        
        assert cfg.alfred.eval_set in splits.keys()
        files = []

        # exclude two obj task
        for e in splits[cfg.alfred.eval_set]:
            # if 'pick_two_obj_and_place' not in e['task'] and 'clean' not in e['task'] and 'cool' not in e['task'] and 'heat' not in e['task'] and 'pick_and_place_with_movable' not in e['task']:
            if cfg.alfred.eval_task in e['task']:
                files.append(e)

        print(f"num of tasks: {len(files)}")
       
        start = cfg.alfred.eval_start_index
        end = cfg.alfred.eval_end_index

        if 0 <= start < len(files) and 0 < end <= len(files) and start < end:
            # 选择文件的子集
            files = files[start:end]

        # 如果需要确保随机性重现，则可以保留 random.seed(cfg.planner.random_seed)
        # seed = cfg.alfred.random_seed_for_eval_subset
        # random.seed(seed)
        random.seed(cfg.planner.random_seed)
        
        # print("START EVAL")
        if False:  # debug mode
            for file in files:
                if 'trial_T20190907_012527_434405' in file['task']:
                    new_files = [file]
                    break
            files = new_files

        # print("START EVAL RUN")
        # run
        start = time.time()
        x_display = cfg.alfred.x_display
        save_path = cfg.out_dir
        os.makedirs(save_path, exist_ok=True)
        results = self.evaluate_main(files, args_dict, planner, x_display, save_path)

        # print results
        log.info(results)
        n = len(results)
        n_success = 0
        for e in results:
            if e['success']:
                n_success += 1
        log.info(f'success rate: {n_success / n * 100:.2f} % ({n_success}/{n})')
        log.info(f'elapsed: {str(datetime.timedelta(seconds=(time.time() - start)))}')
        log.info('------------------------')
        log.info(OmegaConf.to_yaml(cfg))  # print cfg once again for easier configuration lookup

    def evaluate_main(self, tasks, args_dict, planner, x_display, save_path):
        results = []
        model_args = dotdict(args_dict)
        # print("[START ENV]")
        env = ThorConnector(x_display=x_display)
        # print("[FINISH RUN]")

        # save prompt
        if planner is not None:
            with open(os.path.join(save_path, 'prompt.txt'), 'w') as f:
                f.write(planner.prompt)

        train_gt_steps = None
        if self.cfg.alfred.eval_set == 'train':
            # load ground-truth trajectories
            train_gt_steps = {}
            with open(self.cfg.prompt.example_file_path, 'r') as f:
                train_examples = json.load(f)
            for ex in train_examples:
                train_gt_steps[ex['task id']] = ex['NL steps']

        # print("[START RUN]")
        # run
        for i, task in enumerate(tqdm(tasks)):
            try:
                log.info(task)
                traj_data = load_task_json(task)
                r_idx = task['repeat_idx']
                log.info(f"Evaluating ({i+1}/{len(tasks)}): {traj_data['root']}")
                result = self.evaluate_task(env, traj_data, r_idx, model_args, planner, save_path, log_prompt=(i==0), train_gt_steps=train_gt_steps)
                results.append(result)

            except Exception as e:
                import traceback
                traceback.print_exc()
                log.info("Error: " + repr(e))

        return results

    

    
    def evaluate_task(self, env, traj_data, r_idx, model_args, planner, save_path, log_prompt=False, train_gt_steps=None):
        # setup scene
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']
        task_type = traj_data['task_type']

        scene_name = 'FloorPlan%d' % scene_num

        # print goal instr
        instruction_text = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        # instruction_text = "Place a knife on the side table."
        log.info("Task: %s" % instruction_text)
        if len(instruction_text.split(" ")) > 14:
            return



        event = env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize
        env.step(dict(traj_data['scene']['init_action']))
        env.set_task(traj_data, model_args, reward_type='dense')

        

        alfred_objs = ['Cart', 'Potato', 'Faucet', 'Ottoman', 'CoffeeMachine', 'Candle', 'CD', 'Pan', 'Watch',
                   'HandTowel', 'SprayBottle', 'BaseballBat', 'CellPhone', 'Kettle', 'Mug', 'StoveBurner', 'Bowl',
                   'Toilet', 'DiningTable', 'Spoon', 'TissueBox', 'Shelf', 'Apple', 'TennisRacket', 'SoapBar',
                   'Cloth', 'Plunger', 'FloorLamp', 'ToiletPaperHanger', 'CoffeeTable', 'Spatula', 'Plate', 'Bed',
                   'Glassbottle', 'Knife', 'Tomato', 'ButterKnife', 'Dresser', 'Microwave', 'CounterTop',
                   'GarbageCan', 'WateringCan', 'Vase', 'ArmChair', 'Safe', 'KeyChain', 'Pot', 'Pen', 'Cabinet',
                   'Desk', 'Newspaper', 'Drawer', 'Sofa', 'Bread', 'Book', 'Lettuce', 'CreditCard', 'AlarmClock',
                   'ToiletPaper', 'SideTable', 'Fork', 'Box', 'Egg', 'DeskLamp', 'Ladle', 'WineBottle', 'Pencil',
                   'Laptop', 'RemoteControl', 'BasketBall', 'DishSponge', 'Cup', 'SaltShaker', 'PepperShaker',
                   'Pillow', 'Bathtub', 'SoapBottle', 'Statue', 'Fridge', 'Sink',

                   'KeyChain', 'Potato', 'Pot', 'Pen', 'Candle', 'CD', 'Pan', 'Watch', 'Newspaper', 'HandTowel',
                       'SprayBottle', 'BaseballBat', 'Bread', 'CellPhone', 'Book', 'Lettuce', 'CreditCard', 'Mug',
                       'AlarmClock', 'Kettle', 'ToiletPaper', 'Bowl', 'Fork', 'Box', 'Egg', 'Spoon', 'TissueBox',
                       'Apple', 'TennisRacket', 'Ladle', 'WineBottle', 'Cloth', 'Plunger', 'SoapBar', 'Pencil',
                       'Laptop', 'RemoteControl', 'BasketBall', 'DishSponge', 'Cup', 'Spatula', 'SaltShaker',
                       'Plate', 'PepperShaker', 'Pillow', 'Glassbottle', 'SoapBottle', 'Knife', 'Statue', 'Tomato',
                       'ButterKnife', 'WateringCan', 'Vase',

                    'Safe', 'Laptop', 'Fridge', 'Box', 'Microwave', 'Cabinet', 'Drawer',

                    'Potato', 'Lettuce', 'Tomato', 'Apple', 'Bread',

                    'Microwave', 'DeskLamp', 'FloorLamp', 'Faucet',

                    'ArmChair', 'Safe', 'Cart', 'Ottoman', 'Pot', 'CoffeeMachine', 'Desk', 'Cabinet', 'Pan',
                    'Drawer', 'Sofa', 'Mug', 'StoveBurner', 'SideTable', 'Toilet', 'Bowl', 'Box', 'DiningTable',
                    'Shelf', 'ToiletPaperHanger', 'CoffeeTable', 'Cup', 'Plate', 'Bathtub', 'Bed', 'Dresser',
                    'Fridge', 'Microwave', 'CounterTop', 'Sink', 'GarbageCan'
                       ]

        aithor_objects = [ithor_name_to_natural_word(obj) for obj in alfred_objs]
        # print([item['name'].split('_')[0] for item in event.metadata["objects"]])
        env_objects = list(set(ithor_name_to_natural_word(item['name'].split('_')[0]) for item in event.metadata["objects"]))
        # print(env_objects)
        env_objects = list(set(env_objects) & set(aithor_objects))
        print(env_objects)
        
        goal_image = None
        


        image_save_dir = os.path.join(save_path, 'images-test')
        os.makedirs(image_save_dir, exist_ok=True)

        file_save_dir = os.path.join(save_path, 'data.jsonl')
        
        
        original_max_steps = 25
        # inference steps from instruction
        beam_width = 2
        max_candidates = 5
        max_steps = original_max_steps


        # 初始化
        prev_steps = []  # 存储之前的步骤
        prev_action_msg = []  # 存储之前的动作执行信息
        imgs = []  # 存储每个步骤的图像
        t = 0

        # Initialize beam queue with the initial environment
        queue = []
        step_num = 0

        # 队列中每个元素为 (score, path) 其中 path 是一个包含步骤、图像等的元组
        heapq.heappush(queue, (0, 0, (0, prev_steps, prev_action_msg, imgs, t, None, 0)))  # 初始状态（包括 env）



        current_image = Image.fromarray(env.last_event.frame)
        current_image.save(os.path.join(image_save_dir, f'frame_{t}.png'))  # 保存当前图像
        imgs.append(current_image)
        done = False
        done_paths=[]

        while queue and step_num < max_steps:
            # print(f"STEP NUM:{step_num}")
            step_num += 1
            

            # 对每条路径进行扩展
            all_candidates = []
            for level_idx, (_,_2,path) in enumerate(queue):
                prev_score, prev_steps, prev_action_msg, imgs, t, hint, cur_score = path  # 提取环境对象


                # 从planner生成5个候选步骤
                step_candidates, step_contents = planner.generate_step_candidates(
                    instruction_text, prev_steps, prev_action_msg, prev_images=imgs, visible_objs_str=env_objects, n=5, task_type=task_type
                )
                print(step_candidates)

                # 如果有提示，则利用提示生成候选
                if hint:
                    step_candidates_hint, step_contents_hint = planner.generate_step_candidates_hint(
                        instruction_text, prev_steps, prev_action_msg, prev_images=imgs, visible_objs_str=env_objects, n=2, hint=hint, task_type=task_type
                    )
                    step_candidates.extend(step_candidates_hint)
                    step_contents.extend(step_contents_hint)
                print(step_candidates)
               

                unique_step_dict = OrderedDict()
                for step, content in zip(step_candidates, step_contents):
                    if step!= "" and step not in unique_step_dict:  # 如果 step 还没有被添加到字典中
                        unique_step_dict[step] = content

                # 提取去重后的 steps 和对应的 contents
                unique_step_candidates = list(unique_step_dict.keys())
                unique_step_contents = list(unique_step_dict.values())
                
                log.info(f"[STEP {t+1}] {prev_steps} --> {unique_step_candidates}")

             
                for candidate_idx, (step, content) in enumerate(zip(unique_step_candidates, unique_step_contents)):

                    if step is None:
                        log.info("\tmax step reached")
                        break

                    # 当前step重复了上一个step
                    if t >0 and step == prev_steps[-1]:
                        cur_feedback_msg = prev_action_msg[-1]
                        score = [3, "success", None]
                        action_ret = {""}

                        status, message = cur_feedback_msg.split(": ", 1)
                        success = (status == "success")
                        
                        # Construct the action_ret dictionary
                        action_ret = {"action": step, "success": success, "message": message if message else ""}

                        store_candidate_to_jsonl(
                            file_path=file_save_dir,
                            task=os.path.basename(traj_data['root']),
                            goal=instruction_text,
                            step_idx=t,
                            level_idx=level_idx,
                            candidate_idx=candidate_idx,
                            prev_steps=prev_steps,
                            prev_action_msg=prev_action_msg,
                            imgs=imgs + [imgs[-1]],
                            step=step,
                            action_ret=action_ret,
                            score=score,
                            image_dir=image_save_dir,
                            env_visible_objects="",
                            hint=hint,
                            reasoning=content
                        )

                        hint = score[2]
                        all_candidates.append((score[0]+prev_score/2, step, action_ret, prev_steps + [step], prev_action_msg + [cur_feedback_msg], imgs + [image_state], t + 1, level_idx, candidate_idx, hint, score[0]))




                    else:
                        event = env.reset(scene_name)
                        env.restore_scene(object_poses, object_toggles, dirty_and_empty)
                        env.step(dict(traj_data['scene']['init_action']))
                        for prev_step in prev_steps:
                            try:
                                action_ret = env.llm_skill_interact(prev_step)
                            except Exception as e:
                                # log.warning(f"Error executing step: {e}")
                                action_ret = {'action': prev_step, 'success': False, 'message': str(e)}

                        # 执行候选步骤
                        try:
                            action_ret = env.llm_skill_interact(step)
                        except Exception as e:
                            log.warning(f"Error executing step: {e}")
                            action_ret = {'action': step, 'success': False, 'message': str(e)}

                        env_visible_objects = list(set(ithor_name_to_natural_word(item['name'].split('_')[0]) for item in env.last_event.metadata["objects"] if item.get('visible', False)))
                        env_visible_objects = list(set(env_visible_objects) & set(aithor_objects))
                        goal_satisfied = env.get_goal_satisfied()
                        image_state = Image.fromarray(env.last_event.frame)
                        cur_feedback_msg = f"{'success' if action_ret['success'] else 'failed'}: {action_ret['message']}"


                        # 如果当前节点抵达目标条件
                        if goal_satisfied:
                            # Store the successful step
                            score = [5, "success", "finish the goal"]
                            store_candidate_to_jsonl(
                                file_path=file_save_dir,
                                task=os.path.basename(traj_data['root']),
                                goal=instruction_text,
                                step_idx=t,
                                level_idx=level_idx,
                                candidate_idx=candidate_idx,
                                prev_steps=prev_steps,
                                prev_action_msg=prev_action_msg,
                                imgs=imgs + [image_state],
                                step=step,
                                action_ret=action_ret,
                                score=score,
                                image_dir=image_save_dir,
                                env_visible_objects=env_visible_objects,
                                hint=hint,
                                reasoning=content
                            )


                            # Add 'done' step after the successful step
                            done_score = 100
                            done_step = 'done'
                            done_action_ret = {'action': done_step, 'success': True, 'message': "finish the goal"}
                            done_feedback_msg = str(True)
                            done_hint = "The current state has met the goal and the planning can be ended"

                            
                            # Store the done step
                            store_candidate_to_jsonl(
                                file_path=file_save_dir,
                                task=os.path.basename(traj_data['root']),
                                goal=instruction_text,
                                step_idx=t+1,
                                level_idx=level_idx,
                                candidate_idx=candidate_idx,
                                prev_steps=prev_steps + [step],
                                prev_action_msg=prev_action_msg + [cur_feedback_msg],
                                imgs=imgs + [image_state, image_state],
                                step=done_step,
                                action_ret=done_action_ret,
                                score=[done_score, "", None],
                                image_dir=image_save_dir,
                                env_visible_objects=env_visible_objects,
                                hint=done_hint,
                                reasoning="Goal satisfied, terminating trajectory"
                            )

                            step_candidates_done, step_contents_done = planner.generate_step_candidates(
                                instruction_text, prev_steps + [step], prev_action_msg+ [cur_feedback_msg], prev_images=imgs+[image_state], visible_objs_str=env_objects, n=3
                            )

                            unique_step_dict_done = OrderedDict()
                            for step_done, content_done in zip(step_candidates_done, step_contents_done):
                                if step_done!= "" and step_done not in unique_step_dict_done:  # 如果 step 还没有被添加到字典中
                                    unique_step_dict_done[step_done] = content_done

                            # 提取去重后的 steps 和对应的 contents
                            unique_step_candidates_done = list(unique_step_dict_done.keys())
                            unique_step_contents_done = list(unique_step_dict_done.values())

                            for done_idx, (step_done, content_done) in enumerate(zip(unique_step_candidates_done, unique_step_contents_done)):
                                if 'done' not in step_done:
                                    store_candidate_to_jsonl(file_path=file_save_dir, task=os.path.basename(traj_data['root']), goal=instruction_text, step_idx=t+1, level_idx=-1, candidate_idx=done_idx, prev_steps=prev_steps + [step], prev_action_msg=prev_action_msg + [cur_feedback_msg], imgs=imgs + [image_state, image_state], step=step_done, action_ret= "", score=[0, "", None], image_dir=image_save_dir, env_visible_objects=env_visible_objects, hint=done_hint, reasoning="Goal satisfied, terminating trajectory"
                                    )


                            # Add done step to candidates
                            all_candidates.append((done_score + prev_score/2, done_step, done_action_ret, 
                                                prev_steps + [step, done_step],
                                                prev_action_msg + [cur_feedback_msg, done_feedback_msg],
                                                imgs + [image_state, image_state], t + 1, -1, candidate_idx, "The goal is finished.", 5))

                            done = True


                            # Record and save results after all paths have been processed
                            log_entry = {
                                'trial': traj_data['task_id'], 'scene': scene_name, 'type': traj_data['task_type'], 'repeat_idx': int(r_idx), 'goal_instr': instruction_text, 'candidate_idx': _2, 'inferred_steps': prev_steps+ [step, done_step], 'success': goal_satisfied
                            }

                            # Save the result
                            self.save_result(log_entry, imgs + [image_state, image_state.copy()], save_path)



                        # 如果没有达到目标条件
                        else:
                                                        
                            # 评估候选步骤的得分
                            score = planner.evaluate_score_gpt(instruction_text, goal_image, prev_steps, prev_action_msg, imgs + [image_state], step, cur_feedback_msg, env_visible_objects, task_type=task_type)

                            store_candidate_to_jsonl(
                                file_path=file_save_dir,
                                task=os.path.basename(traj_data['root']),
                                goal=instruction_text,
                                step_idx=t,
                                level_idx=level_idx,
                                candidate_idx=candidate_idx,
                                prev_steps=prev_steps,
                                prev_action_msg=prev_action_msg,
                                imgs=imgs + [image_state],
                                step=step,
                                action_ret=action_ret,
                                score=score,
                                image_dir=image_save_dir,
                                env_visible_objects=env_visible_objects,
                                hint=hint,
                                reasoning=content
                            )

                            
                            all_candidates.append((score[0]+prev_score/2, step, action_ret, prev_steps + [step], prev_action_msg + [cur_feedback_msg], imgs + [image_state], t + 1, level_idx, candidate_idx, score[2], score[0]))
                        
                        log.info(f"{step}: {score}")

        
            # 选择得分最高的候选步骤，维持最大宽度为2的队列
            # 在添加新候选之前，清空队列
            queue.clear()
            for score, step, action_ret, new_prev_steps, new_prev_action_msg, new_imgs, new_t, level_idx, candidate_idx, hint, cur_score in all_candidates:
                # 把得分最高的候选加入队列，维持最大宽度为 2
                heapq.heappush(queue, (score, -(new_t*100+level_idx*10+candidate_idx), (score, new_prev_steps, new_prev_action_msg, new_imgs, new_t, hint, cur_score)))

            log.info(f"all queue: {queue}")

            # 获取top2
            top2_items = [item for item in heapq.nlargest(2, queue) if item[2][6] >= 3.2]
            # 获取所有分数大于等于4的项
            high_score_items = [item for item in queue if item[2][6] >= 4.5]

            # 用一个集合来记录已经添加的项的标识（使用score和candidate_idx组合作为标识）
            seen = set()
            final_items = []

            # 优先添加top2
            for item in top2_items:
                identifier = (item[0], item[1])
                if identifier not in seen and 'done' not in item[2][1]:
                    seen.add(identifier)
                    final_items.append(item)

            # 添加高分项
            for item in high_score_items:
                identifier = (item[0], item[1])
                if identifier not in seen and 'done' not in item[2][1]:
                    seen.add(identifier)
                    final_items.append(item)

            queue = final_items
            heapq.heapify(queue)
            log.info(f"filtered queue: {queue}")

            if done:
                log.info("[One Path SUCCESS already!]")
                # break
                if max_steps == original_max_steps:  # 只在未修改过max_steps时执行
                    max_steps = step_num + 3
                    # max_steps = step_num + 3




        # After finishing the beam search, evaluate the final set of paths
        # print(queue)
        for _,_2, path in queue:
            score, prev_steps, prev_action_msg, imgs, t, hint, cur_score = path
            
            event = env.reset(scene_name)
            env.restore_scene(object_poses, object_toggles, dirty_and_empty)
            env.step(dict(traj_data['scene']['init_action']))
            for step in prev_steps:
                try:
                    action_ret = env.llm_skill_interact(step)
                except Exception as e:
                    # log.warning(f"Error executing step: {e}")
                    action_ret = {'action': step, 'success': False, 'message': str(e)}


            # Check if the goal was satisfied for the final path
            goal_satisfied = env.get_goal_satisfied()
            log.info('Target goal: ' + json.dumps(env.task.get_targets()))
            log.info('Success: ' + str(goal_satisfied))

            success = False
            # If the goal is satisfied, mark the task as successful
            if goal_satisfied:
                success = True

            # Record and save results after all paths have been processed
            log_entry = {
                'trial': traj_data['task_id'], 'scene': scene_name, 'type': traj_data['task_type'], 'repeat_idx': int(r_idx), 'goal_instr': instruction_text, 'candidate_idx': _2, 'inferred_steps': prev_steps, 'success': success
            }

            # Save the result
            self.save_result(log_entry, imgs, save_path)

        return log_entry

    def save_result(self, result_dict, imgs, base_path='results'):
        imgs_copy = [img.copy() for img in imgs]
        if result_dict:
            filename = f"{result_dict['trial']}_{result_dict['repeat_idx']}_{result_dict['candidate_idx']}"

            # write json
            with open(os.path.join(base_path, filename + '.json'), "w") as outfile:
                json.dump(result_dict, outfile)
        else:
            filename = "images"

        # write img
        widths, heights = zip(*(i.size for i in imgs_copy))
        total_width = widths[0] * 5
        textbox_height = 70  # max two lines
        total_height = math.ceil(len(imgs_copy) / 5) * heights[0] + textbox_height
        new_im = Image.new('RGB', (total_width, total_height), color='white')

        # draw text
        if result_dict:
            text = 'Instruction: ' + result_dict['goal_instr']
            text_color = (0, 0, 0)  # black
            # text_color = (100, 255, 100) if result_dict['success'] else (255, 100, 100)
            lines = textwrap.wrap(text, width=110)
            draw = ImageDraw.Draw(new_im)
            y_start = 10 if len(lines) > 1 else 35
            draw.multiline_text((10, y_start), '\n'.join(lines), font=font, fill=text_color)
            y_offset = textbox_height
        else:
            y_offset = 0

        x_offset = 0
        for idx, im in enumerate(imgs_copy):

            if idx > 0:
                step_text = f"Step {idx}: " + result_dict["inferred_steps"][idx-1]
                draw = ImageDraw.Draw(im)
                lines = textwrap.wrap(step_text, width=20)
                y_text = 6
                for line in lines:
                    width, height = font.getsize(line)
                    draw.text((6, y_text), line, font=font, fill=(255, 255, 255))
                    y_text += height

            new_im.paste(im, (x_offset, y_offset))
            x_offset += im.size[0]
            if x_offset >= total_width:
                x_offset = 0
                y_offset += im.size[1]

        success_str = 'success' if result_dict['success'] else 'fail'
        new_im.save(os.path.join(base_path, f"{filename}_{success_str}.png"))
