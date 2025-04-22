import os, json, re
import string
import subprocess


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_task_json(task):
    '''
    load preprocessed json from disk
    '''
    json_path = os.path.join('alfred/data/json_2.1.0', task['task'], 'pp',
                             'ann_%d.json' % task['repeat_idx'])
    with open(json_path) as f:
        data = json.load(f)
    return data


def print_gpu_usage(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """

    def query(field):
        return (subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
             '--format=csv,nounits,noheader'],
            encoding='utf-8'))

    def to_int(result):
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used / total
    print('\n' + msg, f'{100 * pct:2.1f}% ({used} out of {total})')


def ithor_name_to_natural_word(w):
    # e.g., RemoteController -> remote controller
    if w == 'CD':
        return w
    else:
        return re.sub(r"(\w)([A-Z])", r"\1 \2", w).lower()


def natural_word_to_ithor_name(w):
    # e.g., floor lamp -> FloorLamp
    if w == 'CD':
        return w
    else:
        return ''.join([string.capwords(x) for x in w.split()])


def find_indefinite_article(w):
    # simple rule, not always correct
    w = w.lower()
    if w[0] in ['a', 'e', 'i', 'o', 'u']:
        return 'an'
    else:
        return 'a'


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