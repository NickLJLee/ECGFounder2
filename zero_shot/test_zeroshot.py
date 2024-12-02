import os
import random
import yaml as yaml
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch

import sys
sys.path.append("../utils")
import utils_builder
from zeroshot_val import zeroshot_eval
from model import CLIP

# os.environ["TOKENIZERS_PARALLELISM"] = "true"

device_id = 'cuda:5'

config = yaml.load(open("zeroshot_config.yaml", "r"), Loader=yaml.FullLoader)

# torch.manual_seed(42)
# random.seed(0)
# np.random.seed(0)
def load_clip(context_length=77): 
    """
    FUNCTION: load_clip
    ---------------------------------
    """
    device = torch.device("cuda:5")
    # use new model params
    params = {
    'context_length': context_length,
    'vocab_size': 49408,
    'transformer_width': 512,
    'transformer_heads': 8,
    'transformer_layers': 12
    }

    model = CLIP(**params)
    try: 
        # use new model params
        params = {
        'context_length': context_length,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12
        }

        model = CLIP(**params)          
        # checkpoint = torch.load('/data/0shared/lijun/code/ECGFounder_CLIP/checkpoints/1st_round/checkpoint_29800.pt', map_location=lambda storage, loc: storage.cuda(0))
        checkpoint = torch.load('/data/0shared/lijun/code/ECGFounder_CLIP/checkpoints/delete_temporal_label/checkpoint_18000.pt', map_location=lambda storage, loc: storage.cuda(5))
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
    except: 
        print("Argument error. Set pretrained = True.", sys.exc_info()[0])
        raise
    return model
# model = utils_builder.ECGCLIP(config['network'])
# ckpt = 'your_ckpt_path'
model = load_clip(context_length=77)
# ckpt = torch.load(f'{ckpt}', map_location='cpu')
# model.load_state_dict(ckpt)
# model = model.to(device_id)
# model = torch.nn.DataParallel(model)

args_zeroshot_eval = config['zeroshot']

avg_f1, avg_acc, avg_auc = 0, 0, 0
for set_name in args_zeroshot_eval['test_sets'].keys():

        f1, acc, auc, _, _, _, res_dict = \
        zeroshot_eval(model=model, 
        set_name=set_name, 
        device=device_id, 
        args_zeroshot_eval=args_zeroshot_eval)

        avg_f1 += f1
        avg_acc += acc
        avg_auc += auc

avg_f1 = avg_f1/len(args_zeroshot_eval['test_sets'].keys())
avg_acc = avg_acc/len(args_zeroshot_eval['test_sets'].keys())
avg_auc = avg_auc/len(args_zeroshot_eval['test_sets'].keys())