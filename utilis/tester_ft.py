import sys
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_init import model_init


def tester_ft(dataset, trainset, valset, model_config, args):
    # [Debug] 核心入口检查
    if not isinstance(model_config, dict):
        raise TypeError(f">>> [tester_ft] model_config MUST be a dict, but got {type(model_config)}. Value: {str(model_config)[:100]}")

    # --------------------------Load Backbone---------------------------------#
    if 'backbone' in model_config:
        if 'state_dict' in model_config and 'backbone' in model_config['state_dict']:
            backbone_state_dict = model_config['state_dict']['backbone']
        else:
            # Fallback for old structure
            backbone_path = model_config.get('backbone', {}).get('path')
            if backbone_path and os.path.exists(backbone_path):
                backbone_state_dict = torch.load(backbone_path)['state_dict']['model']
            else:
                backbone_state_dict = None
        
        if backbone_state_dict:
            backbone = model_init(model_config['backbone'],
                                model_config['dataset']['name'],
                                backbone_state_dict)
        else:
            backbone = None
    else:
        backbone = None
        
    # --------------------------Load Model------------------------------------#
    state_dict = model_config.get('state_dict', {})
    
    if 'model' in state_dict:
        model_state_dict = state_dict['model']
    elif 'backbone' in state_dict:
        model_state_dict = state_dict['backbone']
    else:
        # [修复] 如果 state_dict 已经直接包含了权重 (没有包装键)，则直接使用它
        # 这是一个常见的轻量级保存格式
        if any(not k.startswith('model.') and not k.startswith('backbone.') for k in state_dict.keys()):
            print(">>> [tester_ft] Using state_dict directly as model weights")
            model_state_dict = state_dict
        else:
            # 最后的退路：尝试从 checkpoint 路径加载 (通常不推荐，但在老代码中存在)
            ckpt_path = model_config.get('checkpoint', {}).get('save_path')
            if ckpt_path and os.path.exists(ckpt_path):
                model_state_dict = torch.load(ckpt_path)['state_dict']['model']
            else:
                raise KeyError(">>> [tester_ft] Could not find model weights in state_dict and no valid checkpoint path provided.")

    model = model_init(model_config['model'],
                       model_config['dataset']['name'],
                       model_state_dict)
    return model
