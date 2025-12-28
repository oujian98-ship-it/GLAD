import os
import argparse

import numpy as np
import torch
import time
import logging
import json
import copy

from utilis.tester_ft import tester_ft
from utilis.config_parse import config_setup
from dataloader.data_loader_wrapper import data_loader_wrapper
from dataloader.Custom_Dataloader import Custom_dataset, data_loader_wrapper_cust


def test_ft(datapath, args, modelpath=None, crt_modelpath=None, test_cfg=None):
    # ---------------------------Load Saved Model---------------------------#
    if modelpath is not None:
        try:
            raw_data = torch.load(modelpath)
            model_config = dict(raw_data)
            
            # [关键隔离] 如果 checkpoint 内部包含了 'config' 键 (这是 GALD-DC 保存格式特有的)
            # 我们将其重命名，避免与后续处理逻辑中的 config 变量产生命名空间污染
            if 'config' in model_config:
                print(">>> [test_ft] Isolated internal 'config' key to 'checkpoint_config'")
                model_config['checkpoint_config'] = model_config.pop('config')
                
            print(f">>> [test_ft] Config keys: {list(model_config.keys())}")
        except:
            raise Exception('Unable to load checkpoint via torch.load at %s , please check the path.' % modelpath)

    if crt_modelpath is not None:
        try:
            model_config = dict(torch.load(crt_modelpath))
        except:
            raise Exception('Unable to load checkpoint via torch.load at %s , please check the path.' % crt_modelpath)

    if (crt_modelpath is None) and (modelpath is None):
        raise Exception('Checkpoint of the model should be given in either --modelpath or --crt_modelpath.')
    elif (crt_modelpath is not None) and (modelpath is not None):
        print('warning: both --modelpath and --crt_modelpath are given, will ignore --modelpath.')

    # imb_logname = int(1 / model_config['dataset']['imb_factor']) if model_config['dataset']['imb_factor'] is not None else 'None'
    path = crt_modelpath if crt_modelpath is not None else modelpath
    # Generate print version of config (without ['state_dict'], ['train_info']['class_num_list'])
    cfg_print = copy.deepcopy({k: model_config[k] for k in set(list(model_config.keys())) - set(['state_dict'])})
    
    # [修复] 增加对不同版本 Checkpoint 结构的兼容性
    # 如果 dataset 是字符串，则说明是旧格式，需要转换为字典
    if isinstance(model_config.get('dataset'), str):
        print(f">>> [test_ft] Converting dataset string '{model_config['dataset']}' to dict")
        model_config['dataset'] = {'name': model_config['dataset']}
    
    # 确保 train_info 字典存在
    if 'train_info' not in model_config:
        model_config['train_info'] = {}

    # [修复] 补全缺失的 'model' 配置字典
    # model_init.py 要求 cfg 必须是字典，且包含 'name', 'fc_norm', 'ensemble_info' 等键
    if 'model' not in model_config or isinstance(model_config['model'], str):
        model_name = model_config.get('model', 'resnet32') if isinstance(model_config.get('model'), str) else 'resnet32'
        print(f">>> [test_ft] Constructing full model config dictionary for '{model_name}'")
        model_config['model'] = {
            'name': model_name,
            'output_dim': model_config.get('dataset', {}).get('class_num', 10),
            'fc_norm': False,
            'ensemble_info': {
                'name': 'none',
                'drop_rate': 0.5,
                'mask_rate': 2.0,
                'mask': [False, False, False, False],
                'ensemble_num': None
            },
            'gpu': None
        }
        
    if 'train_info' in cfg_print:
        cfg_print.get('train_info', {}).pop('class_num_list', None)

    # ----------------Loading the dataset, create dataloader----------------#
    # 确保 dataset 字典结构完整
    if 'name' not in model_config['dataset']:
        model_config['dataset']['name'] = args.dataset
        
    model_config['dataset']['path'] = datapath + model_config['dataset']['name']   
    if torch.cuda.device_count() == 1:
        model_config['dataset']['num_workers'] = 4

    if args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
        cust_dataset = Custom_dataset(args=args)
        train_set, val_set, test_set, dset_info = data_loader_wrapper_cust(cust_dataset)
    elif args.dataset == "ImageNet":
        cfg, finish = config_setup(args.config,
                                   args.checkpoint,
                                   args.datapath,
                                   update=False)
        train_set, val_set, test_set, dset_info = data_loader_wrapper(cfg.dataset)
    
    # 确保训练信息中有类别列表
    if 'train_info' not in model_config: model_config['train_info'] = {}
    model_config['train_info']['class_num_list'] = dset_info['per_class_img_num']

    # -------------------------Test the Model-------------------------------#
    logging.info('Test performance on test set.')
    print(f">>> [test_ft] Final check before tester_ft. model_config type: {type(model_config)}")
    model = tester_ft(test_set, train_set, val_set, model_config, args)
    return model


def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.round(4).tolist()
    elif isinstance(obj, np.float32):
        return round(float(obj), 4)
    raise TypeError('Not serializable')
