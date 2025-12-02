import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import logging
import argparse
from datetime import datetime

import torch
from torch import nn

from dataloader.Custom_Dataloader import Custom_dataset, data_loader_wrapper_cust
from dataloader.data_loader_wrapper import data_loader_wrapper
from dataloader.data_loader_wrapper import Custom_dataset_ImageNet
from utilis.feature_encode import feature_encode
from utilis.diffusion_model import diffusion_train
from utilis.utils import clear_cuda_cache
from utilis.config_parse import config_setup
from utilis.diffusion_model_colab import diffusion_train_colab
from utilis.test_ft import test_ft
from fine_tune_tr import fine_tune_fc
from eval import evaluation
from train_strategy_a import train_strategy_a


parser = argparse.ArgumentParser(description='Long-Tailed Diffusion Model training   ----Author: Pengxiao Han')
parser.add_argument('--datapath', default=r"E:\Projects\LDMLR-main\data", type=str, help='dataset path')
parser.add_argument('--config', default="./config/cifar10/cifar10_LSC_Mixup.txt", help='path to config file')

parser.add_argument('--epoch', default=400, type=int, help='epoch number to train')
parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset name it may be CIFAR10, CIFAR100 or ImageNet')
parser.add_argument('--imb_factor', default=0.01, type=float, help='long-tailed imbalance factor')
parser.add_argument('--diffusion_epoch', default=201, type=int, help='diffusion epoch to train')
parser.add_argument('--model_fixed', default='./pretrained_models/resnet32_cifar10_lt001.checkpoint', type=str, help='the encoder model path')
parser.add_argument('--feature_ratio', default=0.20, type=float, help='The ratio of generating feature')
parser.add_argument('--diffusion_step', default=1000, type=int, help='The steps of diffusion')
parser.add_argument('--checkpoint', default=None, type=str, help='model path to resume previous training, default None')
parser.add_argument('--batch_size_fc', default=1024, type=int, help='CNN fully connected layer batch size')
parser.add_argument('--learning_rate_fc', default=0.001, type=float, help='CNN fully connected layer learning rate')
parser.add_argument('--eval', default=None, type=str, help='evaluate the model performance')
parser.add_argument('--is_diffusion_pretrained', default = None, help='pre-trained diffusion model path. Training from scratch if None')
parser.add_argument('--generation_mmf', default=None, type=str, help='CNN fully connected layer batch size')

parser.add_argument('--training_strategy', default='strategy_a', type=str, choices=['original', 'strategy_a'], 
                    help='Training strategy to use: original or strategy_a')

# 训练策略A特有参数
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--lambda_ema', default=0.3, type=float, help='EMA decay rate for class prototypes')
parser.add_argument('--eta_p', default=0.05, type=float, help='weight for prototype loss')
parser.add_argument('--eta_r', default=0.05, type=float, help='weight for radius constraint loss (previously covariance loss)')
parser.add_argument('--lambda_sem', default=0.01, type=float, help='weight for semantic loss')
parser.add_argument('--gamma_ge', default=0.15, type=float, help='weight for generation loss')
parser.add_argument('--warmup_epochs', default=30, type=int, help='warmup epochs for semantic loss')
parser.add_argument('--generation_interval', default=10, type=int, help='generation interval for fake features')
parser.add_argument('--ddim_steps', default=50, type=int, help='DDIM sampling steps')

# 等半径约束参数
parser.add_argument('--use_radius_constraint', default=True, type=bool, help='whether to use radius constraint instead of covariance matching')
parser.add_argument('--target_radius', default=1.0, type=float, help='default target radius for equal radius constraint (used for classes with no samples)')

# WCDAS参数
parser.add_argument('--use_wcdas', default=True, type=bool, help='whether to use WCDAS for accuracy calculation') #WCDAS是否参与准确率计算，false时使用CE
parser.add_argument('--wcdas_gamma', default=0, type=float, help='initial gamma parameter for WCDAS')
parser.add_argument('--wcdas_traiFalsenable_scale', default=True, type=bool, help='whether the scale parameter in WCDAS is trainable')


def main():
    
    args = parser.parse_args()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    logs_dir = "./logs"
    log_filename = os.path.join(logs_dir, args.dataset + '_' + args.dataset + '_' +  str(args.imb_factor) + f"_{current_time}.log")
    print(f"Command arguments - datapath: {args.datapath}")
    print(f"Command arguments - model_fixed: {args.model_fixed}")
    print(f"Command arguments - config: {args.config}")
    print(f"Log file will be saved to: {log_filename}")
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    logging.info("--epoch: {} --dataset: {} --imb_factor:{} --feature_ratio:{} "
          "--diffusion_step:{} (DDPM official)".format(args.epoch, args.dataset, args.imb_factor,
                                                                             args.feature_ratio, args.diffusion_step))
    print("--epoch: {} --dataset: {} --imb_factor:{} --feature_ratio:{} "
          "--diffusion_step:{} (DDPM official)".format(args.epoch, args.dataset, args.imb_factor, args.feature_ratio, args.diffusion_step))
    
    # Load data ------------------------------------------------------------------------------------------------------------------
    # ouput: dataloader of cifar10/cifar100/ImageNet
    cfg, finish = config_setup(args.config,
                               args.checkpoint,
                               args.datapath,
                               update=False)
    if args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
        # customized dataset
        dataset_info = Custom_dataset(args)
        train_set, _, test_set, dset_info = data_loader_wrapper_cust(dataset_info)
    elif args.dataset == "ImageNet":
        dataset_info = Custom_dataset_ImageNet(args)
        train_set, val_set, test_set, dset_info = data_loader_wrapper(cfg.dataset)

    # If eval the model performance
    if args.eval != None:
        evaluation(test_set, dset_info, dataset_info, args, cfg)
        exit()

    # Select different training processes based on the training strategy
    if args.training_strategy == 'strategy_a':
        print("Using training strategy A: Synchronous joint training")
        train_strategy_a(args)
    else:
        print("Use the original training strategy: phased training")
        # 1. Encoder - encode images into features (batch_size, feature_dim) --------------------------------------------------------
        # input: image dataloader (batch_size, 3, 32, 32); output: feature dataloader (batch_size, 64)
        feature_dataset_tr, feature_dataloader_tr = feature_encode(train_set, dataset_info, args.model_fixed, args)

        # 2. Training a diffusion model and generate features
        # input: cifar training set, cifar testing set, feature dataloader; output: generated features by diffusion -----------------
        generated_features, fake_classes = diffusion_train(train_set, test_set, feature_dataloader_tr, dataset_info, dset_info, args)  # FIXME Official DDPM/DDIM

        # 3. Fine-tuning a fully-connected layer using generated features
        fine_tune_fc(generated_features, fake_classes, feature_dataset_tr, test_set, dataset_info, args, dset_info,cfg)

    print(" ------------Finish--------------")


if __name__ == '__main__':
    main()
