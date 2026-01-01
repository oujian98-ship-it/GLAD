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
from train_gald_dc import main as train_gald_dc


parser = argparse.ArgumentParser(description='Long-Tailed Diffusion Model training   ----Author: Pengxiao Han')
parser.add_argument('--datapath', default="./data", type=str, help='dataset path')
parser.add_argument('--config', default="./config/cifar10/cifar10_LSC_Mixup.txt", help='path to config file')


parser.add_argument('--dataset', default="CIFAR100", type=str, help='dataset name it may be CIFAR10, CIFAR100 or ImageNet')
parser.add_argument('--imb_factor', default=0.01, type=float, help='long-tailed imbalance factor')
parser.add_argument('--diffusion_epoch', default=201, type=int, help='diffusion epoch to train')
#parser.add_argument('--model_fixed', default='./pretrained_models/resnet32_cifar10_lt001.checkpoint', type=str, help='the encoder model path')
parser.add_argument('--model_fixed', default=None, type=str, help='the encoder model path (autodetected if None)')
parser.add_argument('--feature_ratio', default=0.20, type=float, help='The ratio of generating feature')
parser.add_argument('--diffusion_step', default=1000, type=int, help='The steps of diffusion')
parser.add_argument('--checkpoint', default=None, type=str, help='model path to resume previous training, default None')
parser.add_argument('--batch_size_fc', default=1024, type=int, help='CNN fully connected layer batch size')
parser.add_argument('--learning_rate_fc', default=0.001, type=float, help='CNN fully connected layer learning rate')
parser.add_argument('--eval', default=None, type=str, help='evaluate the model performance')
parser.add_argument('--is_diffusion_pretrained', default = None, help='pre-trained diffusion model path. Training from scratch if None')
parser.add_argument('--generation_mmf', default=None, type=str, help='CNN fully connected layer batch size')

parser.add_argument('--training_strategy', default='gald_dc', type=str, choices=['original', 'gald_dc'], 
                    help='Training strategy to use: original or gald_dc')

# GALD
parser.add_argument('--epoch', default=400, type=int, help='epoch number to train')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lambda_ema', default=0.2, type=float, help='EMA decay rate for class prototypes (β_proto)')
parser.add_argument('--beta_radius', default=0.1, type=float, help='EMA decay rate for class radii (β_radius)')
parser.add_argument('--lambda_sem', default=0.01, type=float, help='weight for semantic loss')
parser.add_argument('--gamma_ge', default=0.15, type=float, help='weight for generation loss')
parser.add_argument('--eta_p', default=0.01, type=float, help='weight for prototype loss')
parser.add_argument('--eta_r', default=0.1, type=float, help='weight for radius constraint loss (previously covariance loss)')

parser.add_argument('--generation_interval', default=5, type=int, help='generation interval for fake features')
parser.add_argument('--ddim_steps', default=100, type=int, help='DDIM sampling steps')

# Equal radius constraint parameter
parser.add_argument('--use_radius_constraint', default=True, type=bool, help='whether to use radius constraint instead of covariance matching')
parser.add_argument('--target_radius', default=1.0, type=float, help='default target radius for equal radius constraint (used for classes with no samples)')

# WCDAS parameter
parser.add_argument('--use_wcdas', default=False, type=bool, help='whether to use WCDAS for accuracy calculation') #WCDAS是否参与准确率计算，false时使用CE
parser.add_argument('--wcdas_gamma', default=0, type=float, help='initial gamma parameter for WCDAS')
parser.add_argument('--wcdas_trainable_scale', default=False, type=bool, help='whether the scale parameter in WCDAS is trainable')

# Distribution calibration parameters (Section 2.4)
parser.add_argument('--tau', default=-1, type=int, 
                    help='Head/Tail category sample count threshold (-1=Auto calculation, Positive integer=Manual specification)')
parser.add_argument('--lambda_cal', default=0.1, type=float, help='Tail radius calibration blending factor (0=pure prior, 1=pure observed)')

# Margin constraint parameters 
parser.add_argument('--eta_m', default=0.3, type=float, help='Margin loss weight')
parser.add_argument('--margin_m', default=3.5, type=float, help='Margin distance m')

# Stage 3 training mode parameters
parser.add_argument('--stage3_mode', default='hybrid', type=str, choices=['stable', 'hybrid'],
                    help="Stage 3 training mode: 'stable'(freeze Encoder) or 'hybrid'(unfreeze Encoder+consistency loss)")
parser.add_argument('--beta_cons', default=0.15, type=float, help='Consistency loss weight ')
parser.add_argument('--gamma_pseudo', default=0.6, type=float, help='Pseudo-feature classification loss weight γ (hybrid mode only)')

parser.add_argument('--stage1_end_epoch', default=200, type=int, help='Stage 1 end epoch (Enc+Cls pre-training)')
parser.add_argument('--stage2_end_epoch', default=250, type=int, help='Stage 2 end epoch (Diffusion training)')

# Stage 3 Explicit Calibration
parser.add_argument('--enable_stage3_calibration', default=True, type=bool, help='Enable Stage 3 explicit calibration')
parser.add_argument('--stage3_calibration_strength', default=0.5, type=float, help='Stage 3 calibration strength (0.0-1.0)')


def main():
    
    args = parser.parse_args()
    
    # Auto-select model_fixed if not provided
    if args.model_fixed is None:
        if args.dataset == "CIFAR10":
            if args.imb_factor == 0.01:
                args.model_fixed = './pretrained_models/resnet32_cifar10_lt001.checkpoint'
            elif args.imb_factor == 0.1:
                args.model_fixed = './pretrained_models/resnet32_cifar10_lt01.checkpoint'
        elif args.dataset == "CIFAR100":
            if args.imb_factor == 0.01:
                args.model_fixed = './pretrained_models/resnet32_cifar100_lt001.checkpoint'
            elif args.imb_factor == 0.1:
                args.model_fixed = './pretrained_models/resnet32_cifar100_lt01.checkpoint'
        elif args.dataset == "ImageNet":
            args.model_fixed = './pretrained_models/imagenetlt-resnet10.checkpoint'
            
        if args.model_fixed:
            print(f">>> [Auto-Select] Selected model_fixed: {args.model_fixed}")
        else:
            print(f">>> [Warning] No model_fixed found for dataset={args.dataset}, imb_factor={args.imb_factor}")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    logs_dir = os.path.join("./logs", f"{args.dataset}_{args.imb_factor}")
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = os.path.join(logs_dir, f"{args.dataset}_{args.imb_factor}_{current_time}.log")
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    
    
    # Load data ------------------------------------------------------------------------------------------------------------------
    # ouput: dataloader of cifar10/cifar100/ImageNet
    cfg, finish = config_setup(args.config,
                               args.checkpoint,
                               args.datapath,
                               update=False)
    if args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
        # customized dataset
        dataset_info = Custom_dataset(args)
        train_set, val_set, test_set, dset_info = data_loader_wrapper_cust(dataset_info)
    elif args.dataset == "ImageNet":
        dataset_info = Custom_dataset_ImageNet(args)
        train_set, val_set, test_set, dset_info = data_loader_wrapper(cfg.dataset)

    # If eval the model performance
    if args.eval != None:
        evaluation(test_set, dset_info, dataset_info, args, cfg)
        exit()

    # Select different training processes based on the training strategy
    if args.training_strategy == 'gald_dc':
        print("Using training strategy: GALD-DC (Synchronous joint training)")
        train_gald_dc(args)
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
