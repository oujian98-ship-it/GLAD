"""
模型管理模块
负责模型的初始化、配置和管理
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

try:
    from utilis.test_ft import test_ft
    from model.ddpm_conditional import UNet_conditional, ConditionalDiffusion1D
    DDPM_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入DDPM模块: {e}")
    print("这可能是由于缺少ema_pytorch依赖或其他模块路径问题")
    DDPM_AVAILABLE = False
    
    # 提供简单的替代实现
    class UNet_conditional(nn.Module):
        def __init__(self, dim, dim_mults=(1, 2, 4, 8), channels=1, num_classes=10):
            super().__init__()
            self.fc = nn.Linear(dim, dim)
        
        def forward(self, x, t, y):
            return x
    
    class ConditionalDiffusion1D(nn.Module):
        def __init__(self, unet, seq_length=64, timesteps=1000, objective='pred_x0'):
            super().__init__()
            self.model = unet
            self.seq_length = seq_length
            self.timesteps = timesteps
            self.objective = objective
            self.loss_weight = torch.ones(timesteps)
        
        def q_sample(self, x_start, t, noise=None):
            if noise is None:
                noise = torch.randn_like(x_start)
            return noise
        
        def model_predictions(self, x, t, y):
            class Result:
                def __init__(self, shape):
                    self.pred_x_start = torch.randn(shape)
                    self.pred_noise = torch.randn(shape)
            return Result(x.shape)
    
    def test_ft(datapath, args, modelpath, crt_modelpath, test_cfg):
        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3*32*32, 64)
            
            def forward_no_fc(self, x):
                batch_size = x.shape[0]
                x_flat = x.view(batch_size, -1)
                return self.fc(x_flat)
        
        return MockEncoder()

from .config import TrainingConfig


class SimpleClassifier(nn.Module):
    """
    简单分类器模型
    """
    
    def __init__(self, feature_dim: int, num_classes: int):
        """
        初始化分类器
        """
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        return self.fc(x)


class ModelManager:
    """
    模型管理器，负责模型的初始化和配置
    """
    
    def __init__(self, config: TrainingConfig, device: torch.device):
        """
        初始化模型管理器
        """
        self.config = config
        self.device = device
        
    def initialize_models(self, num_classes: int, dataset_info: Dict) -> Tuple[nn.Module, nn.Module, nn.Module, int]:
        """
        初始化所有模型，根据配置选择分类器类型
        """
        # 1. 加载预训练编码器 
        encoder = test_ft(
            datapath=self.config.datapath, 
            args=self.config, 
            modelpath=self.config.model_fixed, 
            crt_modelpath=None, 
            test_cfg=None
        )
        encoder.to(self.device)
        
        # 动态获取特征维度
        feature_dim = self._get_feature_dim(encoder)
        print(f"Detected feature dimension: {feature_dim}")
        
        # 2. 根据配置选择分类器类型
        if self.config.use_wcdas:
            # 使用WCDAS分类器
            from .loss_calculator import WCDASLoss
            classifier = WCDASLoss(
                in_features=feature_dim,
                out_features=num_classes,
                bias=False,
                gamma=self.config.wcdas_gamma,
                s_trainable=self.config.wcdas_trainable_scale
            )
            print("Using WCDAS classifier for training")
        else:
            # 使用标准分类器
            classifier = SimpleClassifier(feature_dim, num_classes)
            print("Using standard classifier for training")
        
        classifier.to(self.device)
        
        # 3. 初始化扩散模型
        diffusion_model = self._create_diffusion_model(feature_dim, num_classes)
        diffusion_model.to(self.device)
        
        return encoder, classifier, diffusion_model, feature_dim
    
    def initialize_wcdas_models(self, num_classes: int, dataset_info: Dict) -> Tuple[nn.Module, nn.Module, nn.Module, int]:
        """
        初始化WCDAS模型
        """
        # 1. 加载预训练编码器 
        encoder = test_ft(
            datapath=self.config.datapath, 
            args=self.config, 
            modelpath=self.config.model_fixed, 
            crt_modelpath=None, 
            test_cfg=None
        )
        encoder.to(self.device)
        
        # 动态获取特征维度
        feature_dim = self._get_feature_dim(encoder)
        print(f"Detected feature dimension: {feature_dim}")
        
        # 2. 初始化WCDAS分类器
        from .loss_calculator import WCDASLoss
        wcdas_classifier = WCDASLoss(
            in_features=feature_dim,
            out_features=num_classes,
            bias=False,
            gamma=self.config.wcdas_gamma,  # 使用配置中的参数
            s_trainable=self.config.wcdas_trainable_scale  # 使用配置中的参数
        )
        wcdas_classifier.to(self.device)
        
        # 3. 初始化扩散模型
        diffusion_model = self._create_diffusion_model(feature_dim, num_classes)
        diffusion_model.to(self.device)
        
        return encoder, wcdas_classifier, diffusion_model, feature_dim
    
    def _get_feature_dim(self, encoder: nn.Module) -> int:
        """
        获取编码器的特征维度
        """
        encoder.eval()
        with torch.no_grad():
            # 创建虚拟输入（CIFAR10/100: 3x32x32, ImageNet: 3x224x224）
            if self.config.dataset in ["CIFAR10", "CIFAR100"]:
                test_input = torch.randn(1, 3, 32, 32).to(self.device)
            else:  # ImageNet
                test_input = torch.randn(1, 3, 224, 224).to(self.device)
            
            # 前向传播获取特征
            test_features = encoder.forward_no_fc(test_input)
            feature_dim = test_features.shape[1]
        encoder.train()
        return feature_dim
    
    def _create_diffusion_model(self, feature_dim: int, num_classes: int) -> nn.Module:
        """
        创建扩散模型
        """
        # 创建UNet模型，作为扩散模型的核心去噪网络
        unet = UNet_conditional(
            dim=feature_dim,
            # CIFAR10/100使用更大的模型，ImageNet使用更小的模型（适应特征维度）
            dim_mults=(1, 2, 4, 8) if self.config.dataset in ["CIFAR10", "CIFAR100"] else (1, 2, 4),
            channels=1,                    
            num_classes=num_classes
        )
        
        # 创建条件扩散模型
        diffusion_model = ConditionalDiffusion1D(
            unet,                          
            seq_length=feature_dim,         
            timesteps=self.config.diffusion_steps,  
            objective='pred_x0'             
        )
        
        return diffusion_model
    

    def save_best_models(self, encoder, classifier, diffusion_model, accuracy, label_shift_acc):
        """
        保存最佳模型到saved_models文件夹
        """
        # 保存最佳准确率模型
        if accuracy > getattr(self, '_best_accuracy', -1):
            self._best_accuracy = accuracy
            save_path_ce = r"./saved_models/ckpt_best_ce_strategy_A.checkpoint"
            
            # 保存完整模型
            model_state = {
                'encoder_state_dict': encoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'diffusion_model_state_dict': diffusion_model.state_dict(),
                'accuracy': accuracy
            }
            
            torch.save(model_state, save_path_ce)
            print(f"The best accuracy model has been saved to: {save_path_ce}")
        
        # 保存最佳标签移位准确率模型
        if label_shift_acc > getattr(self, '_best_label_shift_acc', -1):
            self._best_label_shift_acc = label_shift_acc
            save_path_pc = r"./saved_models/ckpt_best_pc_strategy_A.checkpoint"
            
            model_state = {
                'encoder_state_dict': encoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'diffusion_model_state_dict': diffusion_model.state_dict(),
                'label_shift_acc': label_shift_acc
            }
            
            torch.save(model_state, save_path_pc)
            print(f"The model with the best label shift accuracy has been saved to: {save_path_pc}")
    
    def save_diffusion_model_to_pretrained(self, diffusion_model, epoch):
        """
        保存扩散模型到pretrained_models文件夹
        """
        # 创建保存路径，格式为diffusion_model_CIFAR10_imb_1_epoch_x.pt
        model_path = f"./pretrained_models/diffusion_model_{self.config.dataset}_imb_1_epoch_{epoch}.pt"
        
        # 保存模型状态字典
        torch.save(diffusion_model.state_dict(), model_path)
        print(f"The diffusion model has been saved to: {model_path}")
        