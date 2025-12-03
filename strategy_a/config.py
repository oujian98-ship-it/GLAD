import torch
from typing import Dict, List, Optional


class TrainingConfig:
    """
    训练策略A的配置类
    包含所有训练参数、超参数和数值稳定性参数
    """
    
    def __init__(self, args):
        # 基础参数 - 数据集和模型配置
        self.datapath = args.datapath           
        self.config = args.config                 
        self.epochs = args.epoch                
        self.dataset = args.dataset               
        self.imb_factor = args.imb_factor        
        self.model_fixed = args.model_fixed      
        self.diffusion_steps = args.diffusion_step 
        
        # 训练超参数
        self.lr = args.lr                       
        self.lambda_ema = args.lambda_ema       
        self.eta_p = args.eta_p               
        self.eta_r = args.eta_r               # 原协方差权重，现用于等半径约束
        self.lambda_sem = args.lambda_sem      
        self.gamma_ge = args.gamma_ge         
        self.warmup_epochs = args.warmup_epochs  
        self.generation_interval = args.generation_interval  
        self.ddim_steps = args.ddim_steps
        
        # 等半径约束参数
        self.use_radius_constraint = getattr(args, 'use_radius_constraint', True)  # 是否使用等半径约束
        self.target_radius = getattr(args, 'target_radius', 1.0)  # 默认目标半径值（用于没有样本的类别）
        
        # 学习率调度参数
        self.use_lr_scheduler = getattr(args, 'use_lr_scheduler', True)  # 是否使用学习率调度器
        self.lr_scheduler_type = getattr(args, 'lr_scheduler_type', 'cosine')  # 学习率调度器类型 ('cosine', 'step', 'multistep')
        self.lr_min_ratio = getattr(args, 'lr_min_ratio', 0.01)  # 最小学习率与初始学习率的比率
        self.lr_milestones = getattr(args, 'lr_milestones', [150, 250])  # 多步衰减的里程碑 (仅在多步衰减时使用)
        self.lr_gamma = getattr(args, 'lr_gamma', 0.1)  # 学习率衰减因子 (仅在步进衰减时使用)
        self.lr_warmup_epochs = getattr(args, 'lr_warmup_epochs', 10)  # 学习率预热轮数
        
        # WCDAS相关参数
        self.use_wcdas = getattr(args, 'use_wcdas', False)  # 是否使用WCDAS损失函数
        self.wcdas_gamma = getattr(args, 'wcdas_gamma', -1)  # WCDAS的初始gamma参数
        self.wcdas_trainable_scale = getattr(args, 'wcdas_traiFalsenable_scale', False)  # WCDAS的缩放参数是否可训练
        
        # 数值稳定性参数 - 防止训练过程中的数值问题
        self.max_loss_weight = 5.0              
        self.max_diffusion_loss = 10.0            
        self.max_cov_loss = 5.0                  # 保留用于兼容性
        self.max_radius_loss = 5.0               # 等半径约束损失的最大值
        self.max_semantic_loss = 20.0           
        self.max_grad_norm = 0.9                  
                
        # 梯度控制参数
        self.max_gen_loss = 50.0                  
        self.feature_clamp_min = -10.0           
        self.feature_clamp_max = 10.0            

        self.gradient_accumulation_steps = 4      
        self.weight_decay = 1e-4                  
        self.gradient_clipping_enabled = False    
        self.adaptive_lr_factor = 0.5             
        self.learning_rate_warmup_steps = 1000     
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def to_dict(self) -> Dict:
        """将配置转换为字典格式"""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TrainingConfig':
        """从字典创建配置对象"""
        class Args:
            pass
        
        args = Args()
        for key, value in config_dict.items():
            setattr(args, key, value)
        
        return cls(args)
    
    def update(self, key: str, value) -> None:
        """更新特定配置项"""
        setattr(self, key, value)
    
    def log_config(self) -> None:
        """打印当前配置信息"""
        print("=" * 50)
        print("Training Strategy A Configuration Information")
        print("=" * 50)
        print(f"dataset: {self.dataset}")
        print(f"imb_factor: {self.imb_factor}")
        print(f"epochs: {self.epochs}")
        print(f"learn rate: {self.lr}")
        print(f"diffusion_steps: {self.diffusion_steps}")
        print(f"lambda_sem: {self.lambda_sem}")
        print(f"gamma_ge: {self.gamma_ge}")
        print(f"generation_interval: {self.generation_interval}")
        print(f"ddim_steps: {self.ddim_steps}")
        print(f"use_lr_scheduler: {self.use_lr_scheduler}")
        print(f"lr_scheduler_type: {self.lr_scheduler_type}")
        print(f"lr_min_ratio: {self.lr_min_ratio}")
        print("=" * 50)
        print("Equal Radius Constraint Configuration")
        print("=" * 50)
        print(f"use_radius_constraint: {self.use_radius_constraint}")
        print(f"default target_radius: {self.target_radius}")
        print(f"eta_r (radius constraint weight): {self.eta_r}")
        print("=" * 50)
        print("WCDAS Configuration")
        print("=" * 50)
        print(f"use_wcdas: {self.use_wcdas}")
        print(f"wcdas_gamma: {self.wcdas_gamma}")
        print(f"wcdas_trainable_scale: {self.wcdas_trainable_scale}")
        print("=" * 50)