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
        self.eta_r = args.eta_r               # 现用于等半径约束
        self.lambda_sem = args.lambda_sem      
        self.gamma_ge = args.gamma_ge         
        #self.warmup_epochs = args.warmup_epochs  
        self.generation_interval = args.generation_interval  
        self.ddim_steps = args.ddim_steps
        
        # 等半径约束参数
        self.use_radius_constraint = args.use_radius_constraint  # 是否使用等半径约束
        self.target_radius = args.target_radius  # 默认目标半径值（用于没有样本的类别）
        
        
        # WCDAS相关参数
        self.use_wcdas = args.use_wcdas  # 是否使用WCDAS损失函数
        self.wcdas_gamma = args.wcdas_gamma  # WCDAS的初始gamma参数
        self.wcdas_trainable_scale =args.wcdas_trainable_scale  # WCDAS的缩放参数是否可训练
        
        # ==================== GALD-DC 增强参数 ====================
        # 分布校准参数 (Section 2.4)
        self.tau = args.tau  # 头部/尾部类别样本数阈值 (-1=自动计算)
        self.lambda_cal = args.lambda_cal  # 尾部类半径校准混合因子 λ
        self.beta_radius = args.beta_radius  # 半径 EMA 更新速率 β_r
        
        # 判别边距约束参数 (Section 2.6)
        self.eta_m = args.eta_m  # 边距损失权重
        self.margin_m = args.margin_m  # 边距距离 m
        
        # Stage 3 训练模式参数
        self.stage3_mode = args.stage3_mode  # 'stable' 或 'hybrid'
        self.beta_cons = args.beta_cons  # 一致性损失权重 (仅hybrid模式)
        self.gamma_pseudo = args.gamma_pseudo  # 伪特征分类损失权重
        
        # 三阶段分离训练配置
        self.stage1_end_epoch = args.stage1_end_epoch  # Stage 1 结束 epoch
        self.stage2_end_epoch = args.stage2_end_epoch  # Stage 2 结束 epoch
        
        # 数值稳定性参数 - 防止训练过程中的数值问题
        self.max_loss_weight = 5.0              
        self.max_diffusion_loss = 10.0            
        self.max_cov_loss = 5.0                  # 保留用于兼容性
        self.max_radius_loss = 50.0              # 等半径约束损失的最大值 (放宽以允许梯度传播)
        self.max_L_semantic = 20.0           
        self.max_grad_norm = 0.9                  
                
        # 梯度控制参数
        self.max_L_ge = 50.0                  
        self.feature_clamp_min = -10.0           
        self.feature_clamp_max = 10.0            

        self.gradient_accumulation_steps = 4      
        self.weight_decay = 2e-3                  # [减少过拟合] 增大正则化 (原 5e-4 → 2e-3)                  
        self.gradient_clipping_enabled = False    
        self.adaptive_lr_factor = 0.5             
        self.learning_rate_warmup_steps = 1000     
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ==================== Stage 3 显式校准参数 ====================
        # Stage 3伪特征显式校准机制
        self.enable_stage3_calibration = getattr(args, 'enable_stage3_calibration', True)  # 是否启用Stage 3校准
        self.stage3_calibration_strength = getattr(args, 'stage3_calibration_strength', 0.5)  # 校准强度 (0.0-1.0)
        
        # ==================== R2/R3 风险修复参数 ====================
        # R2: Hinge Loss 宽松半径约束 (允许缓冲区，避免球形假设过于严格)
        self.radius_slack = getattr(args, 'radius_slack', 0.5)  # Hinge Loss 缓冲区 δ
        # R3: EMA 冷启动保护 (Stage 2 前 N epochs 冻结 EMA 更新)
        self.ema_warmup_epochs = getattr(args, 'ema_warmup_epochs', 10)  # EMA 冻结期
        
        # ==================== R4/R6 风险修复参数 ====================
        # R4: Top-K Soft Margin (对 K 个最近负类取平均，避免梯度震荡)
        self.margin_top_k = getattr(args, 'margin_top_k', 3)  # Top-K 负类数量
        # R6: LoRA 适配 (允许 DM 轻量级跟踪 Encoder 特征空间漂移)
        self.enable_lora = getattr(args, 'enable_lora', True)  # 是否启用 LoRA
        self.lora_rank = getattr(args, 'lora_rank', 4)  # LoRA 秩 (越小越轻量)
        self.lora_alpha = getattr(args, 'lora_alpha', 8.0)  # LoRA 缩放因子
    
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
    
    @classmethod
    def from_optuna_trial(cls, trial, base_args, search_params: Dict = None) -> 'TrainingConfig':
        """
        从 Optuna Trial 创建配置对象
        
        Args:
            trial: Optuna Trial 对象
            base_args: 基础命令行参数
            search_params: 已采样的超参数字典 (可选，如果为 None 则自动采样)
        
        Returns:
            TrainingConfig: 配置对象
        """
        class Args:
            pass
        
        args = Args()
        
        # 复制基础参数
        for key, value in vars(base_args).items():
            setattr(args, key, value)
        
        # 如果提供了搜索参数，直接使用
        if search_params is not None:
            for key, value in search_params.items():
                setattr(args, key, value)
        
        return cls(args)
    
    def to_json(self) -> str:
        """将配置序列化为 JSON 字符串"""
        import json
        
        # 过滤掉不可序列化的对象
        serializable = {}
        for key, value in self.__dict__.items():
            if key == 'device':
                serializable[key] = str(value)
            elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable[key] = value
        
        return json.dumps(serializable, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TrainingConfig':
        """从 JSON 字符串创建配置对象"""
        import json
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    def update(self, key: str, value) -> None:
        """更新特定配置项"""
        setattr(self, key, value)
    
    def log_config(self) -> None:
        """打印当前配置信息并写入日志文件"""
        import logging
        
        lines = [
            "=" * 60,
            "GALD-DC Training Configuration",
            "=" * 60,
            f"dataset: {self.dataset}",
            f"imb_factor: {self.imb_factor}",
            f"epochs: {self.epochs}",
            f"learning_rate: {self.lr}",
            f"weight_decay: {self.weight_decay}",
            f"diffusion_steps: {self.diffusion_steps}",
            f"ddim_steps: {self.ddim_steps}",
            "-" * 60,
            "Loss Weights:",
            f"  lambda_ema: {self.lambda_ema}",
            f"  beta_radius: {getattr(self, 'beta_radius', self.lambda_ema)}",
            f"  eta_p (prototype loss): {self.eta_p}",
            f"  eta_r (radius loss): {self.eta_r}",
            f"  eta_m (margin loss): {self.eta_m}",
            f"  lambda_sem: {self.lambda_sem}",
            f"  gamma_ge (gen loss): {self.gamma_ge}",
            "-" * 60,
            "GALD-DC Parameters:",
            f"  tau (head/tail threshold): {self.tau}",
            f"  lambda_cal (calibration factor): {self.lambda_cal}",
            f"  margin_m (margin distance): {self.margin_m}",
            f"  stage3_mode: {self.stage3_mode}",
            f"  beta_cons (consistency weight): {self.beta_cons}",
            f"  gamma_pseudo (pseudo loss weight): {self.gamma_pseudo}",
            "-" * 60,
            "Stage 3 Explicit Calibration:",
            f"  enable_stage3_calibration: {self.enable_stage3_calibration}",
            f"  stage3_calibration_strength: {self.stage3_calibration_strength}",
            "-" * 60,
            "Three-Stage Training:",
            f"  Stage 1 (CE Pre-training): epochs 0-{self.stage1_end_epoch - 1}",
            f"  Stage 2 (Diffusion Training): epochs {self.stage1_end_epoch}-{self.stage2_end_epoch - 1}",
            f"  Stage 3 (Controlled Fine-tuning): epochs {self.stage2_end_epoch}-{self.epochs - 1}",
            "-" * 60,
            "Other Settings:",
            f"  use_wcdas: {self.use_wcdas}",
            f"  use_radius_constraint: {self.use_radius_constraint}",
            f"  generation_interval: {self.generation_interval}",
            "=" * 60,
        ]
        
        # 同时打印到控制台和写入日志文件
        for line in lines:
            print(line)
            logging.info(line)