"""
支持 Optuna 的 GALD-DC 训练器

继承自 StrategyATrainer，添加：
1. 中间结果上报功能（用于剪枝判断）
2. Trial 对象集成
3. 训练进度回调
"""

import os
import sys
import torch
import optuna
from typing import Optional, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gald_dc.trainer import GALDDCTrainer
from gald_dc.config import TrainingConfig


class OptunaStrategyATrainer(GALDDCTrainer):
    """
    支持 Optuna 超参数优化的训练器
    
    继承自 StrategyATrainer，添加以下功能：
    1. 在 Stage 3 的每个验证点向 Optuna 上报准确率
    2. 支持 Optuna 的剪枝机制（提前终止效果不好的试验）
    3. 记录并返回最佳准确率供 Optuna 优化
    """
    
    def __init__(self, config: TrainingConfig, trial: Optional[optuna.Trial] = None,
                 report_interval: int = 10):
        """
        初始化 Optuna 训练器
        
        Args:
            config: 训练配置
            trial: Optuna Trial 对象 (可选，用于上报结果和剪枝)
            report_interval: 上报间隔 (每 N 个 epoch 上报一次)
        """
        super().__init__(config)
        self.trial = trial
        self.report_interval = report_interval
        self.best_accuracy = 0.0
        self.pruned = False
    
    def train(self) -> float:
        """
        执行训练并返回最佳准确率
        
        Returns:
            float: Stage 3 期间达到的最佳验证准确率
        """
        # ==================== 配置日志 ====================
        import logging
        from datetime import datetime
        
        # 创建日志目录
        logs_dir = "./logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # 生成日志文件名 (包含 trial 编号)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        trial_id = self.trial.number if self.trial else 0
        log_filename = os.path.join(
            logs_dir, 
            f"optuna_trial{trial_id}_{self.config.dataset}_{self.config.imb_factor}_{current_time}.log"
        )
        
        # 配置日志 (每个 trial 独立的日志文件)
        # 移除之前的 handlers 避免重复
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        logging.basicConfig(
            filename=log_filename, 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        print(f"\n[Optuna Trial {trial_id}] 日志保存到: {log_filename}")
        logging.info(f"[Optuna Trial {trial_id}] 开始训练")
        
        # 打印并记录配置
        self.config.log_config()
        
        # ==================== 原有初始化逻辑 ====================
        from utilis.config_parse import config_setup
        import copy
        
        cfg, finish = config_setup(self.config.config, None, self.config.datapath, update=False)
        train_set, test_set, num_classes, dataset_info = self._load_data(cfg)
        
        # 获取训练集类别统计量
        if 'per_class_img_num' in dataset_info:
            train_class_counts = torch.tensor(dataset_info['per_class_img_num'], device=self.device).float()
        else:
            print(">>> Counting training samples per class...")
            train_class_counts = torch.zeros(num_classes, device=self.device)
            for _, labels in train_set:
                train_class_counts.index_add_(0, labels.to(self.device), torch.ones_like(labels).float())
        
        # 初始化模型
        if self.config.use_wcdas:
            encoder, classifier, diffusion_model, feature_dim = \
                self.model_manager.initialize_wcdas_models(num_classes, dataset_info)
        else:
            encoder, classifier, diffusion_model, feature_dim = \
                self.model_manager.initialize_models(num_classes, dataset_info)
        
        # 初始化优化器
        from torch import optim
        optimizer = optim.Adam([
            {'params': encoder.parameters(), 'lr': self.config.lr * 0.01},
            {'params': classifier.parameters(), 'lr': self.config.lr},
            {'params': diffusion_model.parameters(), 'lr': self.config.lr}
        ], weight_decay=self.config.weight_decay)
        
        self.optimizer = optimizer
        self.scheduler = None
        
        # 初始化扩散调度
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = self._get_diffusion_schedule()
        
        # 初始化几何统计量
        with torch.no_grad():
            class_prototypes = self._initialize_class_prototypes(num_classes, feature_dim)
            class_prototypes = self._compute_true_class_prototypes(encoder, train_set, class_prototypes, num_classes, feature_dim)
            target_radii = self._compute_target_radii_from_real_features(encoder, train_set, class_prototypes, num_classes)
        
        # GALD-DC 初始化
        if self.config.tau == -1:
            tau = self._compute_auto_tau(train_class_counts, self.config.imb_factor, self.config.dataset)
        else:
            tau = self.config.tau
        
        self.tau = tau
        r_prior = self.loss_calculator.compute_head_class_prior(target_radii, train_class_counts, tau)
        self.r_prior = r_prior
        self.class_counts = train_class_counts
        
        print(f"\n[Optuna Trial {trial_id}] 开始训练...")
        print(f"  lr={self.config.lr:.6f}, margin_m={self.config.margin_m:.2f}")
        print(f"  lambda_cal={self.config.lambda_cal:.2f}, eta_m={self.config.eta_m:.2f}")
        logging.info(f"[Optuna Trial {trial_id}] 超参数: lr={self.config.lr:.6f}, margin_m={self.config.margin_m:.2f}, lambda_cal={self.config.lambda_cal:.2f}, eta_m={self.config.eta_m:.2f}")
        
        # 训练循环
        for epoch in range(self.config.epochs):
            stage = self._get_training_stage(epoch)
            loss_weights = self._get_dynamic_loss_weights(epoch)
            
            # Stage 1 结束时的处理
            if epoch == self.config.stage1_end_epoch:
                with torch.no_grad():
                    class_prototypes = self._compute_true_class_prototypes(
                        encoder, train_set, class_prototypes, num_classes, feature_dim)
                    target_radii = self._compute_target_radii_from_real_features(
                        encoder, train_set, class_prototypes, num_classes)
                    r_prior = self.loss_calculator.compute_head_class_prior(
                        target_radii, train_class_counts, self.tau)
                    self.r_prior = r_prior
                    
                    # 保存冻结编码器副本
                    self.frozen_encoder = copy.deepcopy(encoder)
                    self.frozen_encoder.eval()
                    for param in self.frozen_encoder.parameters():
                        param.requires_grad = False
            
            # 训练一个 epoch
            self._train_epoch(epoch, encoder, classifier, diffusion_model, optimizer,
                             train_set, class_prototypes, target_radii,
                             sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                             num_classes, feature_dim, loss_weights, train_class_counts,
                             r_prior)
            
            # Stage 3: 验证并上报结果
            if stage == 3:
                test_loss, accuracy, label_shift_acc, mmf_acc, mmf_acc_pc = \
                    self._validate(encoder, classifier, test_set, dataset_info, train_class_counts)
                
                # 记录到日志 (与原项目 monitor.log_validation 格式一致)
                self.monitor.log_validation(epoch, accuracy, test_loss, label_shift_acc, mmf_acc, mmf_acc_pc)
                
                # 更新最佳准确率
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_label_shift_acc = label_shift_acc
                
                # 向 Optuna 上报中间结果 (用于剪枝判断)
                if self.trial is not None:
                    # 计算 Stage 3 内的相对 epoch
                    stage3_epoch = epoch - self.config.stage2_end_epoch
                    
                    if stage3_epoch % self.report_interval == 0:
                        self.trial.report(accuracy, stage3_epoch)
                        
                        # 检查是否应该剪枝
                        if self.trial.should_prune():
                            import logging
                            logging.info(f"[Optuna] Trial 被剪枝 @ epoch {epoch}, accuracy={100*accuracy:.2f}%")
                            print(f"\n[Optuna] Trial 被剪枝 @ epoch {epoch}, accuracy={100*accuracy:.2f}%")
                            self.pruned = True
                            raise optuna.TrialPruned()
                
                # 简化的进度日志
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}: Acc={100*accuracy:.2f}%, Best={100*self.best_accuracy:.2f}%")
        
        # 训练完成，记录最终结果
        import logging
        logging.info(f"[Optuna Trial] 训练完成! Best Accuracy: {100*self.best_accuracy:.2f}%")
        logging.info(f"[Optuna Trial] Best Label Shift Accuracy: {getattr(self, 'best_label_shift_acc', 0):.2f}%")
        print(f"\n[Optuna Trial] 训练完成! Best Accuracy: {100*self.best_accuracy:.2f}%")
        return self.best_accuracy


def create_optuna_trainer(trial: optuna.Trial, base_args, 
                          search_params: Dict[str, Any]) -> OptunaStrategyATrainer:
    """
    创建 Optuna 训练器实例
    
    Args:
        trial: Optuna Trial 对象
        base_args: 基础命令行参数
        search_params: 从 Optuna 采样的超参数
    
    Returns:
        OptunaStrategyATrainer: 配置好的训练器实例
    """
    # 合并基础参数和搜索参数
    class CombinedArgs:
        pass
    
    args = CombinedArgs()
    
    # 复制基础参数
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    
    # 覆盖搜索参数
    for key, value in search_params.items():
        setattr(args, key, value)
    
    # 创建配置和训练器
    config = TrainingConfig(args)
    trainer = OptunaStrategyATrainer(config, trial=trial)
    
    return trainer
