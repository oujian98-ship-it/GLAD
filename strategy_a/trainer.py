import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np

from dataloader.Custom_Dataloader import Custom_dataset, data_loader_wrapper_cust
from dataloader.data_loader_wrapper import data_loader_wrapper
from dataloader.data_loader_wrapper import Custom_dataset_ImageNet
from utilis.config_parse import config_setup
from model.metrics import *
from model.label_shift_est import LSC

from .config import TrainingConfig
from .model_manager import ModelManager
from .loss_calculator import LossCalculator
from .training_monitor import TrainingMonitor


class StrategyATrainer:
    """
    训练策略A的主训练器 (WCDAS/CE + LDMLR集成版 - 最终修复版)
    
    修复内容:
    1. WCDAS 验证逻辑 (传入 Training Priors)
    2. 内存优化 (单套模型初始化)
    3. 样本统计逻辑 (区分真实分布与均匀分布)
    4. 课程学习与几何约束修复
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化辅助组件
        self.model_manager = ModelManager(config, self.device)
        self.loss_calculator = LossCalculator(config)
        self.monitor = TrainingMonitor(config)
        
        print(f"Using device: {self.device}")
    
    def train(self):
        # 1. 加载数据
        cfg, finish = config_setup(self.config.config, None, self.config.datapath, update=False)
        train_set, test_set, num_classes, dataset_info = self._load_data(cfg)
        
        # [关键修复 3] 获取正确的训练集类别统计量 (Training Priors)
        # WCDAS 验证时必须使用训练集的分布，而不是测试集或均匀分布
        if 'per_class_img_num' in dataset_info:
            # 列表转 Tensor
            train_class_counts = torch.tensor(dataset_info['per_class_img_num'], device=self.device).float()
        else:
            # 如果没有预存信息，则手动遍历一次训练集进行统计
            print(">>> Counting training samples per class...")
            train_class_counts = torch.zeros(num_classes, device=self.device)
            for _, labels in train_set:
                 train_class_counts.index_add_(0, labels.to(self.device), torch.ones_like(labels).float())
        
        # 2. 初始化模型 (根据配置决定模式，解决问题 2 内存开销)
        if self.config.use_wcdas:
            print(f">>> [Mode] WCDAS Training (Tail-aware Attack) + LDMLR Equipment")
            # 初始化 WCDAS 专用模型
            encoder, classifier, diffusion_model, feature_dim = \
                self.model_manager.initialize_wcdas_models(num_classes, dataset_info)
        else:
            print(f">>> [Mode] Standard CE Training (Baseline Attack) + LDMLR Equipment")
            # 初始化 标准 CE 模型
            encoder, classifier, diffusion_model, feature_dim = \
                self.model_manager.initialize_models(num_classes, dataset_info)
        
        # 3. 初始化优化器（使用差分学习率）
        # 编码器是预训练的，需要较小的学习率防止灾难性遗忘
        # 分类器和扩散模型需要正常学习率从零开始学习
        optimizer = optim.Adam([
            {'params': encoder.parameters(), 'lr': self.config.lr * 0.01},  # 编码器使用1%的学习率
            {'params': classifier.parameters(), 'lr': self.config.lr},      # 分类器使用正常学习率
            {'params': diffusion_model.parameters(), 'lr': self.config.lr}   # 扩散模型使用正常学习率
        ], weight_decay=self.config.weight_decay)
        
        print(f">>> [Optimizer] Encoder lr: {self.config.lr * 0.01:.6f}, Classifier lr: {self.config.lr:.6f}, Diffusion lr: {self.config.lr:.6f}")
        
        self.optimizer = optimizer
        self.scheduler = None # 如有需要可添加 Scheduler
        
        # 初始化扩散调度
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = self._get_diffusion_schedule()
        
        # 4. 初始化几何统计量 (原型 & 半径)
        with torch.no_grad():
            class_prototypes = self._initialize_class_prototypes(num_classes, feature_dim)
            class_prototypes = self._compute_true_class_prototypes(encoder, train_set, class_prototypes, num_classes, feature_dim)
            target_radii = self._compute_target_radii_from_real_features(encoder, train_set, class_prototypes, num_classes)
        
        # ==================== GALD-DC Stage 3-H 初始化 ====================
        # 5. 自动计算 tau (如果设置为 -1)
        if self.config.tau == -1:
            tau = self._compute_auto_tau(train_class_counts, self.config.imb_factor, self.config.dataset)
            print(f">>> [GALD-DC] 自动计算 tau: {tau}")
        else:
            tau = self.config.tau
            print(f">>> [GALD-DC] 使用手动指定 tau: {tau}")
        
        # 保存 tau 到实例变量供后续使用
        self.tau = tau
        
        # 6. 计算头部类全局半径先验 r_prior
        r_prior = self.loss_calculator.compute_head_class_prior(
            target_radii, train_class_counts, tau
        )
        print(f">>> [GALD-DC] r_prior (头部类半径先验): {r_prior:.4f}")
        print(f">>> [GALD-DC] Stage 3 模式: {self.config.stage3_mode}")
        
        # 统计头/尾类数量
        head_count = (train_class_counts >= tau).sum().item()
        tail_count = (train_class_counts < tau).sum().item()
        print(f">>> [GALD-DC] 头部类数量: {int(head_count)}, 尾部类数量: {int(tail_count)}")
        
        # 7. 创建冻结编码器副本 E^(0) - 但在 Stage 1 结束后才真正保存
        import copy
        self.r_prior = r_prior
        self.class_counts = train_class_counts  # 保存类别计数供尾部过采样使用
        
        # ==================== 三阶段分离训练 ====================
        print(f"\n{'='*60}")
        print("三阶段分离训练配置:")
        print(f"  Stage 1 (Enc+Cls预训练): Epoch 0-{self.config.stage1_end_epoch-1}")
        print(f"  Stage 2 (Diffusion训练): Epoch {self.config.stage1_end_epoch}-{self.config.stage2_end_epoch-1}")
        print(f"  Stage 3 (受控微调):      Epoch {self.config.stage2_end_epoch}-{self.config.epochs-1}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.epochs):
            stage = self._get_training_stage(epoch)
            loss_weights = self._get_dynamic_loss_weights(epoch)
            
            # Stage 1 结束时：一次性计算统计量并保存冻结编码器副本
            if epoch == self.config.stage1_end_epoch:
                print(f"\n{'='*60}")
                print(f"[Stage 1 完成] 计算类统计量并冻结 Encoder...")
                
                # ====== CE Baseline 评估 ======
                print(f"\n[CE Baseline 评估] Stage 1 结束时的纯 CE 训练结果:")
                test_loss, accuracy, label_shift_acc, mmf_acc, mmf_acc_pc = \
                    self._validate(encoder, classifier, test_set, dataset_info, train_class_counts)
                
                # 保存 CE Baseline 结果到 monitor
                self.monitor.ce_baseline_accuracy = accuracy
                self.monitor.ce_baseline_label_shift_acc = label_shift_acc
                self.monitor.ce_baseline_mmf = mmf_acc
                self.monitor.ce_baseline_mmf_pc = mmf_acc_pc
                
                print(f"  CE Baseline Accuracy: {100 * accuracy:.2f}%")
                print(f"  CE Baseline MMF: {mmf_acc}")
                print(f"  CE Baseline Label Shift Accuracy: {label_shift_acc:.2f}%")
                
                with torch.no_grad():
                    # 重新计算更准确的原型和半径（基于完全训练的 encoder）
                    class_prototypes = self._compute_true_class_prototypes(
                        encoder, train_set, class_prototypes, num_classes, feature_dim)
                    target_radii = self._compute_target_radii_from_real_features(
                        encoder, train_set, class_prototypes, num_classes)
                    
                    # 重新计算 r_prior
                    r_prior = self.loss_calculator.compute_head_class_prior(
                        target_radii, train_class_counts, self.tau)
                    self.r_prior = r_prior
                    print(f"  更新后 r_prior: {r_prior:.4f}")
                    
                    # 保存冻结编码器副本 E^(0)
                    self.frozen_encoder = copy.deepcopy(encoder)
                    self.frozen_encoder.eval()
                    for param in self.frozen_encoder.parameters():
                        param.requires_grad = False
                    print(f"  冻结编码器副本 E^(0) 已保存")
                print(f"{'='*60}\n")
            
            # 打印阶段切换信息
            if epoch == 0 or epoch == self.config.stage1_end_epoch or epoch == self.config.stage2_end_epoch:
                print(f"\n>>> [Stage {stage}] 开始 - Epoch {epoch}")
            
            # 训练一个 Epoch
            self._train_epoch(epoch, encoder, classifier, diffusion_model, optimizer,
                            train_set, class_prototypes, target_radii,
                            sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                            num_classes, feature_dim, loss_weights, train_class_counts,
                            r_prior)
            
            # 只在 Stage 3 计算准确率
            if stage == 3:
                test_loss, accuracy, label_shift_acc, mmf_acc, mmf_acc_pc = \
                    self._validate(encoder, classifier, test_set, dataset_info, train_class_counts)
                
                self.monitor.log_validation(epoch, accuracy, test_loss, label_shift_acc, mmf_acc, mmf_acc_pc)
                
                # 保存最佳模型
                if accuracy > self.monitor.best_accuracy:
                    self._save_checkpoint(epoch, encoder, classifier, diffusion_model, optimizer, accuracy, 'ce')
                
                if label_shift_acc > self.monitor.best_label_shift_acc:
                    self.monitor.best_label_shift_acc = label_shift_acc
                    self._save_checkpoint(epoch, encoder, classifier, diffusion_model, optimizer, label_shift_acc, 'pc')
            
            # 定期保存扩散模型
            if epoch % 50 == 0:
                self.model_manager.save_diffusion_model_to_pretrained(diffusion_model, epoch)
        
        # 训练结束
        self.encoder = encoder
        self.classifier = classifier
        self.diffusion_model = diffusion_model
        self._save_accuracy_history()
    
    def _get_training_stage(self, epoch: int) -> int:
        """返回当前训练阶段: 1, 2, 或 3"""
        if epoch < self.config.stage1_end_epoch:
            return 1
        elif epoch < self.config.stage2_end_epoch:
            return 2
        else:
            return 3
    
    def _train_epoch(self, epoch: int, encoder: nn.Module, classifier: nn.Module, 
                    diffusion_model: nn.Module, optimizer: optim.Optimizer,
                    train_set: DataLoader, 
                    class_prototypes: Dict[int, torch.Tensor],
                    target_radii: torch.Tensor,
                    sqrt_alphas_cumprod: torch.Tensor, 
                    sqrt_one_minus_alphas_cumprod: torch.Tensor,
                    num_classes: int, feature_dim: int, loss_weights: Dict[str, float],
                    train_class_counts: torch.Tensor, r_prior: float = 1.0):
        
        
        stage = self._get_training_stage(epoch)
        #训练 Encoder + Classifier，获取头部类半径先验
        if stage == 1:
            # Stage 1: 只训练 Encoder + Classifier (CE 预训练)
            encoder.train()
            classifier.train()
            diffusion_model.eval()
            for param in encoder.parameters(): param.requires_grad = True
            for param in classifier.parameters(): param.requires_grad = True
            for param in diffusion_model.parameters(): param.requires_grad = False
        
        elif stage == 2:
            # Stage 2: 冻结 Encoder + Classifier，只训练 Diffusion
            encoder.eval()
            classifier.eval()
            diffusion_model.train()
            for param in encoder.parameters(): param.requires_grad = False
            for param in classifier.parameters(): param.requires_grad = False
            for param in diffusion_model.parameters(): param.requires_grad = True
            
        else:  # stage == 3
            # Stage 3: 受控微调 (根据模式)
            if self.config.stage3_mode == 'hybrid':
                # Stage 3-H: 解冻 Encoder，训练 Encoder + Classifier，冻结 Diffusion
                encoder.train()
                classifier.train()
                diffusion_model.eval()
                for param in encoder.parameters(): param.requires_grad = True
                for param in classifier.parameters(): param.requires_grad = True
                for param in diffusion_model.parameters(): param.requires_grad = False
            else:
                # Stage 3-S: 冻结 Encoder 和 Diffusion，仅训练 Classifier
                encoder.eval()
                classifier.train()
                diffusion_model.eval()
                for param in encoder.parameters(): param.requires_grad = False
                for param in classifier.parameters(): param.requires_grad = True
                for param in diffusion_model.parameters(): param.requires_grad = False
        
        # 添加损失跟踪 (包含 Stage 2 详细损失)
        running_losses = {
            'real': 0.0, 'diffusion': 0.0, 'prototype': 0.0, 'radius': 0.0,
            'margin': 0.0, 'semantic': 0.0, 'gen': 0.0, 'consistency': 0.0, 'total': 0.0
        }
        total_loss = 0.0 
        
        for batch_idx, (inputs, labels) in enumerate(train_set):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward - 根据训练阶段决定是否需要梯度
            if stage == 3 and self.config.stage3_mode == 'hybrid':
                # Stage 3-H: Encoder 需要梯度，同时计算冻结编码器特征用于一致性损失
                real_features = encoder.forward_no_fc(inputs)
                with torch.no_grad():
                    frozen_features = self.frozen_encoder.forward_no_fc(inputs)
            elif stage == 1:
                # Stage 1: Encoder 需要梯度
                real_features = encoder.forward_no_fc(inputs)
                frozen_features = None
            else:
                # Stage 2 或 Stage 3-S: Encoder 冻结
                with torch.no_grad():
                    real_features = encoder.forward_no_fc(inputs)
                frozen_features = None
            
            # 计算损失 (传入 stage, frozen_features, r_prior)
            losses = self._compute_batch_losses(
                encoder, classifier, diffusion_model, 
                real_features, inputs, labels,
                class_prototypes, target_radii, sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod, num_classes, feature_dim, epoch, batch_idx, loss_weights,
                train_class_counts, stage, frozen_features, r_prior
            )
            
            losses['total'].backward()
            self._control_gradients(encoder, classifier, diffusion_model, epoch * len(train_set), batch_idx)
            optimizer.step()
            optimizer.zero_grad()

            # Stage 2 EMA 更新统计量 (在扩散模型训练期间持续更新)
            if stage == 2:
                self._update_stats_ema(real_features.detach(), labels, class_prototypes, target_radii, num_classes)
            
            for key in running_losses:
                running_losses[key] += losses[key].item()
            total_loss += losses['total'].item()
            
            # 日志记录 (传入 stage 参数)
            if batch_idx % 100 == 0:
                self.monitor.log_batch_progress(epoch, batch_idx, {k: v.item() for k, v in losses.items()}, stage)
        
        train_loss = total_loss / len(train_set)
        # 只在 Stage 3 计算训练集准确率
        if stage == 3:
            train_accuracy = self._compute_train_accuracy(encoder, classifier, train_set, train_class_counts, num_classes)
        else:
            train_accuracy = None
        
        avg_losses = {key: value / len(train_set) for key, value in running_losses.items()}
        self.monitor.log_epoch_summary(epoch, avg_losses, train_accuracy, train_loss, stage)
    
    def _compute_batch_losses(self, encoder, classifier, diffusion_model, 
                             real_features: torch.Tensor, inputs: torch.Tensor, 
                             labels: torch.Tensor, class_prototypes: Dict[int, torch.Tensor],
                             target_radii: torch.Tensor, 
                             sqrt_alphas_cumprod: torch.Tensor,
                             sqrt_one_minus_alphas_cumprod: torch.Tensor,
                             num_classes: int, feature_dim: int, 
                             epoch: int, batch_idx: int, loss_weights: Dict[str, float],
                             train_class_counts: torch.Tensor,
                             stage: int = 1, frozen_features: torch.Tensor = None,
                             r_prior: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        计算批次损失 (三阶段分离训练版)
        
        Stage 1: L = L_CE (纯交叉熵预训练)
        Stage 2: L = L_diffusion + η_p*L_proto + η_r*L_rad + η_m*L_margin (扩散模型训练)
        Stage 3: L = L_real + γ*L_ge + β*L_cons (受控微调)
        """
        batch_size = inputs.size(0)
        
        # 初始化损失值
        real_loss = torch.tensor(0.0, device=self.device)
        semantic_loss = torch.tensor(0.0, device=self.device)
        gen_loss = torch.tensor(0.0, device=self.device)
        margin_loss = torch.tensor(0.0, device=self.device)
        consistency_loss = torch.tensor(0.0, device=self.device)
        
        if stage == 1:
            # ==================== Stage 1: 纯 CE 预训练 ====================
            if self.config.use_wcdas:
                real_loss = self.loss_calculator.compute_real_loss(
                    classifier, real_features, labels, train_class_counts, 'wcdas'
                )
            else:
                real_loss = self.loss_calculator.compute_real_loss(classifier, real_features, labels)
            
            total_loss = real_loss
            
        elif stage == 2:
            # ==================== Stage 2: 扩散模型 + 几何约束训练 ====================
            # 扩散损失
            diffusion_loss = self.loss_calculator.compute_diffusion_loss(
                diffusion_model, real_features, labels, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
            )
            
            # 估计去噪特征用于几何约束
            estimated_clean = self._estimate_clean_features(
                diffusion_model, real_features, labels, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
            )
            
            # 原型损失
            prototype_loss = self.loss_calculator.compute_prototype_loss(
                estimated_clean, labels, class_prototypes, num_classes
            )
            
            # 校准半径损失 (GALD-DC)
            calibrated_radii = self.loss_calculator.compute_calibrated_radius(
                target_radii, r_prior, train_class_counts, 
                self.tau, self.config.lambda_cal
            )
            radius_loss = self.loss_calculator.compute_radius_constraint_loss(
                estimated_clean, labels, class_prototypes, calibrated_radii, num_classes
            )
            
            # 判别边距损失 (GALD-DC)
            margin_loss = self.loss_calculator.compute_margin_loss(
                estimated_clean, labels, class_prototypes, num_classes, self.config.margin_m
            )
            
            # 数值保护
            diffusion_loss = self._safe_loss(diffusion_loss, 5.0)
            prototype_loss = self._safe_loss(prototype_loss, 10.0)
            radius_loss = self._safe_loss(radius_loss, 10.0)
            margin_loss = self._safe_loss(margin_loss, 10.0)
            
            # [修复] Stage 2 几何约束 warmup
            # 前 20% 的 Stage 2 epochs 使用较小的权重，让扩散模型先学习基本去噪
            stage2_progress = (epoch - self.config.stage1_end_epoch) / (self.config.stage2_end_epoch - self.config.stage1_end_epoch)
            warmup_factor = min(1.0, stage2_progress / 0.2)  # 前 20% 线性增长到 1.0
            
            # 总语义损失 (几何约束根据 warmup 调整)
            semantic_loss = (diffusion_loss + 
                           warmup_factor * self.config.eta_p * prototype_loss + 
                           warmup_factor * self.config.eta_r * radius_loss + 
                           warmup_factor * self.config.eta_m * margin_loss)
            semantic_loss = self._safe_loss(semantic_loss, self.config.max_semantic_loss)
            
            # Stage 2 总损失 (不包含 real_loss，因为分类器冻结)
            total_loss = semantic_loss
            
        else:  # stage == 3
            # ==================== Stage 3: 受控微调 ====================
            # 真实数据分类损失
            if self.config.use_wcdas:
                real_loss = self.loss_calculator.compute_real_loss(
                    classifier, real_features, labels, train_class_counts, 'wcdas'
                )
            else:
                real_loss = self.loss_calculator.compute_real_loss(classifier, real_features, labels)
            
            # 一致性损失 (仅 hybrid 模式)
            if self.config.stage3_mode == 'hybrid' and frozen_features is not None:
                consistency_loss = self.loss_calculator.compute_consistency_loss(
                    real_features, frozen_features
                )
                consistency_loss = self._safe_loss(consistency_loss, 5.0)
            
            # 伪特征分类损失 (on-the-fly 生成，带显式校准)
            gen_loss = self._compute_generation_loss(
                diffusion_model, classifier, batch_size, feature_dim, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, num_classes, batch_idx,
                class_prototypes=class_prototypes,  # 传入类原型
                target_radii=target_radii           # 传入目标半径
            )
            gen_loss = self._safe_loss(gen_loss, self.config.max_gen_loss)
            
            # Stage 3 总损失
            total_loss = (real_loss + 
                         self.config.gamma_pseudo * gen_loss + 
                         self.config.beta_cons * consistency_loss)
        
        # 根据阶段返回不同的损失字典
        if stage == 2:
            return {
                'real': real_loss,
                'diffusion': diffusion_loss,
                'prototype': prototype_loss,
                'radius': radius_loss,
                'margin': margin_loss,
                'semantic': semantic_loss,
                'gen': gen_loss,
                'consistency': consistency_loss,
                'total': total_loss
            }
        else:
            return {
                'real': real_loss,
                'diffusion': torch.tensor(0.0, device=self.device),
                'prototype': torch.tensor(0.0, device=self.device),
                'radius': torch.tensor(0.0, device=self.device),
                'margin': margin_loss,
                'semantic': semantic_loss,
                'gen': gen_loss,
                'consistency': consistency_loss,
                'total': total_loss
            }

    def _compute_generation_loss(self, diffusion_model, classifier, batch_size, feature_dim, 
                               sqrt_alphas, sqrt_one_minus_alphas, num_classes, batch_idx,
                               class_prototypes: Dict[int, torch.Tensor] = None,
                               target_radii: torch.Tensor = None):
        """
        计算生成特征的分类损失（带显式校准）
        
        改进: Stage 3显式应用GALD-DC校准机制
        """
        if batch_idx % self.config.generation_interval != 0:
            return torch.tensor(0.0, device=self.device)
        
        time_steps = torch.linspace(self.config.diffusion_steps-1, 0, 
                                  self.config.ddim_steps+1, dtype=torch.long, device=self.device)
        # [修复2] 对尾部类过采样，而非均匀采样
        # 计算采样权重：样本数越少的类，采样概率越高
        if hasattr(self, 'class_counts') and self.class_counts is not None:
            # 逆频率权重：少数类权重更高
            inv_freq = 1.0 / (self.class_counts.float() + 1e-6)
            sample_weights = inv_freq / inv_freq.sum()
            fake_labels = torch.multinomial(sample_weights, batch_size, replacement=True).to(self.device)
        else:
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=self.device)
        fake_features = torch.randn(batch_size, feature_dim, device=self.device)
        
        # DDIM采样生成伪特征
        fake_features = self._ddim_sample(diffusion_model, fake_features, fake_labels, 
                                        time_steps, sqrt_alphas, sqrt_one_minus_alphas)
        
        # ==================== 显式校准机制 (GALD-DC Stage 3) ====================
        # 检查是否启用校准且有必要参数
        if (getattr(self.config, 'enable_stage3_calibration', True) and 
            class_prototypes is not None and 
            target_radii is not None and
            hasattr(self, 'class_counts')):
            
            # 计算校准半径（与Stage 2一致）
            calibrated_radii = self.loss_calculator.compute_calibrated_radius(
                target_radii,              # 观测半径
                self.r_prior,              # 头部类先验
                self.class_counts,         # 类别样本数
                self.tau,                  # 头/尾阈值
                self.config.lambda_cal     # 校准混合因子
            )
            
            # 应用校准（调整特征位置使其符合校准半径）
            calibration_strength = getattr(self.config, 'stage3_calibration_strength', 0.5)
            fake_features = self._calibrate_features(
                fake_features,
                fake_labels,
                class_prototypes,
                calibrated_radii,
                calibration_strength=calibration_strength
            )
        
        # [针对问题 3 的修复]
        # 生成数据是类别平衡的，因此这里的 counts 应该是均匀分布 (Uniform)
        # 告诉 WCDAS 不要对这批数据做长尾去偏，否则会破坏生成数据的分布
        
        if self.config.use_wcdas:
            # 均匀分布的 counts
            uniform_counts = torch.ones(num_classes, device=self.device) * (batch_size / num_classes)
            
            # 调用 WCDAS 分类器
            if hasattr(classifier, 'forward'):
                try:
                    # 尝试传入 sample_per_class
                    outputs = classifier(
                        input=fake_features.detach(),
                        target=fake_labels,
                        sample_per_class=uniform_counts # Uniform!
                    )
                    # 处理可能的 tuple 返回 (logits, ...)
                    if isinstance(outputs, tuple):
                        fake_logits = outputs[0]
                    else:
                        fake_logits = outputs
                except TypeError:
                    # 如果 forward 不接受 sample_per_class (回退)
                    fake_logits = classifier(fake_features.detach())
            else:
                 fake_logits = classifier(fake_features.detach())
        else:
            fake_logits = classifier(fake_features.detach())
        
        gen_loss = F.cross_entropy(fake_logits, fake_labels)
        return gen_loss

    def _validate(self, encoder, classifier, test_set, dataset_info, train_class_counts):
        """
        验证函数 (修复版)
        :param train_class_counts: 训练集的类别统计量，用于 WCDAS 的 Test-time Adjustment
        """
        encoder.eval()
        classifier.eval()
        loss_fn = nn.CrossEntropyLoss()
        test_loss, correct, total = 0, 0, 0
        probs, labels_list = [], []
        
        num_classes = dataset_info['class_num']
        
        with torch.no_grad():
            for img, label in test_set:
                img, label = img.to(self.device), label.to(self.device)
                labels_list.append(label)
                features = encoder.forward_no_fc(img)
                
                # [针对问题 1 的修复] WCDAS 推理逻辑
                if self.config.use_wcdas:
                    # 关键: 必须传入训练集的先验分布 (train_class_counts)
                    # 这样 WCDAS 才能知道要把哪些类别的 Logits 压低/抬高
                    if hasattr(classifier, 'forward'):
                        try:
                            outputs = classifier(
                                input=features,
                                target=label, # 验证时不需要 target 计算梯度，但接口可能需要
                                sample_per_class=train_class_counts # Real Training Distribution!
                            )
                            if isinstance(outputs, tuple): logits = outputs[0]
                            else: logits = outputs
                        except TypeError:
                            logits = classifier(features)
                    else:
                        logits = classifier(features)
                else:
                    # 标准 CE 模式
                    logits = classifier(features)
                
                test_loss += loss_fn(logits, label).item()
                prob = F.softmax(logits, dim=1)
                probs.extend(list(prob.cpu().numpy()))
                pred = prob.argmax(dim=1)
                correct += (pred == label).type(torch.float).sum().item()
                total += label.size(0) # [修复问题 4] 正确累计样本数
        
        probs = np.array(probs)
        labels = torch.cat(labels_list)
        accuracy = correct / total
        test_loss /= len(test_set)
        
        # Metrics
        _, mmf_acc = self._get_metrics(probs, labels, dataset_info['per_class_img_num'])
        
        # Label Shift (仅在 CE 模式下作为参考，WCDAS 自带 Shift 修正)
        pc_probs = LSC(probs, cls_num_list=dataset_info['per_class_img_num'])
        label_shift_acc, mmf_acc_pc = self._get_metrics(pc_probs, labels, dataset_info['per_class_img_num'])
        
        return test_loss, accuracy, label_shift_acc, mmf_acc, mmf_acc_pc

    def _update_stats_ema(self, real_features, labels, class_prototypes, target_radii, num_classes):
        """动态更新类别原型和半径"""
        unique_labels = torch.unique(labels)
        
        if torch.isnan(real_features).any() or torch.isinf(real_features).any():
            return
        
        real_features = torch.clamp(real_features, -self.config.feature_clamp_max, self.config.feature_clamp_max)
        
        with torch.no_grad():
            for cls in unique_labels:
                cls_idx = cls.item()
                if not (0 <= cls_idx < num_classes): continue
                mask = (labels == cls)
                if mask.sum() == 0: continue
                
                # 更新原型
                cls_feats = real_features[mask]
                cls_mean = cls_feats.mean(dim=0)
                old_proto = class_prototypes[cls_idx]
                if torch.isnan(old_proto).any(): old_proto = cls_mean
                
                new_proto = (1 - self.config.lambda_ema) * old_proto + self.config.lambda_ema * cls_mean
                class_prototypes[cls_idx] = new_proto
                
                # [修复3] 更新半径 - 只有样本数 >= 2 时才更新，防止坍塌
                if mask.sum() >= 2:
                    dists = torch.norm(cls_feats - new_proto, p=2, dim=1)
                    current_avg_radius = dists.mean()
                    old_radius = target_radii[cls_idx]
                    # 使用独立的 beta_radius 参数
                    new_radius = (1 - self.config.beta_radius) * old_radius + self.config.beta_radius * current_avg_radius
                    target_radii[cls_idx] = new_radius
                # 样本数 < 2 时跳过半径更新，保持旧值

    def _ddim_sample(self, diffusion_model, fake_features, fake_labels, time_steps, sqrt_alphas, sqrt_one_minus_alphas):
        """
        DDIM 采样生成伪特征
        
        修复: 最后一步应该返回 estimated_clean 而非混合噪声的 fake_features
        """
        batch_size = fake_features.size(0)
        estimated_clean = fake_features  # 初始化
        
        for i in range(self.config.ddim_steps):
            current_step = time_steps[i]
            next_step = time_steps[i+1] if i < self.config.ddim_steps - 1 else -1
            
            fake_features_reshaped = fake_features.unsqueeze(1)
            batched_times = torch.full((batch_size,), current_step, device=self.device, dtype=torch.long)
            
            predictions = diffusion_model.model_predictions(fake_features_reshaped, batched_times, fake_labels)
            estimated_clean = predictions.pred_x_start.squeeze(1)
            predicted_noise = predictions.pred_noise.squeeze(1)
            
            estimated_clean = torch.clamp(estimated_clean, -10.0, 10.0)
            
            if next_step >= 0:
                alpha_next = sqrt_alphas[next_step]
                sigma = torch.sqrt(1 - alpha_next**2)
                fake_features = alpha_next * estimated_clean + sigma * predicted_noise
        
        # [修复] 返回最终的干净特征估计，而非混合噪声的 fake_features
        return estimated_clean
    
    def _calibrate_features(self, features: torch.Tensor, labels: torch.Tensor, 
                           class_prototypes: Dict[int, torch.Tensor], 
                           calibrated_radii: torch.Tensor,
                           calibration_strength: float = 1.0) -> torch.Tensor:
        """
        显式校准生成的伪特征，使其符合GALD-DC的半径约束
        
        核心思想:
        1. 对于每个生成特征，计算其到对应类中心的距离
        2. 如果距离与校准半径不一致，调整特征位置
        3. 保持方向不变，只调整半径
        
        公式:
            z_calibrated = μ_y + (z - μ_y) * (r_cal_y / ||z - μ_y||)
        
        其中:
            - z: 原始生成特征
            - μ_y: 类中心（原型）
            - r_cal_y: 校准半径（来自GALD-DC）
            - z_calibrated: 校准后的特征
        
        Args:
            features: 生成的特征 [batch_size, feature_dim]
            labels: 对应的类别标签 [batch_size]
            class_prototypes: 类别原型字典 {class_idx: prototype_tensor}
            calibrated_radii: 校准半径 [num_classes]
            calibration_strength: 校准强度 (0.0=不校准, 1.0=完全校准到目标半径)
            
        Returns:
            calibrated_features: 校准后的特征 [batch_size, feature_dim]
        """
        if calibration_strength <= 0.0:
            return features  # 不进行校准
        
        calibrated = features.clone()
        batch_size = features.size(0)
        
        for i in range(batch_size):
            cls_idx = labels[i].item()
            
            # 检查类别索引有效性
            if cls_idx not in class_prototypes:
                continue
            if cls_idx >= len(calibrated_radii):
                continue
            
            # 获取类中心和目标半径
            prototype = class_prototypes[cls_idx]
            target_radius = calibrated_radii[cls_idx]
            
            # 数值稳定性检查
            if torch.isnan(prototype).any() or torch.isinf(prototype).any():
                continue
            if torch.isnan(target_radius) or torch.isinf(target_radius):
                continue
            if target_radius <= 0:
                continue
            
            # 计算当前特征到类中心的方向向量
            direction = features[i] - prototype
            current_radius = torch.norm(direction, p=2)
            
            # 避免除零
            if current_radius < 1e-6:
                # 如果特征几乎等于原型，随机生成一个方向
                direction = torch.randn_like(direction)
                current_radius = torch.norm(direction, p=2)
            
            # 归一化方向向量
            direction_normalized = direction / current_radius
            
            # 计算目标位置（原型 + 方向 * 目标半径）
            target_position = prototype + direction_normalized * target_radius
            
            # 根据校准强度插值
            # strength=1.0: 完全移动到目标位置
            # strength=0.5: 移动到中间位置
            calibrated[i] = (1 - calibration_strength) * features[i] + \
                          calibration_strength * target_position
        
        return calibrated

    def _estimate_clean_features(self, diffusion_model, real_features, labels, sqrt_alphas, sqrt_one_minus_alphas):
        """
        估计去噪后的干净特征，用于几何约束计算
        
        修复: 使用较小的 t 值范围 (0-200)，避免在高噪声步骤时估计质量太差
        这样扩散模型可以更容易地学习几何结构
        """
        batch_size = real_features.size(0)
        # [修复] 限制 t 在较小范围内，提高估计质量
        # 使用 0-200 而非 0-999，因为高噪声步骤的估计质量很差
        max_t = min(200, self.config.diffusion_steps)
        t = torch.randint(0, max_t, (batch_size,), device=self.device)
        noise = torch.randn_like(real_features)
        
        noisy_features = sqrt_alphas[t].view(-1, 1) * real_features + \
                         sqrt_one_minus_alphas[t].view(-1, 1) * noise
        noisy_features_reshaped = noisy_features.unsqueeze(1)
        
        predictions = diffusion_model.model_predictions(noisy_features_reshaped, t, labels)
        estimated_clean_features = predictions.pred_x_start.squeeze(1)
        
        return estimated_clean_features

    def _safe_loss(self, loss_tensor, max_val=10.0):
        if torch.isnan(loss_tensor) or torch.isinf(loss_tensor):
            return torch.tensor(0.0, device=self.device)
        return torch.clamp(loss_tensor, max=max_val)

    # --- 辅助函数 ---

    def _load_data(self, cfg) -> Tuple[DataLoader, DataLoader, int, Dict]:
        # 保持原有逻辑
        if self.config.dataset == "CIFAR10" or self.config.dataset == "CIFAR100":
            dataset_info = Custom_dataset(self.config)
            train_set, _, test_set, dset_info = data_loader_wrapper_cust(dataset_info)
        elif self.config.dataset == "ImageNet":
            dataset_info = Custom_dataset_ImageNet(self.config)
            train_set, val_set, test_set, dset_info = data_loader_wrapper(cfg.dataset)
        
        num_classes = dset_info["class_num"]
        dataset_info = {
            "path": self.config.datapath,
            "class_num": num_classes,
            "dataset_name": self.config.dataset,
            "per_class_img_num": dset_info["per_class_img_num"]
        }
        return train_set, test_set, num_classes, dataset_info
    
    def _get_dynamic_loss_weights(self, epoch: int) -> Dict[str, float]:
        return {'lambda_sem': self.config.lambda_sem, 'gamma_ge': self.config.gamma_ge}
    
    def _compute_auto_tau(self, class_counts: torch.Tensor, imb_factor: float, dataset: str) -> int:
        """
        根据数据集分布自动计算 tau (头/尾类阈值)
        
        策略:
        1. 计算类别样本数的中位数
        2. 结合 imb_factor 进行调整
        3. 确保有合理的头/尾类比例 (约 30-70% 为头部类)
        
        Args:
            class_counts: 每个类别的样本数
            imb_factor: 不平衡因子
            dataset: 数据集名称
            
        Returns:
            tau: 自动计算的阈值
        """
        counts = class_counts.cpu().numpy()
        num_classes = len(counts)
        
        # 基础策略: 使用中位数附近的值作为阈值
        sorted_counts = np.sort(counts)[::-1]  # 降序排列
        median_count = np.median(counts)
        max_count = np.max(counts)
        min_count = np.min(counts)
        
        # 根据不平衡因子调整
        # imb_factor 越小，数据越不平衡，tau 应该越低以包含更多头部类
        if imb_factor <= 0.01:
            # 极端不平衡: 使用较低阈值，让更多类成为头部类
            # 目标: 约 30-40% 的类为头部类
            target_head_ratio = 0.35
        elif imb_factor <= 0.1:
            # 中等不平衡: 使用中等阈值
            # 目标: 约 50-60% 的类为头部类
            target_head_ratio = 0.55
        else:
            # 轻微不平衡
            target_head_ratio = 0.7
        
        # 计算使得头部类比例接近目标的 tau
        target_head_count = int(num_classes * target_head_ratio)
        target_head_count = max(1, min(num_classes - 1, target_head_count))  # 至少1个头部类和1个尾部类
        
        # tau = 第 target_head_count 大的样本数
        tau = int(sorted_counts[target_head_count - 1])
        
        # 确保 tau 至少为最小样本数的 2 倍，避免所有类都成为头部类
        tau = max(tau, int(min_count * 2))
        
        # 数据集特定调整
        if dataset == "CIFAR10":
            # CIFAR-10 类别少，使用较大的 tau
            tau = max(tau, 300)
        elif dataset == "CIFAR100":
            # CIFAR-100 类别多，使用较小的 tau
            tau = max(tau, 50)
        
        print(f">>> [Auto-tau] 样本数范围: [{int(min_count)}, {int(max_count)}], 中位数: {median_count:.0f}")
        print(f">>> [Auto-tau] 目标头部类比例: {target_head_ratio:.0%}, 计算 tau: {tau}")
        
        return tau
    
    def _initialize_class_prototypes(self, num_classes: int, feature_dim: int) -> Dict[int, torch.Tensor]:
        class_prototypes = {}
        for cls in range(num_classes):
            class_prototypes[cls] = torch.zeros(feature_dim).to(self.device)
        return class_prototypes
    
    def _get_diffusion_schedule(self) -> Tuple[torch.Tensor, torch.Tensor]:
        betas = torch.linspace(0.0001, 0.02, self.config.diffusion_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        return sqrt_alphas_cumprod.to(self.device), sqrt_one_minus_alphas_cumprod.to(self.device)
    
    def _compute_true_class_prototypes(self, encoder, dataloader, class_prototypes, num_classes, feature_dim):
        # 简化: 直接遍历一次
        class_features = [[] for _ in range(num_classes)]
        encoder.eval()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            real_features = encoder.forward_no_fc(inputs)
            for i in range(len(labels)):
                cls_idx = labels[i].item()
                if 0 <= cls_idx < num_classes:
                    class_features[cls_idx].append(real_features[i].detach())
        
        for cls_idx in range(num_classes):
            if len(class_features[cls_idx]) > 0:
                features_tensor = torch.stack(class_features[cls_idx])
                class_prototypes[cls_idx] = torch.mean(features_tensor, dim=0).detach()
        return class_prototypes
    
    def _compute_target_radii_from_real_features(self, encoder, dataloader, class_prototypes, num_classes):
        target_radii = torch.zeros(num_classes, device=self.device)
        class_counts = torch.zeros(num_classes, device=self.device)
        
        encoder.eval()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            real_features = encoder.forward_no_fc(inputs)
            for i in range(len(labels)):
                cls_idx = labels[i].item()
                if 0 <= cls_idx < num_classes:
                    feat = real_features[i].detach()
                    proto = class_prototypes[cls_idx]
                    dist = torch.norm(feat - proto, p=2)
                    target_radii[cls_idx] += dist
                    class_counts[cls_idx] += 1
        
        mask = class_counts > 0
        target_radii[mask] /= class_counts[mask]
        
        # 修改：使用有样本类别的平均半径作为无样本类别的默认值
        if mask.any():
            # 计算有样本类别的平均半径
            avg_radius = target_radii[mask].mean()
            target_radii[~mask] = avg_radius
            
        else:
            # 极端情况：所有类别都没有样本，使用原始默认值
            target_radii = self.config.target_radius * torch.ones(num_classes, device=self.device)
            
        return target_radii
    
    def _control_gradients(self, encoder, classifier, diffusion_model, global_step, batch_idx):
        self._normalize_gradients(encoder, classifier, diffusion_model)
        # 可以在此添加日志记录逻辑
        pass

    def _normalize_gradients(self, *models):
        for model in models:
            if model is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)

    def _get_metrics(self, probs, labels, cls_num_list):
        labels = [l.cpu().item() if isinstance(l, torch.Tensor) else l for l in labels]
        acc = acc_cal(probs, labels, method='top1')
        mmf_acc = list(mmf_acc_cal(probs, labels, cls_num_list))
        return acc, mmf_acc

    def _save_checkpoint(self, epoch, encoder, classifier, diffusion_model, optimizer, accuracy, model_type='ce'):
        """
        保存检查点
        
        Args:
            model_type: 'ce' for CE/WCDAS best model, 'pc' for Label Shift best model
        """
        checkpoint = {
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'classifier': classifier.state_dict(),
            'diffusion': diffusion_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': accuracy
        }
        # 文件名包含模式和模型类型
        mode = "wcdas" if self.config.use_wcdas else "ce"
        if model_type == 'ce':
            path = f"ckpt_strategy_A_{mode}_best_ce.pth"
            print(f"Saved best CE/WCDAS checkpoint (epoch {epoch}, acc {accuracy:.4f}) to {path}")
        else:  # model_type == 'pc'
            path = f"ckpt_strategy_A_{mode}_best_pc.pth"
            print(f"Saved best Label Shift checkpoint (epoch {epoch}, acc {accuracy:.4f}) to {path}")
        
        torch.save(checkpoint, path)

    def _compute_train_accuracy(self, encoder: nn.Module, classifier: nn.Module, 
                                train_set: DataLoader, train_class_counts: torch.Tensor, 
                                num_classes: int) -> float:
        """
        计算训练集准确率
        """
        encoder.eval()
        classifier.eval()
        
        correct = 0
        total = 0
        
        # 为节省计算时间，随机采样部分训练数据
        subset_size = min(1000, len(train_set.dataset))
        indices = torch.randperm(len(train_set.dataset))[:subset_size]
        subset_sampler = torch.utils.data.SubsetRandomSampler(indices)
        subset_loader = torch.utils.data.DataLoader(
            train_set.dataset, batch_size=train_set.batch_size, sampler=subset_sampler
        )
        
        with torch.no_grad():
            for inputs, labels in subset_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 提取特征
                features = encoder.forward_no_fc(inputs)
                
                # 根据分类器类型获取预测
                if self.config.use_wcdas:
                    if hasattr(classifier, 'forward'):
                        try:
                            outputs = classifier(
                                input=features,
                                target=labels,
                                sample_per_class=train_class_counts
                            )
                            if isinstance(outputs, tuple):
                                logits = outputs[0]
                            else:
                                logits = outputs
                        except TypeError:
                            logits = classifier(features)
                    else:
                        logits = classifier(features)
                else:
                    logits = classifier(features)
                
                # 计算准确率
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        # 恢复训练模式
        encoder.train()
        classifier.train()
        
        return correct / total if total > 0 else 0.0

    def _save_accuracy_history(self):
        """
        保存准确率历史记录
        """
        if hasattr(self, 'encoder') and hasattr(self, 'classifier') and hasattr(self, 'diffusion_model'):
            # 使用模型管理器保存最佳模型
            self.monitor.save_best_checkpoints(self.encoder, self.classifier, self.diffusion_model, self.model_manager)
        
        # 输出最佳模型准确率信息
        self.monitor.log_training_complete()