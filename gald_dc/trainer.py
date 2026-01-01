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
from .lora_adapter import apply_lora_to_diffusion_model


class GALDDCTrainer:
    """
    GALD-DC: Geometry-Aware Latent Diffusion with Distribution Calibration Trainer
    
    修复内容:
    1. WCDAS 验证逻辑 (传入 Training Priors)
    2. 内存优化 (单套模型初始化)
    3. 样本统计逻辑 (区分真实分布与均匀分布)
    4. 课程学习与几何约束修正
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
        train_set, val_set, test_set, num_classes, dataset_info = self._load_data(cfg)
    
        # 统计 Stage 3 的平均准确率
        stage3_accs = []
        stage3_ls_accs = []
        
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
        
        # 2. 初始化模型
        if self.config.use_wcdas:
            print(f">>> [Mode] WCDAS Training (Tail-aware Attack) + LDMLR Equipment")
            # 初始化 WCDAS 专用模型
            encoder, classifier, diffusion_model, feature_dim = \
                self.model_manager.initialize_wcdas_models(num_classes, dataset_info)
        else:
            print(f">>> [Mode] Standard CE Training (Baseline Attack) + LDMLR Equipment")
            # 初始化标准 CE 模型
            encoder, classifier, diffusion_model, feature_dim = \
                self.model_manager.initialize_models(num_classes, dataset_info)
        
        # 3. 初始化优化器
        # 编码器是预训练的，需要较小的学习率防止灾难性遗忘
        # 分类器和扩散模型需要正常学习率从零开始学习
        optimizer = optim.SGD([
            {'params': encoder.parameters(), 'lr': self.config.lr * 0.01},
            {'params': classifier.parameters(), 'lr': self.config.lr},
            {'params': diffusion_model.parameters(), 'lr': self.config.lr}
        ], momentum=0.9, nesterov=True, weight_decay=self.config.weight_decay)
        
        print(f">>> [Optimizer] SGD with momentum=0.9, nesterov=True")
        print(f">>> [Optimizer] Encoder lr: {self.config.lr * 0.01:.6f}, Classifier lr: {self.config.lr:.6f}, Diffusion lr: {self.config.lr:.6f}")
        
        self.optimizer = optimizer
        
        # 初始化扩散调度
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = self._get_diffusion_schedule()
        
        # 4. 初始化几何统计量 (原型 & 半径)
        with torch.no_grad():
            class_mu = self._initialize_class_mu(num_classes, feature_dim)
            class_mu = self._compute_true_class_mu(encoder, train_set, class_mu, num_classes, feature_dim)
            r_obs = self._compute_r_obs_from_real_features(encoder, train_set, class_mu, num_classes)
        
        # ==================== GALD-DC Stage 3-H 初始化 ====================
        # 5. 自动计算 tau (如果设置-1)
        if self.config.tau == -1:
            tau = self._compute_auto_tau(train_class_counts, self.config.imb_factor, self.config.dataset)
            print(f">>> [GALD-DC] Auto-computed tau: {tau}")
        else:
            tau = self.config.tau
            print(f">>> [GALD-DC] Using manual tau: {tau}")
        
        # 保存 tau 到实例变量供后续使用
        self.tau = tau
        
        # 6. 计算头部类全局半径先验 r_prior
        r_prior = self.loss_calculator.compute_head_class_prior(
            r_obs, train_class_counts, tau
        )
        print(f">>> [GALD-DC] r_prior (head class radius prior): {r_prior:.4f}")
        print(f">>> [GALD-DC] Stage 3 mode: {self.config.stage3_mode}")
        
        # Count head/tail classes
        head_count = (train_class_counts >= tau).sum().item()
        tail_count = (train_class_counts < tau).sum().item()
        print(f">>> [GALD-DC] Head classes: {int(head_count)}, Tail classes: {int(tail_count)}")
        
        # 7. 创建冻结编码器副本E^(0) - 但在 Stage 1 结束后才真正保存
        import copy
        self.r_prior = r_prior
        self.class_counts = train_class_counts  # 保存类别计数供尾部过采样使用
        
        # ==================== Three-Stage Training ====================
        print(f"\n{'='*60}")
        print("Three-Stage Training Configuration:")
        print(f"  Stage 1 (Enc+Cls Pre-training): Epoch 0-{self.config.stage1_end_epoch-1}")
        print(f"  Stage 2 (Diffusion Training):   Epoch {self.config.stage1_end_epoch}-{self.config.stage2_end_epoch-1}")
        print(f"  Stage 3 (Controlled Fine-tune): Epoch {self.config.stage2_end_epoch}-{self.config.epochs-1}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.epochs):
            stage = self._get_training_stage(epoch)
            loss_weights = self._get_dynamic_loss_weights(epoch)
            
            # Stage 1 Complete: Compute statistics and freeze encoder
            if epoch == self.config.stage1_end_epoch:
                print(f"\n{'='*60}")
                print(f"[Stage 1 Complete] Computing class statistics and freezing Encoder...")
                with torch.no_grad():
                    # 重新计算更准确的原型和半径（基于完全训练后的 encoder）
                    class_mu = self._compute_true_class_mu(
                        encoder, train_set, class_mu, num_classes, feature_dim)
                    r_obs = self._compute_r_obs_from_real_features(
                        encoder, train_set, class_mu, num_classes)
                    
                    # 重新计算 r_prior
                    r_prior = self.loss_calculator.compute_head_class_prior(
                        r_obs, train_class_counts, self.tau)
                    self.r_prior = r_prior
                    print(f"  Updated r_prior: {r_prior:.4f}")
                    
                    # 保存冻结编码器副本 E^(0)
                    self.frozen_encoder = copy.deepcopy(encoder)
                    self.frozen_encoder.eval()
                    for param in self.frozen_encoder.parameters():
                        param.requires_grad = False
                    print(f"  Frozen encoder copy E^(0) saved")
                print(f"{'='*60}\n")

            
            # Print stage transition info
            if epoch == 0 or epoch == self.config.stage1_end_epoch or epoch == self.config.stage2_end_epoch:
                print(f"\n>>> [Stage {stage}] Started - Epoch {epoch}")
            
            # 训练一个 Epoch
            train_loss, train_accuracy = self._train_epoch(epoch, encoder, classifier, diffusion_model, optimizer,
                            train_set, class_mu, r_obs,
                            sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                            num_classes, feature_dim, loss_weights, train_class_counts,
                            r_prior)
            
            # 只在 Stage 3 计算准确率并展示“三连报”
            if stage == 3:
                t_loss, t_acc, t_ls_acc, t_mmf, t_mmf_ls = \
                    self._validate(encoder, classifier, test_set, dataset_info, train_class_counts)
                
                self.monitor.log_validation(epoch, t_acc, t_loss, t_ls_acc, t_mmf, t_mmf_ls, mode='Test')
                
                if t_ls_acc > self.monitor.best_label_shift_acc:
                    self.monitor.best_label_shift_acc = t_ls_acc
                    self._save_checkpoint(epoch, encoder, classifier, diffusion_model, optimizer, t_ls_acc, 'pc')
                
                # 记录 Stage 3 准确率用于计算平均值
                stage3_accs.append(t_acc)
                stage3_ls_accs.append(t_ls_acc)
            
            if epoch == self.config.stage2_end_epoch - 1:
                print(f"\n[Stage 2 Complete] Saving final diffusion model...")
                self.model_manager.save_diffusion_model_to_pretrained(diffusion_model, epoch)
                
                if getattr(self.config, 'enable_lora', True):
                    print(f"\n[R6 LoRA] Injecting LoRA layers into diffusion model...")
                    self.lora_adapter, self.lora_params = apply_lora_to_diffusion_model(
                        diffusion_model,
                        rank=getattr(self.config, 'lora_rank', 4),
                        alpha=getattr(self.config, 'lora_alpha', 8.0)
                    )
                    
                    if self.lora_params:
                        optimizer.add_param_group({
                            'params': self.lora_params,
                            'lr': self.config.lr 
                        })
                        print(f"[R6 LoRA] LoRA parameters added to optimizer")
        
        
        self.encoder = encoder
        self.classifier = classifier
        self.diffusion_model = diffusion_model
        
        # 计算 Stage 3 平均准确率
        avg_acc = sum(stage3_accs) / len(stage3_accs) if stage3_accs else None
        avg_ls_acc = sum(stage3_ls_accs) / len(stage3_ls_accs) if stage3_ls_accs else None
        
        self._save_accuracy_history(avg_acc, avg_ls_acc)
    
    def _get_training_stage(self, epoch: int) -> int:
        """返回当前训练阶段: 1, 2, 3"""
        if epoch < self.config.stage1_end_epoch:
            return 1
        elif epoch < self.config.stage2_end_epoch:
            return 2
        else:
            return 3
    
    def _train_epoch(self, epoch: int, encoder: nn.Module, classifier: nn.Module, 
                    diffusion_model: nn.Module, optimizer: optim.Optimizer,
                    train_set: DataLoader, 
                    class_mu: Dict[int, torch.Tensor],
                    r_obs: torch.Tensor,
                    sqrt_alphas_cumprod: torch.Tensor, 
                    sqrt_one_minus_alphas_cumprod: torch.Tensor,
                    num_classes: int, feature_dim: int, loss_weights: Dict[str, float],
                    train_class_counts: torch.Tensor, r_prior: float = 1.0):
        
        
        stage = self._get_training_stage(epoch)
        #训练 Encoder + Classifier，获取头部类半径先验
        if stage == 1:
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
                encoder.train()
                classifier.train()
                diffusion_model.eval()
                for param in encoder.parameters(): param.requires_grad = True
                for param in classifier.parameters(): param.requires_grad = True
                for param in diffusion_model.parameters(): param.requires_grad = False
            else:
                encoder.eval()
                classifier.train()
                diffusion_model.eval()
                for param in encoder.parameters(): param.requires_grad = False
                for param in classifier.parameters(): param.requires_grad = True
                for param in diffusion_model.parameters(): param.requires_grad = False
        
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
                real_features = encoder.forward_no_fc(inputs)
                frozen_features = None
            else:
                with torch.no_grad():
                    real_features = encoder.forward_no_fc(inputs)
                frozen_features = None
            
            # 计算损失 (传入 stage, frozen_features, r_prior)
            losses = self._compute_batch_losses(
                encoder, classifier, diffusion_model, 
                real_features, inputs, labels,
                class_mu, r_obs, sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod, num_classes, feature_dim, epoch, batch_idx, loss_weights,
                train_class_counts, stage, frozen_features, r_prior
            )
            
            losses['total'].backward()
            self._control_gradients(encoder, classifier, diffusion_model, epoch * len(train_set), batch_idx)
            optimizer.step()
            optimizer.zero_grad()

            # [R3 修复] EMA 冷启动保护：Stage 2 前 N epochs 冻结 EMA 更新
            # 原因：早期噪声特征会污染 class_mu 和 r_obs，导致几何约束失效
            ema_warmup_epochs = getattr(self.config, 'ema_warmup_epochs', 10)
            if stage == 2:
                # 检查是否在 EMA 冻结期内
                stage2_epoch = epoch - self.config.stage1_end_epoch  # Stage 2 内的 epoch 序号
                if stage2_epoch >= ema_warmup_epochs:
                    # 解冻期：正常更新 EMA
                    self._update_stats_ema(real_features.detach(), labels, class_mu, r_obs, num_classes)
                # else: 冻结期，跳过 EMA 更新，使用 Stage 1 结束时计算的静态统计量
            
            for key in running_losses:
                running_losses[key] += losses[key].item()
            total_loss += losses['total'].item()
            
            # 日志记录 (传入 stage 参数)
            if batch_idx % 50 == 0:
                self.monitor.log_batch_progress(epoch, batch_idx, {k: v.item() for k, v in losses.items()}, stage)
        
        train_loss = total_loss / len(train_set)
        # 只在 Stage 3 计算训练集准确率
        if stage == 3:
            train_accuracy = self._compute_train_accuracy(encoder, classifier, train_set, train_class_counts, num_classes)
        else:
            train_accuracy = None
        
        avg_losses = {key: value / len(train_set) for key, value in running_losses.items()}
        self.monitor.log_epoch_summary(epoch, avg_losses, train_accuracy, train_loss, stage)
        return train_loss, train_accuracy
    
    def _compute_batch_losses(self, encoder, classifier, diffusion_model, 
                             real_features: torch.Tensor, inputs: torch.Tensor, 
                             labels: torch.Tensor, class_mu: Dict[int, torch.Tensor],
                             r_obs: torch.Tensor, 
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
        
        # 初始化损失 ?
        L_real = torch.tensor(0.0, device=self.device)
        L_semantic = torch.tensor(0.0, device=self.device)
        L_ge = torch.tensor(0.0, device=self.device)
        L_margin = torch.tensor(0.0, device=self.device)
        L_cons = torch.tensor(0.0, device=self.device)
        
        if stage == 1:
            # ==================== Stage 1: 纯 CE 预训练 ====================
            if self.config.use_wcdas:
                L_real = self.loss_calculator.compute_real_loss(
                    classifier, real_features, labels, train_class_counts, 'wcdas'
                )
            else:
                L_real = self.loss_calculator.compute_real_loss(classifier, real_features, labels)
            
            total_loss = L_real
            
        elif stage == 2:
            # ==================== Stage 2: 扩散模型 + 几何约束训练 ====================
            # 扩散损失
            L_ldm = self.loss_calculator.compute_diffusion_loss(
                diffusion_model, real_features, labels, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
            )
            
            # 估计去噪特征用于几何约束
            estimated_clean = self._estimate_clean_features(
                diffusion_model, real_features, labels, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                epoch=epoch
            )
            
            # 原型损失
            L_proto = self.loss_calculator.compute_prototype_loss(
                estimated_clean, labels, class_mu, num_classes
            )
            
            # 校准半径损失 (GALD-DC)
            r_cal = self.loss_calculator.compute_calibrated_radius(
                r_obs, r_prior, train_class_counts, 
                self.tau, self.config.lambda_cal
            )
            L_rad = self.loss_calculator.compute_radius_constraint_loss(
                estimated_clean, labels, class_mu, r_cal, num_classes
            )
            
            # 判别边距损失 (GALD-DC)
            L_margin = self.loss_calculator.compute_margin_loss(
                estimated_clean, labels, class_mu, num_classes, self.config.margin_m
            )
            
            
            L_ldm = self._safe_loss(L_ldm, 5.0)
            L_proto = self._safe_loss(L_proto, 10.0)
            L_rad = self._safe_loss(L_rad, 10.0)
            L_margin = self._safe_loss(L_margin, 10.0)
            
            # Stage 2 几何约束 warmup
            #  20% Stage 2 epochs 使用较小的权重，让扩散模型先学习基本去噪
            stage2_progress = (epoch - self.config.stage1_end_epoch) / (self.config.stage2_end_epoch - self.config.stage1_end_epoch)
            
            
            # 总语义损失(几何约束根据 warmup 调整)
            L_semantic = (L_ldm + 
                            self.config.eta_p * L_proto + 
                            self.config.eta_r * L_rad + 
                            self.config.eta_m * L_margin)
            L_semantic = self._safe_loss(L_semantic, self.config.max_L_semantic)
            
            # Stage 2 总损失
            total_loss = L_semantic
            
        else:  # stage == 3
            # ==================== Stage 3: 受控微调 ====================
            # 真实数据分类损失
            if self.config.use_wcdas:
                L_real = self.loss_calculator.compute_real_loss(
                    classifier, real_features, labels, train_class_counts, 'wcdas'
                )
            else:
                L_real = self.loss_calculator.compute_real_loss(classifier, real_features, labels)
            
            # 一致性损失( hybrid 模式)
            if self.config.stage3_mode == 'hybrid' and frozen_features is not None:
                L_cons = self.loss_calculator.compute_consistency_loss(
                    real_features, frozen_features
                )
                L_cons = self._safe_loss(L_cons, 5.0)
            
            # 伪特征分类损失 (on-the-fly 生成，带显式校准)
            L_ge = self._compute_generation_loss(
                diffusion_model, classifier, batch_size, feature_dim, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, num_classes, batch_idx,
                class_mu=class_mu,  # 传入类原型
                r_obs=r_obs           # 传入目标半径
            )
            L_ge = self._safe_loss(L_ge, self.config.max_L_ge)
            
            # Stage 3 总损失
            total_loss = (L_real + 
                         self.config.gamma_pseudo * L_ge + 
                         self.config.beta_cons * L_cons)
        
        # 根据阶段返回不同的损失字典
        if stage == 2:
            return {
                'real': L_real,
                'diffusion': L_ldm,
                'prototype': L_proto,
                'radius': L_rad,
                'margin': L_margin,
                'semantic': L_semantic,
                'gen': L_ge,
                'consistency': L_cons,
                'total': total_loss
            }
        else:
            return {
                'real': L_real,
                'diffusion': torch.tensor(0.0, device=self.device),
                'prototype': torch.tensor(0.0, device=self.device),
                'radius': torch.tensor(0.0, device=self.device),
                'margin': L_margin,
                'semantic': L_semantic,
                'gen': L_ge,
                'consistency': L_cons,
                'total': total_loss
            }

    def _compute_generation_loss(self, diffusion_model, classifier, batch_size, feature_dim, 
                               sqrt_alphas, sqrt_one_minus_alphas, num_classes, batch_idx,
                               class_mu: Dict[int, torch.Tensor] = None,
                               r_obs: torch.Tensor = None):
        """
        计算生成特征的分类损失（带显式校准）
        
        改进: Stage 3显式应用GALD-DC校准机制
        """
        if batch_idx % self.config.generation_interval != 0:
            return torch.tensor(0.0, device=self.device)
        
        time_steps = torch.linspace(self.config.diffusion_steps-1, 0, 
                                  self.config.ddim_steps+1, dtype=torch.long, device=self.device)
        # 对尾部类过采样，而非均匀采样
        # 计算采样权重：样本数越少的类，采样概率越高
        if hasattr(self, 'class_counts') and self.class_counts is not None:
            # 逆频率权重：少数类权重更大
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
            class_mu is not None and 
            r_obs is not None and
            hasattr(self, 'class_counts')):
            
            # 计算校准半径（与Stage 2一致）
            r_cal = self.loss_calculator.compute_calibrated_radius(
                r_obs,              # 观测半径
                self.r_prior,              # 头部类先验
                self.class_counts,         # 类别样本数
                self.tau,                  # 长尾阈值
                self.config.lambda_cal     # 校准混合因子
            )
            
            # 应用校准（调整特征位置使其符合校准半径）
            calibration_strength = getattr(self.config, 'stage3_calibration_strength', 0.5)
            fake_features = self._calibrate_features(
                fake_features,
                fake_labels,
                class_mu,
                r_cal,
                calibration_strength=calibration_strength
            )
        
        
       
        
        
        if self.config.use_wcdas:
            # 均匀分布的 counts
            uniform_counts = torch.ones(num_classes, device=self.device) * (batch_size / num_classes)
            
            
            if hasattr(classifier, 'forward'):
                try:
                    
                    outputs = classifier(
                        input=fake_features.detach(),
                        target=fake_labels,
                        sample_per_class=uniform_counts # Uniform!
                    )
                    # 处理可能是 tuple 返回 (logits, ...)
                    if isinstance(outputs, tuple):
                        fake_logits = outputs[0]
                    else:
                        fake_logits = outputs
                except TypeError:
                    # 如果 forward 不接 ?sample_per_class (回退)
                    fake_logits = classifier(fake_features.detach())
            else:
                 fake_logits = classifier(fake_features.detach())
        else:
            fake_logits = classifier(fake_features.detach())
        
        L_ge = F.cross_entropy(fake_logits, fake_labels)
        return L_ge

    def _validate(self, encoder, classifier, test_set, dataset_info, train_class_counts):
        """
        验证函数 (修复版)
        :param train_class_counts: 训练集的类别统计量，用于 WCDAS 和 Test-time Adjustment
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
                    # 这样 WCDAS 才能知道要把哪些类别 ?Logits 压低/抬高
                    if hasattr(classifier, 'forward'):
                        try:
                            outputs = classifier(
                                input=features,
                                target=label, # 验证时不需 target 计算梯度，但接口可能需要
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
        label_shift_acc /= 100.0  # 转换为比例 (0-1)
        
        return test_loss, accuracy, label_shift_acc, mmf_acc, mmf_acc_pc

    def _update_stats_ema(self, real_features, labels, class_mu, r_obs, num_classes):
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
                old_proto = class_mu[cls_idx]
                if torch.isnan(old_proto).any(): old_proto = cls_mean
                
                new_proto = (1 - self.config.lambda_ema) * old_proto + self.config.lambda_ema * cls_mean
                class_mu[cls_idx] = new_proto
                
                #只要有 1 个样本就更新
                if mask.sum() >= 1:
                    dists = torch.norm(cls_feats - new_proto, p=2, dim=1)
                    current_avg_radius = dists.mean()
                    old_radius = r_obs[cls_idx]
                    #使用独立的 beta_radius 参数
                    new_radius = (1 - self.config.beta_radius) * old_radius + self.config.beta_radius * current_avg_radius
                    r_obs[cls_idx] = new_radius
                #样本数 < 1 时跳过半径更新，保持旧值

    def _ddim_sample(self, diffusion_model, fake_features, fake_labels, time_steps, sqrt_alphas, sqrt_one_minus_alphas):
        """
        DDIM 采样生成伪特征
        """
        batch_size = fake_features.size(0)
        estimated_clean = fake_features  
        
        # 严格执行 ddim_steps 次迭代，确保到达 t=0
        for i in range(self.config.ddim_steps):
            current_step = time_steps[i]
            next_step = time_steps[i+1] # 这里的索引最多到 ddim_steps，对应 value 0
            
            fake_features_reshaped = fake_features.unsqueeze(1)
            batched_times = torch.full((batch_size,), current_step, device=self.device, dtype=torch.long)
            
            # 预测干净特征估计 (~z0)
            predictions = diffusion_model.model_predictions(fake_features_reshaped, batched_times, fake_labels)
            estimated_clean = predictions.pred_x_start.squeeze(1)
            predicted_noise = predictions.pred_noise.squeeze(1)
            
            estimated_clean = torch.clamp(estimated_clean, -self.config.feature_clamp_max, self.config.feature_clamp_max)
            
            # 如果还有下一步 (包括 next_step 为 0)，更新容器特征
            if next_step >= 0:
                alpha_next = sqrt_alphas[next_step]
                sigma = torch.sqrt(max(torch.tensor(0.0, device=self.device), 1 - alpha_next**2))
                fake_features = alpha_next * estimated_clean + sigma * predicted_noise
        
        return estimated_clean
    
    def _calibrate_features(self, features: torch.Tensor, labels: torch.Tensor, 
                           class_mu: Dict[int, torch.Tensor], 
                           r_cal: torch.Tensor,
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
            class_mu: 类别原型字典 {class_idx: prototype_tensor}
            r_cal: 校准半径 [num_classes]
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
            if cls_idx not in class_mu:
                continue
            if cls_idx >= len(r_cal):
                continue
            
            # 获取类中心和目标半径
            prototype = class_mu[cls_idx]
            target_radius = r_cal[cls_idx]
            
            # 数值稳定性检查
            if torch.isnan(prototype).any() or torch.isinf(prototype).any():
                continue
            if torch.isnan(target_radius) or torch.isinf(target_radius):
                continue
            if target_radius <= 0:
                continue
            
            # 计算当前特征到类中心的方向向 ?
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
            
            # 根据校准强度插 ?
            # strength=1.0: 完全移动到目标位置
            # strength=0.5: 移动到中间位置
            calibrated[i] = (1 - calibration_strength) * features[i] + \
                          calibration_strength * target_position
        
        return calibrated

    def _estimate_clean_features(self, diffusion_model, real_features, labels, sqrt_alphas, sqrt_one_minus_alphas, epoch=None):
        """
        估计去噪后的干净特征，用于几何约束计算
        
        优化: 使用课程学习策略动态调整 t 的范围
        """
        batch_size = real_features.size(0)
        
        # [优化] 根据 Stage 2 进度动态调整 max_t
        start_t = 200
        end_t = self.config.diffusion_steps
        
        if epoch is not None and epoch >= self.config.stage1_end_epoch and epoch < self.config.stage2_end_epoch:
            # 在 Stage 2 内计算进度 (0.0 -> 1.0)
            stage_len = self.config.stage2_end_epoch - self.config.stage1_end_epoch
            progress = (epoch - self.config.stage1_end_epoch) / max(1, stage_len)
            
            # 线性增加 max_t: 200 -> 1000
            current_max_t = int(start_t + (end_t - start_t) * progress)
            max_t = min(current_max_t, end_t)
        else:
            # 默认值 (Stage 2 之外或未提供 epoch)
            max_t = start_t
            
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

    def _load_data(self, cfg) -> Tuple[DataLoader, DataLoader, DataLoader, int, Dict]:
        # 保持原有逻辑
        if self.config.dataset == "CIFAR10" or self.config.dataset == "CIFAR100":
            dataset_info = Custom_dataset(self.config)
            train_set, val_set, test_set, dset_info = data_loader_wrapper_cust(dataset_info)
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
        
        return train_set, val_set, test_set, num_classes, dataset_info
    
    def _get_dynamic_loss_weights(self, epoch: int) -> Dict[str, float]:
        return {'lambda_sem': self.config.lambda_sem, 'gamma_ge': self.config.gamma_ge}
    
    def _compute_auto_tau(self, class_counts: torch.Tensor, imb_factor: float, dataset: str) -> int:
        """
        根据数据集的实际分布自动计算 tau (Head/Tail 分界阈值)
        
        算法逻辑优化：
        参考 Custom_Dataloader.py 中的 class_num_list 分布，利用几何平均数 (Geometric Mean) 
        作为长尾分布的自然分割点。该值在数学上对应 log-linear 分布的中点，能自动适应不同数据集的量级。
        
        计算示例 (基于 Custom_Dataloader.py):
        - CIFAR10  (0.01): min=50,  max=5000 -> tau = sqrt(50 * 5000) = 500
        - CIFAR100 (0.01): min=5,   max=500  -> tau = sqrt(5  * 500)  = 50
        - CIFAR10  (0.1):  min=500, max=5000 -> tau = sqrt(500* 5000) ≈ 1581
        """
        counts = class_counts.cpu().numpy()
        max_count = np.max(counts)
        min_count = np.min(counts)
        
        # 1. 核心计算：几何平均数 (Geometric Mean)
        # 它可以完美跨越 CIFAR-10 (数千量级) 和 CIFAR-100 (数百量级)
        tau = int(np.sqrt(max_count * min_count))
        
        # 2. 稳定性修正：确保 Head 类有足够的样本来估计稳定的半径 (robs)
        # 对于 CIFAR-100，50 样本是 Many-shot 的常用下限；CIFAR-10 则放宽到 100
        
        tau = max(tau, 100) if dataset == "CIFAR10" else max(tau, 50)
        
        print(f">>> [Auto-tau] Dataset: {dataset}, Range: [{int(min_count)}, {int(max_count)}]")
        print(f">>> [Auto-tau] Mathematical center (Geometric Mean): {int(np.sqrt(max_count * min_count))}")
        print(f">>> [Auto-tau] Final selected tau: {tau}")
        
        return tau
    
    def _initialize_class_mu(self, num_classes: int, feature_dim: int) -> Dict[int, torch.Tensor]:
        class_mu = {}
        for cls in range(num_classes):
            class_mu[cls] = torch.zeros(feature_dim).to(self.device)
        return class_mu
    
    def _get_diffusion_schedule(self) -> Tuple[torch.Tensor, torch.Tensor]:
        betas = torch.linspace(0.0001, 0.02, self.config.diffusion_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        return sqrt_alphas_cumprod.to(self.device), sqrt_one_minus_alphas_cumprod.to(self.device)
    
    def _compute_true_class_mu(self, encoder, dataloader, class_mu, num_classes, feature_dim):
        class_features = [[] for _ in range(num_classes)]
        # 简单处理：直接遍历一次
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
                class_mu[cls_idx] = torch.mean(features_tensor, dim=0).detach()
        return class_mu
    
    def _compute_r_obs_from_real_features(self, encoder, dataloader, class_mu, num_classes):
        r_obs = torch.zeros(num_classes, device=self.device)
        class_counts = torch.zeros(num_classes, device=self.device)
        
        encoder.eval()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            real_features = encoder.forward_no_fc(inputs)
            for i in range(len(labels)):
                cls_idx = labels[i].item()
                if 0 <= cls_idx < num_classes:
                    feat = real_features[i].detach()
                    proto = class_mu[cls_idx]
                    dist = torch.norm(feat - proto, p=2)
                    r_obs[cls_idx] += dist
                    class_counts[cls_idx] += 1
        
        mask = class_counts > 0
        r_obs[mask] /= class_counts[mask]
        
        # 修改：使用有样本类别的平均半径作为无样本类别的默认值
        if mask.any():
            # 计算有样本类别的平均半径
            avg_radius = r_obs[mask].mean()
            r_obs[~mask] = avg_radius
            
        else:
            # 极端情况：所有类别都没有样本，使用原始默认值
            r_obs = self.config.target_radius * torch.ones(num_classes, device=self.device)
            
        return r_obs
    
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
        注意: 保持 classifier 在 train 模式以反映 Dropout 的实际效果
        """
        # Encoder 可以用 eval 模式（因为它没有 Dropout）
        encoder.eval()
        # Classifier 保持 train 模式，以反映 Dropout 的真实影响
        # 这样训练准确率会因为 Dropout 而低于 100%
        classifier.train()
        
        correct = 0
        total = 0
        
        
        with torch.no_grad():
            for inputs, labels in train_set:

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

    def _save_accuracy_history(self, avg_stage3_acc: float = None, avg_stage3_ls_acc: float = None):
        """
        保存准确率历史记录
        """
        if hasattr(self, 'encoder') and hasattr(self, 'classifier') and hasattr(self, 'diffusion_model'):
            # 使用模型管理器保存最佳模型
            self.monitor.save_best_checkpoints(self.encoder, self.classifier, self.diffusion_model, self.model_manager)
        
        # 输出最佳模型准确率信息
        self.monitor.log_training_complete(avg_stage3_acc, avg_stage3_ls_acc)
