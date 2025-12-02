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
        
        # 5. 主训练循环
        for epoch in range(self.config.epochs):
            loss_weights = self._get_dynamic_loss_weights(epoch)
            
            # 训练一个 Epoch
            self._train_epoch(epoch, encoder, classifier, diffusion_model, optimizer,
                            train_set, class_prototypes, target_radii,
                            sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                            num_classes, feature_dim, loss_weights, train_class_counts)
            
            # 验证 (传入 train_class_counts 用于 WCDAS 修正)
            test_loss, accuracy, label_shift_acc, mmf_acc, mmf_acc_pc = \
                self._validate(encoder, classifier, test_set, dataset_info, train_class_counts)
            
            self.monitor.log_validation(epoch, accuracy, test_loss, label_shift_acc, mmf_acc, mmf_acc_pc)
            
            # 保存最佳CE/WCDAS准确率模型
            if accuracy > self.monitor.best_accuracy:
                self._save_checkpoint(epoch, encoder, classifier, diffusion_model, optimizer, accuracy, 'ce')
            
            # 保存最佳Label Shift准确率模型
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
    
    def _train_epoch(self, epoch: int, encoder: nn.Module, classifier: nn.Module, 
                    diffusion_model: nn.Module, optimizer: optim.Optimizer,
                    train_set: DataLoader, 
                    class_prototypes: Dict[int, torch.Tensor],
                    target_radii: torch.Tensor,
                    sqrt_alphas_cumprod: torch.Tensor, 
                    sqrt_one_minus_alphas_cumprod: torch.Tensor,
                    num_classes: int, feature_dim: int, loss_weights: Dict[str, float],
                    train_class_counts: torch.Tensor):
        
        # [Warmup] 前50个epoch冻结Encoder，防止扩散模型崩溃
        if epoch < 100:
            encoder.eval()
            for param in encoder.parameters(): param.requires_grad = False
        else:
            encoder.train()
            for param in encoder.parameters(): param.requires_grad = True
        
        classifier.train()
        diffusion_model.train()
        
        running_losses = {'real': 0.0, 'semantic': 0.0, 'gen': 0.0, 'total': 0.0}
        total_loss = 0.0 
        
        for batch_idx, (inputs, labels) in enumerate(train_set):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward
            if epoch < 100:
                with torch.no_grad():
                    real_features = encoder.forward_no_fc(inputs)
            else:
                real_features = encoder.forward_no_fc(inputs)
            
            # 计算损失
            losses = self._compute_batch_losses(
                encoder, classifier, diffusion_model, 
                real_features, inputs, labels,
                class_prototypes, target_radii, sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod, num_classes, feature_dim, epoch, batch_idx, loss_weights,
                train_class_counts 
            )
            
            losses['total'].backward()
            self._control_gradients(encoder, classifier, diffusion_model, epoch * len(train_set), batch_idx)
            optimizer.step()
            optimizer.zero_grad()

            #⬇️试试动态半径和固定半径的效果
            # [动态更新] 在 Encoder 解冻后，使用 EMA 更新类别原型和目标半径
            # 只在 Epoch >= 50 时更新，因为 Warmup 期 Encoder 冻结，特征空间不变
            if epoch >= 50:
                self._update_stats_ema(real_features.detach(), labels, class_prototypes, target_radii, num_classes)
            # ⬆️
            for key in running_losses:
                running_losses[key] += losses[key].item()
            total_loss += losses['total'].item()
            # 日志记录
            if batch_idx % 100 == 0:
                self.monitor.log_batch_progress(epoch, batch_idx, {k: v.item() for k, v in losses.items()})
        
        train_loss = total_loss / len(train_set)
        # 计算训练集准确率
        train_accuracy = self._compute_train_accuracy(encoder, classifier, train_set, train_class_counts, num_classes) 
        
        avg_losses = {key: value / len(train_set) for key, value in running_losses.items()}
        self.monitor.log_epoch_summary(epoch, avg_losses, train_accuracy, train_loss)
    
    def _compute_batch_losses(self, encoder, classifier, diffusion_model, 
                             real_features: torch.Tensor, inputs: torch.Tensor, 
                             labels: torch.Tensor, class_prototypes: Dict[int, torch.Tensor],
                             target_radii: torch.Tensor, 
                             sqrt_alphas_cumprod: torch.Tensor,
                             sqrt_one_minus_alphas_cumprod: torch.Tensor,
                             num_classes: int, feature_dim: int, 
                             epoch: int, batch_idx: int, loss_weights: Dict[str, float],
                             train_class_counts: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        batch_size = inputs.size(0)
        
        # [核心 1] 真实分类损失
        if self.config.use_wcdas:
            # WCDAS 模式: 传入真实样本分布，用于加权训练
            real_loss = self.loss_calculator.compute_real_loss(
                classifier, real_features, labels, train_class_counts, 'wcdas'
            )
        else:
            # CE 模式
            real_loss = self.loss_calculator.compute_real_loss(classifier, real_features, labels)
        
        # [核心 2] 装备效果 (LDMLR 更新 & 几何约束)
        self._update_stats_ema(real_features, labels, class_prototypes, target_radii, num_classes)
        
        diffusion_loss = self.loss_calculator.compute_diffusion_loss(
            diffusion_model, real_features, labels, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
        )
        
        estimated_clean = self._estimate_clean_features(
            diffusion_model, real_features, labels, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
        )
        
        # 几何 Loss (Prototype + Radius)
        prototype_loss = self.loss_calculator.compute_prototype_loss(
            estimated_clean, labels, class_prototypes, num_classes
        )
        # 注意：允许几何约束优化 Diffusion Model
        radius_loss = self.loss_calculator.compute_radius_constraint_loss(
            estimated_clean, labels, class_prototypes, target_radii, num_classes
        )
        
        # 数值保护
        diffusion_loss = self._safe_loss(diffusion_loss, 5.0)
        prototype_loss = self._safe_loss(prototype_loss, 10.0)
        radius_loss = self._safe_loss(radius_loss, 10.0)
        
        semantic_loss = diffusion_loss + self.config.eta_p * prototype_loss + self.config.eta_r * radius_loss
        semantic_loss = self._safe_loss(semantic_loss, self.config.max_semantic_loss)
        
        # [核心 3] 生成损失 (课程学习)
        # 前 7.5% Epoch 不使用生成数据训练，防止早期模型质量差导致的"喂毒"
        if epoch < (self.config.epochs * 0.075):
             gen_loss = torch.tensor(0.0, device=self.device)
        else:
            gen_loss = self._compute_generation_loss(
                diffusion_model, classifier, batch_size, feature_dim, 
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, num_classes, batch_idx
            )
        gen_loss = self._safe_loss(gen_loss, self.config.max_gen_loss)
        
        # 总损失
        total_loss = self.loss_calculator.compute_total_loss(real_loss, semantic_loss, gen_loss, loss_weights)
        
        return {
            'real': real_loss,
            'semantic': semantic_loss,
            'gen': gen_loss,
            'total': total_loss
        }

    def _compute_generation_loss(self, diffusion_model, classifier, batch_size, feature_dim, 
                               sqrt_alphas, sqrt_one_minus_alphas, num_classes, batch_idx):
        if batch_idx % self.config.generation_interval != 0:
            return torch.tensor(0.0, device=self.device)
        
        time_steps = torch.linspace(self.config.diffusion_steps-1, 0, 
                                  self.config.ddim_steps+1, dtype=torch.long, device=self.device)
        # 生成平衡的 Batch
        fake_features = torch.randn(batch_size, feature_dim, device=self.device)
        fake_labels = torch.randint(0, num_classes, (batch_size,), device=self.device)
        
        fake_features = self._ddim_sample(diffusion_model, fake_features, fake_labels, 
                                        time_steps, sqrt_alphas, sqrt_one_minus_alphas)
        
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
                
                # 更新半径
                dists = torch.norm(cls_feats - new_proto, p=2, dim=1)
                current_avg_radius = dists.mean()
                old_radius = target_radii[cls_idx]
                new_radius = (1 - self.config.lambda_ema) * old_radius + self.config.lambda_ema * current_avg_radius
                target_radii[cls_idx] = new_radius

    def _ddim_sample(self, diffusion_model, fake_features, fake_labels, time_steps, sqrt_alphas, sqrt_one_minus_alphas):
        batch_size = fake_features.size(0)
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
        return fake_features

    def _estimate_clean_features(self, diffusion_model, real_features, labels, sqrt_alphas, sqrt_one_minus_alphas):
        batch_size = real_features.size(0)
        t = torch.randint(0, self.config.diffusion_steps, (batch_size,), device=self.device)
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