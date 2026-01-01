from cmath import tau
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

from .config import TrainingConfig


class WCDASLoss(nn.Module):
    """
    WCDAS (Wrapped Cauchy Distributed Angular Softmax) 损失函数
    
    基于Wrapped Cauchy分布的角 Softmax，用于处理长尾分布问题
    """
    
    def __init__(self, in_features, out_features, bias=False, gamma=-1, s_trainable=True):
        """
        初始化WCDAS损失函数
        
        Args:
            in_features: 输入特征维度
            out_features: 输出类别数
            bias: 是否使用偏置
            gamma: 初始gamma参数（浓度参数）
            s_trainable: 是否训练缩放参数s
        """
        super(WCDASLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 权重参数（类别原型）
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # 缩放参数s
        self.s_ = nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        
        # 浓度参数gamma（每个类别一个）
        self.g = nn.Parameter(data=gamma * torch.ones(out_features), requires_grad=True)
        
        self.s_trainable = s_trainable
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            nn.init.constant_(self.bias, 0.0)
        else:
            self.register_parameter('bias', None)
    
    def loss(self, logits, target, sample_per_class):
        """计算交叉熵损失"""
        return F.cross_entropy(logits, target, weight=None, ignore_index=-100, reduction='mean')
    
    def forward(self, input, target, sample_per_class):
        """
        前向传播
        
        Args:
            input: 输入特征 [batch_size, in_features]
            target: 目标标签 [batch_size]
            sample_per_class: 每个类别的样本数 [num_classes]
            
        Returns:
            logits: WCDAS logits
            loss: 损失值
            gamma: 浓度参数
            scale: 缩放参数
        """
        assert target is not None
        
        # 计算gamma (浓度参数)，使用sigmoid确保在(0,1)范围内
        self.gamma = 1 / (1 + torch.exp(-self.g))
        
        # 计算余弦相似度
        cosine = F.linear(
            F.normalize(input, p=2, dim=1), 
            F.normalize(self.weight, p=2, dim=1), 
            self.bias
        )
        
        # WCDAS核心公式:
        # logit = (1/(2π)) * ((1-γ²)/(1+γ²-2γ*cos(θ)))
        # 其中cos(θ)就是余弦相似度
        logit = 1/2/3.14 * (1-self.gamma**2) / (1+self.gamma**2-2*self.gamma*cosine)
        
        # 计算缩放参数s
        s = F.softplus(self.s_).add(1.0)
        
        # 应用缩放参数
        if self.s_trainable:
            logit = s * logit
        else:
            logit = 250 * logit
            
        # 计算损失
        l = self.loss(logit, target, sample_per_class)
        
        return logit, l, self.gamma, s
            


class LossCalculator:
    """
    损失计算器，统一管理所有损失计算
    """
    
    def __init__(self, config: TrainingConfig):
        """
        初始化损失计算器
        """
        self.config = config
    
    def compute_real_loss(self, classifier: nn.Module, real_features: torch.Tensor, 
                         labels: torch.Tensor, class_sample_counts: torch.Tensor = None,
                         loss_type: str = 'ce') -> torch.Tensor:
        """
        计算真实分类损失，支持多种损失函数
        
        Args:
            classifier: 分类头模型
            real_features: 输入特征
            labels: 标签
            class_sample_counts: 每个类别的样本数（用于WCDAS）
            loss_type: 损失函数类型 ('ce', 'wcdas')
        """
        if loss_type == 'ce':
            # 标准交叉熵损失 + Label Smoothing (减少过拟合)
            real_logits = classifier(real_features)
            return F.cross_entropy(real_logits, labels, label_smoothing=0)
        elif loss_type == 'wcdas':
            # WCDAS损失函数，需要class_sample_counts
            if class_sample_counts is None:
                raise ValueError("WCDAS损失需要class_sample_counts参数")
            
            # 检查classifier是否为WCDASLoss类型
            if isinstance(classifier, WCDASLoss):
                # 使用WCDAS前向传播
                logits, loss, gamma, scale = classifier(
                    input=real_features,
                    target=labels,
                    sample_per_class=class_sample_counts.float()
                )
                return loss
            elif hasattr(classifier, 'forward'):
                # 如果classifier有forward方法，但可能不是WCDASLoss，尝试使用标准CE
                real_logits = classifier(real_features)
                return F.cross_entropy(real_logits, labels)
            else:
                # 其他情况使用标准CE
                real_logits = classifier(real_features)
                return F.cross_entropy(real_logits, labels)
        else:
            # 默认使用CE
            real_logits = classifier(real_features)
            return F.cross_entropy(real_logits, labels)
    
    def compute_wcdas_real_loss(self, wcdas_classifier: nn.Module, real_features: torch.Tensor, 
                               labels: torch.Tensor, class_sample_counts: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        计算WCDAS真实分类损失
        
        Args:
            wcdas_classifier: WCDAS分类器
            real_features: 输入特征 [batch_size, feature_dim]
            labels: 真实标签 [batch_size]
            class_sample_counts: 每个类别的样本数 [num_classes]
            
        Returns:
            loss: WCDAS损失
            metrics: 包含logits、gamma、scale等信息的字典
        """
        # 确保输入是正确的设备
        real_features = real_features.to(wcdas_classifier.weight.device)
        labels = labels.to(wcdas_classifier.weight.device)
        class_sample_counts = class_sample_counts.to(wcdas_classifier.weight.device)
        
        # 计算WCDAS损失
        logits, loss, gamma, scale = wcdas_classifier(
            input=real_features,
            target=labels,
            sample_per_class=class_sample_counts.float()
        )
        
        # 返回损失和相关信息
        metrics = {
            'logits': logits,
            'gamma': gamma,
            'scale': scale,
            'preds': torch.argmax(logits, dim=1)
        }
        
        return loss, metrics
    
    def compute_wcdas_accuracy(self, wcdas_classifier: nn.Module, features: torch.Tensor, 
                              labels: torch.Tensor, class_sample_counts: torch.Tensor, 
                              num_classes: int) -> Tuple[float, Dict]:
        """
        计算WCDAS的准确率
        
        Args:
            wcdas_classifier: WCDAS分类器
            features: 输入特征 [batch_size, feature_dim]
            labels: 真实标签 [batch_size]
            class_sample_counts: 每个类别的样本数 [num_classes]
            num_classes: 类别数量
            
        Returns:
            accuracy: 准确率
            details: 详细的准确率信息
        """
        with torch.no_grad():
            # 计算预测
            loss, metrics = self.compute_wcdas_real_loss(
                wcdas_classifier, features, labels, class_sample_counts
            )
            preds = metrics['preds']
            
            # 计算整体准确率
            correct = (preds == labels).sum().item()
            total = labels.size(0)
            overall_acc = correct / total
            
            # 计算每个类别的准确率
            class_correct = torch.zeros(num_classes)
            class_total = torch.zeros(num_classes)
            
            for i in range(total):
                label = labels[i].item()
                if 0 <= label < num_classes:
                    class_total[label] += 1
                    if preds[i] == labels[i]:
                        class_correct[label] += 1
            
            class_acc = {}
            for i in range(num_classes):
                if class_total[i] > 0:
                    class_acc[i] = (class_correct[i] / class_total[i]).item()
                else:
                    class_acc[i] = 0.0
            
            # 计算多数类和少数类的准确率
            # [修复] 针对 CIFAR-10 和 CIFAR-100 使用不同的阈值
            if num_classes <= 10:
                many_shot_thr = 100
                medium_shot_thr = 60  # 提高阈值，让 50 样本的类归为 Few Shot
            else:
                many_shot_thr = 100
                medium_shot_thr = 20
            
            many_shot_acc = []
            medium_shot_acc = []
            few_shot_acc = []
            
            for i in range(num_classes):
                if class_sample_counts[i] > many_shot_thr:
                    many_shot_acc.append(class_acc[i])
                elif class_sample_counts[i] > medium_shot_thr:
                    medium_shot_acc.append(class_acc[i])
                else:
                    few_shot_acc.append(class_acc[i])
            
            details = {
                'overall_accuracy': overall_acc,
                'class_accuracy': class_acc,
                'many_shot_accuracy': np.mean(many_shot_acc) if many_shot_acc else 0.0,
                'medium_shot_accuracy': np.mean(medium_shot_acc) if medium_shot_acc else 0.0,
                'few_shot_accuracy': np.mean(few_shot_acc) if few_shot_acc else 0.0,
                'class_counts': class_sample_counts.tolist()
            }
            
            return overall_acc, details
    
    def compute_diffusion_loss(self, diffusion_model: nn.Module, 
                              real_features: torch.Tensor, labels: torch.Tensor,
                              sqrt_alphas_cumprod: torch.Tensor, 
                              sqrt_one_minus_alphas_cumprod: torch.Tensor) -> torch.Tensor:
        """
        计算扩散损失 (修复版)
        
        修复内容:
        1. 移除外部加噪逻辑，直接传入纯净特征
        2. 统一在 _manual_diffusion_loss 内部处理加噪
        """
        # [修复] 直接传入纯净的 real_features
        # 之前在此处加噪导致后续 _manual_diffusion_loss 重复加噪
        
        # 调整特征形状，确保是3D (batch_size, 1, feature_dim)
        real_features_reshaped = real_features.unsqueeze(1)
        
        return self._manual_diffusion_loss(diffusion_model, real_features_reshaped, None, labels)
    
    def _manual_diffusion_loss(self, diffusion_model: nn.Module, 
                              x_start: torch.Tensor, t: torch.Tensor, 
                              labels: torch.Tensor) -> torch.Tensor:
        """
        扩散损失计算 (修复版)
        
        修复内容:
        1. 移除归一化逻辑，确保训练和推理分布一致
        2. 统一在此处生成 t 和 noise
        """


        b, c, n = x_start.shape
        device = x_start.device
        
        
        # 1. 生成时间步 t (如果未传入)
        if t is None:
            t = torch.randint(0, self.config.diffusion_steps, (b,), device=device)
            
        # 2. 生成噪声
        noise = torch.randn_like(x_start)
        
        # 3. 前向加噪 (q_sample) - 只加一次
        # 使用 diffusion_model 内部的 q_sample，它会正确使用 sqrt_alphas_cumprod
        x_noisy = diffusion_model.q_sample(x_start=x_start, t=t, noise=noise)
        
        # 处理 NaN
        if torch.isnan(x_noisy).any():
            x_noisy = torch.nan_to_num(x_noisy, nan=0.0)
        
        # 4. 预测目标
        if diffusion_model.objective == 'pred_noise':
            target = noise
        elif diffusion_model.objective == 'pred_x0':
            target = x_start  # [修复2] 目标是原始的 x_start，而非归一化后的
        else:
            raise ValueError(f'Unknown objective: {diffusion_model.objective}')
        
        # 5. 模型预测
        model_output = diffusion_model.model(x_noisy, t, labels)
        
        # 6. 计算损失
        loss = F.mse_loss(model_output, target, reduction='none')
        loss = loss.mean(dim=[1, 2])
        
        # 应用损失权重
        loss_weight = diffusion_model.loss_weight[t]
        loss_weight = torch.clamp(loss_weight, max=self.config.max_loss_weight)
        
        diffusion_loss = (loss * loss_weight).mean()
        diffusion_loss = torch.clamp(diffusion_loss, max=self.config.max_diffusion_loss)
        
        return diffusion_loss
    
    def compute_prototype_loss(self, estimated_clean_features: torch.Tensor, 
                             labels: torch.Tensor, 
                             class_mu: Dict[int, torch.Tensor],
                             num_classes: int) -> torch.Tensor:
        """
        计算原型拉拢损失（增强数值稳定性）
        """
        batch_size = estimated_clean_features.size(0)
        device = estimated_clean_features.device
        
        prototype_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        # 检查估计特征的数值稳定性
        if torch.isnan(estimated_clean_features).any() or torch.isinf(estimated_clean_features).any():
            print("Warning: Estimated clean features contain NaN or Inf values, skipping prototype loss")
            return torch.tensor(0.0, device=device)
        
        # 裁剪估计特征，防止数值爆炸
        estimated_clean_features = torch.clamp(estimated_clean_features, 
                                             min=-self.config.feature_clamp_max,
                                             max=self.config.feature_clamp_max)
        
        for i in range(batch_size):
            cls_idx = labels[i].item()
            if 0 <= cls_idx < num_classes:
                # 检查原型中心的数值稳定性
                if torch.isnan(class_mu[cls_idx]).any() or torch.isinf(class_mu[cls_idx]).any():
                    continue
                
                # 裁剪原型中心，防止数值爆炸
                proto = torch.clamp(class_mu[cls_idx].detach(),
                                              min=-self.config.feature_clamp_max,
                                              max=self.config.feature_clamp_max)
                
                loss_item = F.mse_loss(estimated_clean_features[i], proto)
                
                # 检查损失项的数值稳定性
                if not torch.isnan(loss_item) and not torch.isinf(loss_item):
                    # 限制单个样本的损失值
                    loss_item = torch.clamp(loss_item, max=10.0)
                    prototype_loss += loss_item
                    valid_samples += 1
        
        prototype_loss = prototype_loss / valid_samples if valid_samples > 0 else torch.tensor(0.0, device=device) #计算平均损失，并防止除以零的错误。
        
        # 限制原型损失的最大值
        prototype_loss = torch.clamp(prototype_loss, max=5.0)
        
        return prototype_loss
    
    def compute_radius_constraint_loss(self, estimated_clean_features: torch.Tensor, 
                                     labels: torch.Tensor, 
                                     class_mu: Dict[int, torch.Tensor],
                                     r_obs: List[torch.Tensor],
                                     num_classes: int) -> torch.Tensor:
        """
        计算等半径约束损失（替换协方差匹配）
        
        根据等半径约束公式：
        L_rad = E[(||~z_0 - μ_y||_2 - r_y)^2]
        
        其中：
        - ~z_0：估计的特征（Estimated Feature）
        - μ_y：类别中心（Class Prototype）
        - r_y：目标半径（Target Radius），即真实样本到类中心的平均距离
        """
        device = estimated_clean_features.device
        radius_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        # 首先检查估计特征的数值稳定性
        if torch.isnan(estimated_clean_features).any() or torch.isinf(estimated_clean_features).any():
            print("Warning: Estimated clean features contain NaN or Inf values, skipping radius constraint loss")
            return torch.tensor(0.0, device=device)
        
        # 裁剪估计特征，防止数值爆炸
        estimated_clean_features = torch.clamp(estimated_clean_features, 
                                             min=-self.config.feature_clamp_max,
                                             max=self.config.feature_clamp_max)
        
        # 对每个样本计算损失
        batch_size = estimated_clean_features.size(0)
        for i in range(batch_size):
            cls_idx = labels[i].item()
            if 0 <= cls_idx < num_classes:
                # 获取当前特征和对应的类中心
                feature = estimated_clean_features[i]
                proto = class_mu[cls_idx]
                
                # 检查特征数值稳定性
                if torch.isnan(feature).any() or torch.isinf(feature).any():
                    continue
                
                # 检查原型中心的数值稳定性
                if torch.isnan(proto).any() or torch.isinf(proto).any():
                    continue
                
                # 计算特征到类中心的欧几里得距离 ||~z_0 - μ_y||_2
                distance = torch.norm(feature - proto, p=2)
                
                # 获取目标半径 r_y
                if cls_idx < len(r_obs):
                    target_radius = r_obs[cls_idx]
                    # 确保目标半径在正确的设备上
                    if target_radius.device != device:
                        target_radius = target_radius.to(device)
                else:
                    # 如果没有预设目标，计算有样本类别的平均半径作为默认值
                    if torch.any(r_obs > 0):  # 有实际计算的半径值
                        mask = r_obs > 0
                        avg_radius = r_obs[mask].mean()
                        target_radius = avg_radius.to(device)
                    else:
                        # 极端情况：所有类别都没有预设半径，使用配置中的默认值
                        target_radius = torch.tensor(self.config.target_radius, device=device)
                
                # [R2 修复] 使用 Hinge Loss 替代 MSE，允许缓冲区
                # 原公式: L_rad = (distance - r)²  (过于严格，损害各向异性)
                # 新公式: L_rad = max(0, |distance - r| - δ)  (允许 [r-δ, r+δ] 范围)
                delta = getattr(self.config, 'radius_slack', 0.5)  # 缓冲区宽度
                loss_item = F.relu(torch.abs(distance - target_radius) - delta)
                
                # 检查损失项的数值稳定性
                if not torch.isnan(loss_item) and not torch.isinf(loss_item):
                    radius_loss += loss_item
                    valid_samples += 1
        
        # 计算平均损失 E[(||~z_0 - μ_y||_2 - r_y)^2]
        radius_loss = radius_loss / valid_samples if valid_samples > 0 else torch.tensor(0.0, device=device) #计算平均损失，并防止除以零的错误。
        radius_loss = torch.clamp(radius_loss, max=self.config.max_radius_loss)
        
        return radius_loss
    
    def compute_total_loss(self, real_loss: torch.Tensor, semantic_loss: torch.Tensor,
                          gen_loss: torch.Tensor, loss_weights: Dict[str, float]) -> torch.Tensor:
        """
        计算总损失
        """
        lambda_sem = loss_weights['lambda_sem']
        gamma_ge = loss_weights['gamma_ge']
        
        total_loss = real_loss + lambda_sem * semantic_loss + gamma_ge * gen_loss
        
        # 检查总损失是否为NaN或Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: Total loss is NaN or Inf")
            print(f"Real loss: {real_loss.item()}, Semantic loss: {semantic_loss.item()}, Gen loss: {gen_loss.item()}")
            # 如果总损失异常，使用真实损失作为替代
            total_loss = real_loss
        
        return total_loss
    
    # ==================== GALD-DC 增强功能 ====================
    
    def compute_margin_loss(self, estimated_clean: torch.Tensor, labels: torch.Tensor,
                           class_mu: Dict[int, torch.Tensor], 
                           num_classes: int, margin: float) -> torch.Tensor:
        """
        计算判别边距约束损失 (GALD-DC Section 2.6)
        
        公式: L_margin = E[(m + ||z0 - μ_y||² - ||z0 - μ_y-||²)]+
        
        作用: 将样本推离最近的负类原型，形成至少 margin m 的隔离带
        """
        device = estimated_clean.device
        batch_size = estimated_clean.size(0)
        
        # 数值稳定性检查
        if torch.isnan(estimated_clean).any() or torch.isinf(estimated_clean).any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 裁剪特征防止数值爆炸
        estimated_clean = torch.clamp(estimated_clean, 
                                     min=-self.config.feature_clamp_max,
                                     max=self.config.feature_clamp_max)
        
        # 构建原型矩阵 [num_classes, feature_dim]
        prototype_matrix = torch.zeros(num_classes, estimated_clean.size(1), device=device)
        valid_prototypes = torch.zeros(num_classes, dtype=torch.bool, device=device)
        
        for cls_idx in range(num_classes):
            if cls_idx in class_mu:
                proto = class_mu[cls_idx]
                if not (torch.isnan(proto).any() or torch.isinf(proto).any()):
                    prototype_matrix[cls_idx] = proto.to(device)
                    valid_prototypes[cls_idx] = True
        
        if valid_prototypes.sum() < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 向量化计算所有样本到所有原型的平方距离 [batch_size, num_classes]
        # 使用 estimated_clean 直接计算，保持梯度流
        # 使用平方距离 ||z - μ||²，与文档公式一致
        dists = torch.cdist(estimated_clean, prototype_matrix, p=2) ** 2
        
        # 收集有效样本的损失
        margin_losses = []
        
        for i in range(batch_size):
            cls_idx = labels[i].item()
            if not (0 <= cls_idx < num_classes) or not valid_prototypes[cls_idx]:
                continue
            
            # 到本类原型的距离
            dist_to_pos = dists[i, cls_idx]
            
            # 找最近的负类原型距离
            neg_mask = valid_prototypes.clone()
            neg_mask[cls_idx] = False
            if neg_mask.sum() == 0:
                continue
            
            neg_dists = dists[i, neg_mask]
            
            # [R4 修复] Top-K Soft Margin: 对 K 个最近负类取平均，平滑梯度
            # 原问题: 只用单个最近负类导致 y⁻ 频繁切换，梯度震荡
            top_k = getattr(self.config, 'margin_top_k', 3)
            actual_k = min(top_k, neg_dists.size(0))  # 防止 K 超过可用负类数
            
            # 获取 Top-K 最近负类距离 (smallest K distances)
            top_k_neg_dists, _ = neg_dists.topk(actual_k, largest=False)
            
            # 对每个负类计算 margin loss 并取平均
            loss_items = F.relu(margin + dist_to_pos - top_k_neg_dists)
            loss_item = loss_items.mean()
            
            if not (torch.isnan(loss_item) or torch.isinf(loss_item)):
                margin_losses.append(loss_item)
        
        if len(margin_losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        margin_loss = torch.stack(margin_losses).mean()
        margin_loss = torch.clamp(margin_loss, max=10.0)  # 数值稳定性
        
        return margin_loss
    
    def compute_calibrated_radius(self, observed_radii: torch.Tensor, 
                                  r_prior: float, class_counts: torch.Tensor,
                                  tau: int, lambda_cal: float) -> torch.Tensor:
        """
        计算校准半径 (GALD-DC Section 2.4)
        
        公式:
            r_cal_y = r_obs_y                          (若 y ∈ C_head, 样本数 >= tau)
            r_cal_y = λ·r_obs_y + (1-λ)·r_prior        (若 y ∈ C_tail, 样本数 < tau)
        
        作用: 防止尾部类因样本少而半径塌缩(r_obs -> 0)，用头部类先验撑开尾部类分布
        
        Args:
            observed_radii: 观测半径 [num_classes]
            r_prior: 头部类全局半径先验 (头部类平均半径)
            class_counts: 每个类别的样本数 [num_classes]
            tau: 头部/尾部类别阈值
            lambda_cal: 校准混合因子
            
        Returns:
            r_cal: 校准后的半径 [num_classes]
        """
        device = observed_radii.device
        num_classes = observed_radii.size(0)
        r_cal = observed_radii.clone()
        
        # 识别头部类和尾部类
        head_mask = class_counts >= tau  # C_head
        tail_mask = class_counts < tau   # C_tail
        
        # 头部类: 直接使用观测半径
        # 尾部类: 混合校准
        r_cal[tail_mask] = (
            lambda_cal * observed_radii[tail_mask] + 
            (1 - lambda_cal) * r_prior
        )
        
        return r_cal
    
    def compute_head_class_prior(self, observed_radii: torch.Tensor, 
                                 class_counts: torch.Tensor, tau: int) -> float:
        """
        计算头部类全局半径先验 r_prior
        
        Args:
            observed_radii: 观测半径 [num_classes]
            class_counts: 每个类别的样本数 [num_classes]
            tau: 头部/尾部类别阈值
            
        Returns:
            r_prior: 头部类平均半径
        """
        head_mask = class_counts >= tau
        
        if head_mask.sum() == 0:
            # 如果没有头部类，返回所有类的平均半径
            return observed_radii.mean().item()
        
        r_prior = observed_radii[head_mask].mean().item()
        return r_prior
    
    def compute_consistency_loss(self, current_features: torch.Tensor, 
                                frozen_features: torch.Tensor) -> torch.Tensor:
        """
        计算特征一致性损失 (GALD-DC Stage 3-H Section 3.B.2)
        
        公式: L_cons = E[||E^(t)(x) - detach(E^(0)(x))||²]
        
        作用: 约束当前编码器特征不要偏离冻结编码器特征太远，
              保证扩散模型学到的分布与当前特征空间对齐
        
        Args:
            current_features: 当前编码器输出 E^(t)(x)
            frozen_features: 冻结编码器输出 E^(0)(x)
            
        Returns:
            consistency_loss: 一致性损失
        """
        # detach 冻结特征，梯度只流向当前编码器
        return F.mse_loss(current_features, frozen_features.detach())
