"""
LoRA (Low-Rank Adaptation) 适配器模块

R6 修复：允许扩散模型轻量级跟踪 Encoder 特征空间漂移

参考论文: LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)

核心思想:
- 冻结原始扩散模型权重
- 注入低秩矩阵 A, B 到关键层
- 只训练 A, B (~0.1% 参数量)
- 输出: W*x + (B @ A)*x * (alpha/rank)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class LoRALinear(nn.Module):
    """
    LoRA 线性层包装器
    
    将原始 Linear 层增强为: y = W*x + (B @ A)*x * (alpha/rank)
    其中 W 被冻结，A, B 可训练
    """
    
    def __init__(
        self, 
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # 冻结原始层
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA 低秩矩阵
        # A: [rank, in_features] - 降维
        # B: [out_features, rank] - 升维
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 初始化: A 用 Kaiming，B 用零 (确保初始时 LoRA 输出为 0)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
        # [修复] 确保新参数在正确的设备上 (cuda 或 cpu)
        device = original_layer.weight.device
        self.lora_A.data = self.lora_A.data.to(device)
        self.lora_B.data = self.lora_B.data.to(device)
        
        # Dropout (可选)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始层输出 (冻结权重)
        original_output = self.original_layer(x)
        
        # LoRA 增量: (B @ A) @ x.T -> x @ A.T @ B.T
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        
        return original_output + lora_output * self.scaling
    
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """返回 LoRA 可训练参数"""
        return [self.lora_A, self.lora_B]


class LoRAAdapter:
    """
    LoRA 适配器管理器
    
    用于将 LoRA 注入到扩散模型的指定层
    """
    
    def __init__(self, rank: int = 4, alpha: float = 8.0, dropout: float = 0.0):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.lora_layers: Dict[str, LoRALinear] = {}
    
    def inject_lora_into_model(
        self, 
        model: nn.Module,
        target_modules: Optional[List[str]] = None
    ) -> nn.Module:
        """
        将 LoRA 注入到模型的目标模块
        
        Args:
            model: 原始模型
            target_modules: 目标模块名称列表 (如 ['to_q', 'to_k', 'to_v', 'to_out'])
                          如果为 None，自动检测所有 Linear 层
        
        Returns:
            注入 LoRA 后的模型
        """
        if target_modules is None:
            # 默认目标: UNet 中的关键线性层
            # 针对 ddpm_conditional.py 中的 UNet_conditional 架构
            target_modules = [
                'time_mlp',      # 时间嵌入 MLP (nn.Linear)
                'fc_256in',      # 输入降维层 (如果存在)
                'final_out256',  # 输出升维层 (如果存在)
                'mlp',           # 通用 MLP 层
                '1',             # 匹配 Sequential 中的 Linear 层 (如 time_mlp.1, time_mlp.3)
                '3'
            ]
        
        modules_to_replace = []
        
        # 遍历模型找到目标模块
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # 检查是否匹配目标模块名
                if any(target in name for target in target_modules):
                    modules_to_replace.append((name, module))
        
        # 替换目标模块为 LoRA 版本
        for name, module in modules_to_replace:
            lora_layer = LoRALinear(
                module, 
                rank=self.rank, 
                alpha=self.alpha,
                dropout=self.dropout
            )
            self.lora_layers[name] = lora_layer
            
            # 通过父模块替换
            self._replace_module(model, name, lora_layer)
        
        return model
    
    def _replace_module(self, model: nn.Module, target_name: str, new_module: nn.Module):
        """替换模型中的指定模块"""
        parts = target_name.split('.')
        parent = model
        
        # 导航到父模块
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        # 替换目标模块
        final_name = parts[-1]
        if final_name.isdigit():
            parent[int(final_name)] = new_module
        else:
            setattr(parent, final_name, new_module)
    
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """获取所有 LoRA 可训练参数"""
        params = []
        for lora_layer in self.lora_layers.values():
            params.extend(lora_layer.get_lora_parameters())
        return params
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """获取 LoRA 参数的 state dict"""
        state_dict = {}
        for name, lora_layer in self.lora_layers.items():
            state_dict[f'{name}.lora_A'] = lora_layer.lora_A
            state_dict[f'{name}.lora_B'] = lora_layer.lora_B
        return state_dict
    
    def count_parameters(self) -> Dict[str, int]:
        """统计参数量"""
        lora_params = sum(p.numel() for p in self.get_lora_parameters())
        return {
            'lora_parameters': lora_params,
            'lora_layers': len(self.lora_layers)
        }


def apply_lora_to_diffusion_model(
    diffusion_model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0
) -> tuple:
    """
    便捷函数：将 LoRA 应用到扩散模型
    
    Args:
        diffusion_model: GaussianDiffusion 模型
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        dropout: Dropout 率
    
    Returns:
        (adapter, lora_params) - LoRA 适配器和可训练参数列表
    """
    adapter = LoRAAdapter(rank=rank, alpha=alpha, dropout=dropout)
    
    # 注入到扩散模型的内部网络 (model.model 通常是 UNet 或 MLP)
    if hasattr(diffusion_model, 'model'):
        adapter.inject_lora_into_model(diffusion_model.model)
    else:
        adapter.inject_lora_into_model(diffusion_model)
    
    lora_params = adapter.get_lora_parameters()
    param_count = adapter.count_parameters()
    
    print(f">>> [LoRA] Injected {param_count['lora_layers']} layers, "
          f"{param_count['lora_parameters']:,} trainable parameters")
    
    return adapter, lora_params
