"""
超参数搜索空间定义

定义 GALD-DC 项目中需要调优的超参数及其搜索范围
"""

import optuna
from typing import Dict, Any


# 搜索空间配置
SEARCH_SPACE_CONFIG = {
    # 判别边距参数
    'margin_m': {
        'type': 'float',
        'low': 1.0,
        'high': 5.0,
        'step': 0.5,
        'description': '判别边距距离 m'
    },
    
    # 分布校准参数
    'lambda_cal': {
        'type': 'float',
        'low': 0.1,
        'high': 0.7,
        'step': 0.1,
        'description': '尾部类半径校准混合因子 λ'
    },
    
    # 边距损失权重
    'eta_m': {
        'type': 'float',
        'low': 0.05,
        'high': 0.3,
        'step': 0.05,
        'description': '边距损失权重'
    },
    
    # 原型损失权重
    'eta_p': {
        'type': 'float',
        'low': 0.05,
        'high': 0.3,
        'step': 0.05,
        'description': '原型损失权重'
    },
    
    # 半径约束权重
    'eta_r': {
        'type': 'float',
        'low': 0.1,
        'high': 0.3,
        'step': 0.05,
        'description': '半径约束损失权重'
    },
    
    # 一致性损失权重 (仅 hybrid 模式)
    'beta_cons': {
        'type': 'float',
        'low': 0.05,
        'high': 0.3,
        'step': 0.05,
        'description': '一致性损失权重'
    },
    
    # 伪特征分类损失权重
    'gamma_pseudo': {
        'type': 'float',
        'low': 0.3,
        'high': 0.7,
        'step': 0.1,
        'description': '伪特征分类损失权重'
    },
    
    # EMA 更新率参数
    'lambda_ema': {
        'type': 'float',
        'low': 0.05,
        'high': 0.3,
        'step': 0.05,
        'description': '类别原型 EMA 更新率'
    },
    
    'beta_radius': {
        'type': 'float',
        'low': 0.05,
        'high': 0.4,
        'step': 0.05,
        'description': '类别半径 EMA 更新率'
    },
    
    # ==================== R2/R3/R4/R6 修复参数 ====================
    
    # R2: Hinge Loss 缓冲区
    'radius_slack': {
        'type': 'float',
        'low': 0.2,
        'high': 1.0,
        'step': 0.1,
        'description': 'Hinge Loss 半径缓冲区 δ (越大越宽松)'
    },
    
    # R3: EMA 冷启动保护期
    'ema_warmup_epochs': {
        'type': 'int',
        'low': 3,
        'high': 15,
        'step': 2,
        'description': 'Stage 2 中 EMA 冻结的 epoch 数'
    },
    
    # R4: Top-K 负类数量
    'margin_top_k': {
        'type': 'int',
        'low': 1,
        'high': 5,
        'step': 1,
        'description': 'Margin Loss 中使用的 Top-K 负类数量'
    },
    
    # R6: LoRA 秩
    'lora_rank': {
        'type': 'int',
        'low': 2,
        'high': 8,
        'step': 2,
        'description': 'LoRA 低秩适配的秩 (越小越轻量)'
    },
}


def define_search_space(trial: optuna.Trial, 
                        include_params: list = None,
                        exclude_params: list = None) -> Dict[str, Any]:
    """
    从 Optuna Trial 对象采样超参数
    
    Args:
        trial: Optuna Trial 对象
        include_params: 只包含的参数列表 (None = 全部)
        exclude_params: 排除的参数列表 (None = 不排除)
    
    Returns:
        Dict[str, Any]: 采样的超参数字典
    """
    params = {}
    
    for param_name, config in SEARCH_SPACE_CONFIG.items():
        # 过滤参数
        if include_params is not None and param_name not in include_params:
            continue
        if exclude_params is not None and param_name in exclude_params:
            continue
        
        param_type = config['type']
        
        if param_type == 'log_float':
            params[param_name] = trial.suggest_float(
                param_name, 
                config['low'], 
                config['high'], 
                log=True
            )
        elif param_type == 'float':
            if 'step' in config:
                params[param_name] = trial.suggest_float(
                    param_name, 
                    config['low'], 
                    config['high'], 
                    step=config['step']
                )
            else:
                params[param_name] = trial.suggest_float(
                    param_name, 
                    config['low'], 
                    config['high']
                )
        elif param_type == 'int':
            if 'step' in config:
                params[param_name] = trial.suggest_int(
                    param_name, 
                    config['low'], 
                    config['high'], 
                    step=config['step']
                )
            else:
                params[param_name] = trial.suggest_int(
                    param_name, 
                    config['low'], 
                    config['high']
                )
        elif param_type == 'categorical':
            params[param_name] = trial.suggest_categorical(
                param_name, 
                config['choices']
            )
    
    # 验证 stage epoch 的顺序约束
    if 'stage1_end_epoch' in params and 'stage2_end_epoch' in params:
        # 确保 stage2_end_epoch > stage1_end_epoch + 100
        if params['stage2_end_epoch'] <= params['stage1_end_epoch'] + 100:
            params['stage2_end_epoch'] = params['stage1_end_epoch'] + 150
    
    return params


def get_search_space_summary() -> str:
    """
    获取搜索空间的摘要信息
    
    Returns:
        str: 格式化的搜索空间摘要
    """
    lines = ["=" * 60, "超参数搜索空间", "=" * 60]
    
    for param_name, config in SEARCH_SPACE_CONFIG.items():
        param_type = config['type']
        
        if param_type == 'log_float':
            range_str = f"[{config['low']:.0e}, {config['high']:.0e}] (log scale)"
        elif param_type in ['float', 'int']:
            step_str = f", step={config.get('step', 'N/A')}" if 'step' in config else ""
            range_str = f"[{config['low']}, {config['high']}{step_str}]"
        elif param_type == 'categorical':
            range_str = f"{config['choices']}"
        else:
            range_str = "Unknown"
        
        lines.append(f"  {param_name}: {range_str}")
        lines.append(f"    └─ {config['description']}")
    
    lines.append("=" * 60)
    return "\n".join(lines)
