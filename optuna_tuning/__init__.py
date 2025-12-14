"""
Optuna 自动调参模块

用于 GALD-DC 项目的超参数自动优化

使用方法:
    python -m optuna_tuning.tuner --n_trials 50 --dataset CIFAR10 --imb_factor 0.01
"""

# 注意：不要在顶层导入以避免与 optuna 包的循环导入问题
# 使用时请直接导入:
#   from optuna_tuning.tuner import run_optuna_study
#   from optuna_tuning.search_space import define_search_space

__all__ = [
    'run_optuna_study',
    'create_study',
    'define_search_space',
    'SEARCH_SPACE_CONFIG',
    'OptunaStrategyATrainer'
]

__version__ = '1.0.0'
