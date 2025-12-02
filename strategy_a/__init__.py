from .trainer import StrategyATrainer
from .config import TrainingConfig
from .model_manager import ModelManager, SimpleClassifier
from .loss_calculator import LossCalculator, WCDASLoss
from .training_monitor import TrainingMonitor

def train_strategy_a(args):
    """
    训练策略A的主入口函数，集成WCDAS准确率计算功能
    """
    print(f"Start training strategy A: {args.dataset}, imbalance factor: {args.imb_factor}")
    print(f"Learning rate: {args.lr}, lambda_sem: {args.lambda_sem}, gamma_ge: {args.gamma_ge}")
    
    # 创建配置
    config = TrainingConfig(args)
    config.log_config()
    
    # 创建训练器并开始训练
    trainer = StrategyATrainer(config)
    trainer.train()
    
    print("Training Strategy A finished")


__all__ = [
    'TrainingConfig',
    'ModelManager',
    'SimpleClassifier',
    'LossCalculator',
    'WCDASLoss',
    'TrainingMonitor',
    'StrategyATrainer',
    'train_strategy_a'
]

__version__ = '1.0.0'