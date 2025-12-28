from .trainer import GALDDCTrainer
from .config import TrainingConfig
from .model_manager import ModelManager, SimpleClassifier
from .loss_calculator import LossCalculator, WCDASLoss
from .training_monitor import TrainingMonitor

def train_gald_dc(args):
    """
    GALD-DC (Geometry-Aware Latent Diffusion with Distribution Calibration) entry point.
    """
    print(f"[GALD-DC] Starting training: {args.dataset}, imb_factor: {args.imb_factor}")
    print(f"[GALD-DC] lr: {args.lr}, lambda_sem: {args.lambda_sem}, gamma_ge: {args.gamma_ge}")
    
    config = TrainingConfig(args)
    config.log_config()
    
    trainer = GALDDCTrainer(config)
    trainer.train()
    
    print("[GALD-DC] Training complete")


__all__ = [
    'TrainingConfig',
    'ModelManager',
    'SimpleClassifier',
    'LossCalculator',
    'WCDASLoss',
    'TrainingMonitor',
    'GALDDCTrainer',
    'train_gald_dc'
]

__version__ = '1.0.0'