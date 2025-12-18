"""
Optuna 自动调参主入口

使用方法:
    python -m optuna_tuning.tuner --n_trials 50 --dataset CIFAR10 --imb_factor 0.01
    
断点续跑:
    python -m optuna_tuning.tuner --n_trials 50 --resume
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from optuna_tuning.search_space import define_search_space, get_search_space_summary
from optuna_tuning.optuna_trainer import OptunaStrategyATrainer
from strategy_a.config import TrainingConfig


# Optuna 模块目录
OPTUNA_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OPTUNA_DIR, 'results')
DB_PATH = os.path.join(OPTUNA_DIR, 'optuna.db')


def get_base_args(args):
    """
    创建基础命令行参数对象
    """
    class BaseArgs:
        pass
    
    base = BaseArgs()
    
    # 数据集配置
    base.datapath = args.datapath
    base.config = args.config
    base.dataset = args.dataset
    base.imb_factor = args.imb_factor
    base.model_fixed = args.model_fixed
    
    # 训练配置
    base.epoch = args.epoch
    base.diffusion_step = args.diffusion_step
    base.ddim_steps = args.ddim_steps
    base.generation_interval = args.generation_interval
    base.warmup_epochs = args.warmup_epochs
    
    # 默认超参数 (会被 Optuna 覆盖)
    base.lr = 0.001
    base.lambda_ema = 0.2
    base.beta_radius = 0.1
    base.eta_p = 0.1
    base.eta_r = 0.2
    base.eta_m = 0.25
    base.lambda_sem = 0.01
    base.gamma_ge = 0.15
    base.tau = -1
    base.lambda_cal = 0.4
    base.margin_m = 3.5
    base.stage3_mode = 'hybrid'
    base.beta_cons = 0.1
    base.gamma_pseudo = 0.8
    base.stage1_end_epoch = 100
    base.stage2_end_epoch = 300
    
    # 其他配置
    base.use_wcdas = False
    base.wcdas_gamma = 0
    base.wcdas_traiFalsenable_scale = False
    base.use_radius_constraint = True
    base.target_radius = 1.0
    base.enable_stage3_calibration = True
    base.stage3_calibration_strength = 0.5
    
    return base


def objective(trial: optuna.Trial, base_args) -> float:
    """
    Optuna 目标函数
    
    Args:
        trial: Optuna Trial 对象
        base_args: 基础命令行参数
    
    Returns:
        float: 验证准确率 (越高越好)
    """
    # 1. 从 trial 采样超参数
    search_params = define_search_space(trial)
    
    print(f"\n{'='*60}")
    print(f"[Trial {trial.number}] 开始")
    print(f"{'='*60}")
    print("采样的超参数:")
    for key, value in search_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # 2. 合并参数
    class CombinedArgs:
        pass
    
    args = CombinedArgs()
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    for key, value in search_params.items():
        setattr(args, key, value)
    
    # 3. 创建配置和训练器
    config = TrainingConfig(args)
    trainer = OptunaStrategyATrainer(config, trial=trial, report_interval=10)
    
    # 4. 执行训练
    try:
        best_accuracy = trainer.train()
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"[Trial {trial.number}] 训练失败: {e}")
        raise
    
    print(f"\n[Trial {trial.number}] 完成! Best Accuracy: {100*best_accuracy:.2f}%\n")
    
    return best_accuracy


def create_study(study_name: str = None, resume: bool = False) -> optuna.Study:
    """
    创建或加载 Optuna Study
    
    Args:
        study_name: Study 名称
        resume: 是否从已有 Study 恢复
    
    Returns:
        optuna.Study: Optuna Study 对象
    """
    if study_name is None:
        study_name = f"gald_dc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    storage = f"sqlite:///{DB_PATH}"
    
    # 创建采样器和剪枝器
    sampler = TPESampler(
        n_startup_trials=10,  # 前 10 次使用随机采样
        seed=42
    )
    
    pruner = MedianPruner(
        n_startup_trials=5,   # 至少 5 次试验后才开始剪枝
        n_warmup_steps=10,    # Stage 3 的前 10 个 epoch 不剪枝
        interval_steps=10     # 每 10 个 epoch 检查一次
    )
    
    if resume:
        # 尝试加载已有 Study
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler,
                pruner=pruner
            )
            print(f"[Optuna] 已加载 Study '{study_name}', 已完成 {len(study.trials)} 次试验")
        except KeyError:
            print(f"[Optuna] Study '{study_name}' 不存在，创建新 Study")
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",
                sampler=sampler,
                pruner=pruner
            )
    else:
        # 创建新 Study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
    
    return study


def save_results(study: optuna.Study, output_dir: str = None, 
                 dataset: str = "CIFAR10", imb_factor: float = 0.01):
    """
    保存优化结果
    
    Args:
        study: Optuna Study 对象
        output_dir: 输出目录
        dataset: 数据集名称
        imb_factor: 不平衡因子
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 文件名包含数据集和不平衡因子
    file_suffix = f"{dataset}_imb{imb_factor}"
    
    # 1. 保存最佳参数
    best_params_path = os.path.join(output_dir, f'best_params_{file_suffix}.json')
    best_result = {
        'dataset': dataset,
        'imb_factor': imb_factor,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial_number': study.best_trial.number,
        'n_trials': len(study.trials),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump(best_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Optuna] 最佳参数已保存到: {best_params_path}")
    
    # 2. 保存所有试验结果
    all_trials_path = os.path.join(output_dir, f'all_trials_{file_suffix}.json')
    all_trials = []
    for trial in study.trials:
        trial_data = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': str(trial.state),
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
        }
        all_trials.append(trial_data)
    
    with open(all_trials_path, 'w', encoding='utf-8') as f:
        json.dump(all_trials, f, indent=2, ensure_ascii=False)
    
    print(f"[Optuna] 所有试验结果已保存到: {all_trials_path}")
    
    # 3. 尝试生成可视化
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        # 优化历史图
        fig_history = plot_optimization_history(study)
        history_path = os.path.join(output_dir, f'optimization_history_{file_suffix}.html')
        fig_history.write_html(history_path)
        print(f"[Optuna] 优化历史图已保存到: {history_path}")
        
        # 参数重要性图 (需要至少 10 次成功试验)
        completed_trials = [t for t in study.trials if t.value is not None]
        if len(completed_trials) >= 10:
            fig_importance = plot_param_importances(study)
            importance_path = os.path.join(output_dir, f'param_importances_{file_suffix}.html')
            fig_importance.write_html(importance_path)
            print(f"[Optuna] 参数重要性图已保存到: {importance_path}")
    except ImportError:
        print("[Optuna] 跳过可视化 (需要安装 plotly)")
    except Exception as e:
        print(f"[Optuna] 可视化生成失败: {e}")


def load_best_params(dataset: str, imb_factor: float, results_dir: str = None) -> dict:
    """
    从 JSON 文件加载最佳参数（用于热启动）
    
    Args:
        dataset: 数据集名称
        imb_factor: 不平衡因子
        results_dir: 结果目录
    
    Returns:
        dict: 最佳参数字典，如果文件不存在返回 None
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    
    file_suffix = f"{dataset}_imb{imb_factor}"
    filepath = os.path.join(results_dir, f'best_params_{file_suffix}.json')
    
    if not os.path.exists(filepath):
        print(f"[Optuna] 未找到热启动文件: {filepath}")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[Optuna] 已从 {filepath} 加载热启动参数")
    print(f"  加载的最佳准确率: {100 * data['best_value']:.2f}%")
    
    return data['best_params']


def run_optuna_study(base_args, n_trials: int = 50, study_name: str = None, 
                     resume: bool = False, warm_start: bool = False) -> optuna.Study:
    """
    运行 Optuna 超参数优化
    
    Args:
        base_args: 基础命令行参数
        n_trials: 试验次数
        study_name: Study 名称
        resume: 是否断点续跑
        warm_start: 是否使用热启动（从已有最佳参数开始搜索）
    
    Returns:
        optuna.Study: 完成的 Study 对象
    """
    print("\n" + "=" * 60)
    print("GALD-DC Optuna 自动调参")
    print("=" * 60)
    print(get_search_space_summary())
    print(f"\n计划试验次数: {n_trials}")
    print(f"数据库路径: {DB_PATH}")
    print("=" * 60 + "\n")
    
    # 创建 Study
    study = create_study(study_name=study_name, resume=resume)
    
    # 热启动：将已有最佳参数作为第一个试验点
    if warm_start:
        warm_params = load_best_params(base_args.dataset, base_args.imb_factor)
        if warm_params:
            study.enqueue_trial(warm_params)
            print(f"[Optuna] 已将热启动参数加入队列")
    
    # 运行优化
    study.optimize(
        lambda trial: objective(trial, base_args),
        n_trials=n_trials,
        catch=(Exception,),  # 捕获异常继续下一次试验
        gc_after_trial=True  # 每次试验后进行垃圾回收
    )
    
    # 输出结果
    print("\n" + "=" * 60)
    print("优化完成!")
    print("=" * 60)
    
    # 检查是否有完成的试验
    completed_trials = [t for t in study.trials if t.value is not None]
    if len(completed_trials) > 0:
        print(f"完成的试验数: {len(completed_trials)}")
        print(f"最佳准确率: {100 * study.best_value:.2f}%")
        print(f"最佳参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # 保存结果（包含数据集和不平衡因子信息）
        save_results(study, dataset=base_args.dataset, imb_factor=base_args.imb_factor)
    else:
        print("警告: 没有成功完成的试验，无法输出最佳参数")
        print("请检查训练过程中的错误日志")
    
    print("=" * 60)
    
    return study


def main():
    parser = argparse.ArgumentParser(description='GALD-DC Optuna 自动调参')
    
    # Optuna 参数
    parser.add_argument('--n_trials', type=int, default=50, help='试验次数')
    parser.add_argument('--study_name', type=str, default=None, help='Study 名称')
    parser.add_argument('--resume', action='store_true', help='断点续跑')
    parser.add_argument('--warm_start', action='store_true', 
                        help='热启动：从已保存的最佳参数开始搜索')
    
    # 数据集参数
    parser.add_argument('--datapath', default=r"E:\Projects\LDMLR-main\data", type=str, help='数据集路径')
    parser.add_argument('--config', default="./config/cifar10/cifar10_LSC_Mixup.txt", help='配置文件路径')
    parser.add_argument('--dataset', default="CIFAR10", type=str, help='数据集名称')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='不平衡因子')
    parser.add_argument('--model_fixed', default='./pretrained_models/resnet32_cifar10_lt001.checkpoint', 
                        type=str, help='预训练模型路径')
    
    # 训练参数
    parser.add_argument('--epoch', default=500, type=int, help='总训练轮数')
    parser.add_argument('--diffusion_step', default=1000, type=int, help='扩散步数')
    parser.add_argument('--ddim_steps', default=100, type=int, help='DDIM 采样步数')
    parser.add_argument('--generation_interval', default=10, type=int, help='生成间隔')
    parser.add_argument('--warmup_epochs', default=30, type=int, help='预热轮数')
    
    args = parser.parse_args()
    
    # 创建基础参数
    base_args = get_base_args(args)
    
    # 运行优化
    study = run_optuna_study(
        base_args=base_args,
        n_trials=args.n_trials,
        study_name=args.study_name,
        resume=args.resume,
        warm_start=args.warm_start
    )
    
    return study


if __name__ == '__main__':
    main()
