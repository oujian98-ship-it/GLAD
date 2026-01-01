"""
训练监控模块
负责训练过程中的日志记录、进度监控和模型保存
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainingConfig


class TrainingMonitor:
    """
    训练监控器，负责日志记录和进度监控
    """
    
    def __init__(self, config: TrainingConfig):
        """
        初始化训练监控器
        """
        self.config = config
        self._setup_logging()
        self.best_accuracy = 0.0  # 最佳CE/WCDAS准确率
        self.best_label_shift_acc = 0.0  # 最佳Label Shift准确率
        self.accuracies_history = []  # 保存每个epoch的准确率历史
    
    def _setup_logging(self):
        """
        设置日志系统
        """
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 使用项目根目录的 logs 文件夹 (修复相对路径问题)
        # 获取当前文件所在目录，向上一级到达项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(project_root, "logs")
        
        # 确保 logs 目录存在
        os.makedirs(logs_dir, exist_ok=True)
        
        log_filename = os.path.join(logs_dir, f"{self.config.dataset}_{self.config.imb_factor}_{current_time}.log")
        logging.basicConfig(
            filename=log_filename, 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def log_batch_progress(self, epoch: int, batch_idx: int, losses: Dict[str, float], stage: int = 3):
        """
        记录批次进度 (根据阶段显示不同内容)
        """
        if batch_idx % 100 == 0:
            if stage == 1:
                print(f" Epoch: {epoch}, Batch: {batch_idx}, CE Loss: {losses['real']:.4f}")
            elif stage == 2:
                print(f" Epoch: {epoch}, Batch: {batch_idx}, "
                      f"L_LDM: {losses.get('diffusion', 0):.4f}, "
                      f"L_proto: {losses.get('prototype', 0):.4f}, "
                      f"L_rad: {losses.get('radius', 0):.4f}, "
                      f"L_margin: {losses.get('margin', 0):.4f}")
            else:
                print(f" Epoch: {epoch}, Batch: {batch_idx}, "
                      f"Real: {losses['real']:.4f}, Gen: {losses['gen']:.4f}, "
                      f"Cons: {losses.get('consistency', 0):.4f}")
    
    def log_epoch_summary(self, epoch: int, avg_losses: Dict[str, float], 
                         train_accuracy: float = None, train_loss: float = None, stage: int = 3):
        """
        记录轮次总结 (根据阶段显示不同内容)
        """
        if stage == 1:
            msg = f" Epoch {epoch} - CE Loss: {avg_losses['real']:.4f}"
            print(msg)
            logging.info(msg)
            
        elif stage == 2:
            msg = (f"Epoch {epoch} - "
                   f"L_LDM: {avg_losses.get('diffusion', 0):.4f}, "
                   f"L_proto: {avg_losses.get('prototype', 0):.4f}, "
                   f"L_rad: {avg_losses.get('radius', 0):.4f}, "
                   f"L_margin: {avg_losses.get('margin', 0):.4f}, "
                   f"Total: {avg_losses['total']:.4f}")
            print(msg)
            logging.info(msg)
            
        else: 
            if train_loss is not None and train_accuracy is not None:
                logging.info("cnn training loss is {}; cnn training accuracy is {:.2f}".format(
                    train_loss, train_accuracy * 100))
                print("cnn training loss is {}; cnn training accuracy is {:.2f}".format(
                    train_loss, train_accuracy * 100))

            msg = (f"Epoch {epoch} Summary - "
                   f"Real Loss: {avg_losses['real']:.4f}, "
                   f"Gen Loss: {avg_losses['gen']:.4f}, "
                   f"Consistency: {avg_losses.get('consistency', 0):.4f}, "
                   f"Total: {avg_losses['total']:.4f}")
            print(msg)
            logging.info(msg)
    
    def log_validation(self, epoch: int, accuracy: float, test_loss: float, 
                      label_shift_acc: float, mmf_acc: List, mmf_acc_pc: List, 
                      wcdas_accuracy: float = 0.0, mode: str = 'Val'):
        """
        按照用户喜好的格式记录验证结果 (垂直排列)
        """
        method_name = "WCDAS" if self.config.use_wcdas else "CE"
        method_display = f"{method_name} " if method_name == "CE" else method_name
        set_name = "Validation" if mode == 'Val' else "Test"
        
        # 保存历史记录 (记录所有验证/测试点)
        self.accuracies_history.append({
            'epoch': epoch,
            'mode': mode,
            'method': method_name,
            'base_accuracy': accuracy,
            'accuracy': accuracy,
            'label_shift_acc': label_shift_acc,
            'base_mmf': mmf_acc,
            'mmf_acc': mmf_acc,
            'label_shift_mmf': mmf_acc_pc,
            'mmf_acc_pc': mmf_acc_pc
        })
        
        # 终端输出 (还原为用户熟悉的垂直样式)
        print(f"Epoch {epoch} {set_name} Results")
        print(f"{method_display} Results:")
        print(f"  {set_name} Loss:       {test_loss:.4f}")
        print(f"  {set_name} Acc:        {100 * accuracy:.2f}%")
        print(f"  MMF Acc:         {mmf_acc}")
        
        print(f"Label Shift Results:")
        print(f"  {set_name} Acc:        {label_shift_acc:.2f}%")
        print(f"  MMF Acc:         {mmf_acc_pc}\n")
        
        # 记录日志
        logging.info(f"Epoch {epoch} {set_name} Results - {method_display}")
        logging.info(f"  {set_name} Accuracy: {100 * accuracy:.2f}%, Loss: {test_loss:.4f}")
        logging.info(f"  {set_name} MMF: {mmf_acc}")
        logging.info(f"  Label Shift Accuracy: {label_shift_acc:.2f}%")
        logging.info(f"  Label Shift MMF: {mmf_acc_pc}")
        
        # 更新最佳准确率
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print(f" New best {set_name} {method_display} accuracy: {100 * accuracy:.2f}%\n")
            logging.info(f"New best {set_name} accuracy: {100 * accuracy:.2f}%")

    
    def get_accuracy_history(self) -> List[Dict]:
        """
        获取准确率历史记录
        """
        return self.accuracies_history
    
    def get_best_accuracy(self) -> float:
        """
        获取最佳准确率
        
        返回:
            float: 最佳准确率
        """
        return self.best_accuracy
    
    def save_best_checkpoints(self, encoder, classifier, diffusion_model, model_manager):
        """
        保存最佳模型检查点
        """
        if self.accuracies_history:
            # 找出最佳准确率和对应的Label Shift准确率
            best_epoch = max(self.accuracies_history, key=lambda x: x['base_accuracy'])
            best_accuracy = best_epoch['base_accuracy']
            best_label_shift_acc = best_epoch['label_shift_acc']
            
            # 使用模型管理器保存最佳模型
            model_manager.save_best_models(encoder, classifier, diffusion_model, 
                                         best_accuracy, best_label_shift_acc)
    
    def log_training_complete(self):
        """
        记录训练完成信息，输出最佳模型准确率
        """
        if self.accuracies_history:
            # 确定方法名称
            method_name = "WCDAS" if self.config.use_wcdas else "CE"
            
            # 找出最佳准确率和对应的Label Shift准确率
            best_epoch = max(self.accuracies_history, key=lambda x: x['base_accuracy'])
            best_accuracy = best_epoch['base_accuracy']
            best_label_shift_acc = best_epoch['label_shift_acc']
            best_mmf_acc = best_epoch['base_mmf']
            best_mmf_acc_pc = best_epoch['label_shift_mmf']
            
            # 找出最佳Label Shift准确率
            best_label_shift_epoch = max(self.accuracies_history, key=lambda x: x['label_shift_acc'])
            best_label_shift_only = best_label_shift_epoch['label_shift_acc']
            best_mmf_acc_pc_best = best_label_shift_epoch['label_shift_mmf']
            
            # 输出分隔线
            print(f"\n{'='*70}")
            print(f"Training Complete - Final Results")
            
            # 使用 "Ours" 后缀表示包含 GALD-DC 增强
            method_display = f"{method_name} + Ours" if method_name == "CE" else method_name
            
            # 输出最佳模型信息
            print(f"Best {method_display} :")
            print(f"  Accuracy:        {100 * best_accuracy:.2f}%")
            print(f"  MMF Acc:         {best_mmf_acc}")
            
            print(f"\nBest Label Shift  :")
            print(f"  Accuracy:        {best_label_shift_only:.2f}%")
            print(f"  MMF Acc:         {best_mmf_acc_pc_best}")
            
            
            # 记录到日志文件
            logging.info(f"Training Complete - Method: {method_display}")
            logging.info(f"  Best {method_display} Accuracy: {100 * best_accuracy:.2f}%")
            logging.info(f"  Best {method_display} MMF: {best_mmf_acc}")
            logging.info(f"  Best Label Shift  Accuracy: {best_label_shift_only:.2f}%")
            logging.info(f"  Best Label Shift  MMF: {best_mmf_acc_pc_best}")