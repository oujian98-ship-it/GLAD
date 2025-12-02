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
        
        # 确保logs目录存在
        logs_dir = "../logs"
        log_filename = os.path.join(logs_dir, f"{self.config.dataset}_strategy_A_{self.config.imb_factor}_{current_time}.log")
        logging.basicConfig(
            filename=log_filename, 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def log_batch_progress(self, epoch: int, batch_idx: int, losses: Dict[str, float]):
        """
        记录批次进度
        """
        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, "
                  f"Real Loss: {losses['real']:.4f}, "
                  f"Semantic Loss: {losses['semantic']:.4f}, "
                  f"Gen Loss: {losses['gen']:.4f}, "
                  f"Total Loss: {losses['total']:.4f}")
    
    def log_epoch_summary(self, epoch: int, avg_losses: Dict[str, float], 
                         train_accuracy: float = None, train_loss: float = None):
        """
        记录轮次总结
        """
        if train_loss is not None and train_accuracy is not None:
            logging.info("cnn training loss is {}; cnn training accuracy is {:.2f}".format(
                train_loss, train_accuracy * 100))
            print("cnn training loss is {}; cnn training accuracy is {:.2f}".format(
                train_loss, train_accuracy * 100))

        print(f"Epoch {epoch} Summary - "
              f"Real Loss: {avg_losses['real']:.4f}, "
              f"Semantic Loss: {avg_losses['semantic']:.4f}, "
              f"Gen Loss: {avg_losses['gen']:.4f}, "
              f"Total Loss: {avg_losses['total']:.4f}")
        
        logging.info(f"Epoch {epoch} Summary - "
                    f"Real Loss: {avg_losses['real']:.4f}, "
                    f"Semantic Loss: {avg_losses['semantic']:.4f}, "
                    f"Gen Loss: {avg_losses['gen']:.4f}, "
                    f"Total Loss: {avg_losses['total']:.4f}")
    
    def log_validation(self, epoch: int, accuracy: float, test_loss: float, 
                      label_shift_acc: float, mmf_acc: List, mmf_acc_pc: List, 
                      wcdas_accuracy: float = 0.0):
        """
        记录验证结果（保留原方法以兼容旧代码）
        """
        # 保存当前epoch的准确率到历史记录
        self.accuracies_history.append({
            'epoch': epoch,
            'accuracy': accuracy,
            'label_shift_acc': label_shift_acc,
            'mmf_acc': mmf_acc,
            'mmf_acc_pc': mmf_acc_pc,
            'wcdas_accuracy': wcdas_accuracy
        })
        
        # 准确率输出
        print(f"Test Error:   Accuracy: {100 * accuracy:.2f}, Avg loss: {test_loss:.4f}")
        logging.info(f"Test Error:   Accuracy: {100 * accuracy:.2f}, Avg loss: {test_loss:.4f}")
        
        # Many-Medium-Few shot准确率
        print(f"Many Medium Few shot Top1 Acc: {mmf_acc}")
        logging.info(f"Many Medium Few shot Top1 Acc: {mmf_acc}")
        
        # Label Shift后补偿准确率
        print(f"Label Shift Accuracy is: {label_shift_acc}")
        logging.info(f"Label Shift Accuracy is: {label_shift_acc}")
        
        # Many-Medium-Few shot后补偿准确率
        print(f"Many Medium Few shot Top1 Acc (Post Compensation): {mmf_acc_pc}")
        logging.info(f"Many Medium Few shot Top1 Acc (Post Compensation): {mmf_acc_pc}")
        print("\n")
        logging.info("\n")
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print(f"New best accuracy: {100 * accuracy:.2f}%")
            print("\n")
            logging.info("\n")
    

    
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
            best_epoch = max(self.accuracies_history, key=lambda x: x['accuracy'])
            best_accuracy = best_epoch['accuracy']
            best_label_shift_acc = best_epoch['label_shift_acc']
            
            # 使用模型管理器保存最佳模型
            model_manager.save_best_models(encoder, classifier, diffusion_model, 
                                         best_accuracy, best_label_shift_acc)
    
    def log_training_complete(self):
        """
        记录训练完成信息，输出最佳模型准确率
        """
        if self.accuracies_history:
            # 找出最佳准确率和对应的Label Shift准确率
            best_epoch = max(self.accuracies_history, key=lambda x: x['accuracy'])
            best_accuracy = best_epoch['accuracy']
            best_label_shift_acc = best_epoch['label_shift_acc']
            best_mmf_acc = best_epoch['mmf_acc']
            best_mmf_acc_pc = best_epoch['mmf_acc_pc']
            
            # 找出最佳Label Shift准确率
            best_label_shift_epoch = max(self.accuracies_history, key=lambda x: x['label_shift_acc'])
            best_label_shift_only = best_label_shift_epoch['label_shift_acc']
            best_mmf_acc_pc_best = best_label_shift_epoch['mmf_acc_pc']
            
            # 找出最佳WCDAS准确率
            best_wcdas_epoch = max(self.accuracies_history, key=lambda x: x.get('wcdas_accuracy', 0))
            best_wcdas_accuracy = best_wcdas_epoch.get('wcdas_accuracy', 0)
            
            # 输出最佳模型信息
            if self.config.use_wcdas:
                # 使用WCDAS训练时，best accuracy就是WCDAS准确率
                logging.info(" The best WCDAS accuracy is {:.2f}, The best label shift accuracy is {:.2f}".format(
                    100 * best_accuracy, best_label_shift_only))
                logging.info(" best accuracy wcdas mmf is {}; best acc pc mmf is {}".format(
                    best_mmf_acc, best_mmf_acc_pc_best))
                print(" The best WCDAS accuracy is {:.2f}, The best label shift accuracy is {:.2f}".format(
                    100 * best_accuracy, best_label_shift_only))
                print(" best accuracy wcdas mmf is {}; best acc pc mmf is {}".format(
                    best_mmf_acc, best_mmf_acc_pc_best))
            else:
                # 使用标准训练时
                logging.info(" The best accuracy is {:.2f}, The best label shift accuracy is {:.2f}".format(
                    100 * best_accuracy, best_label_shift_only))
                logging.info(" best accuracy ce mmf is {}; best acc pc mmf is {}".format(
                    best_mmf_acc, best_mmf_acc_pc_best))
                print(" The best accuracy is {:.2f}, The best label shift accuracy is {:.2f}".format(
                    100 * best_accuracy, best_label_shift_only))
                print(" best accuracy ce mmf is {}; best acc pc mmf is {}".format(
                    best_mmf_acc, best_mmf_acc_pc_best))