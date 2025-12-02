# Label Shift 最佳模型保存功能 - 手动修改指南

## 概述

添加 Label Shift 最佳模型的保存功能，使 Strategy A 与原项目保持一致。

**修改目标**：
- 追踪两个独立的最佳准确率：CE/WCDAS 和 Label Shift
- 保存两个独立的模型检查点：`best_ce.pth` 和 `best_pc.pth`

---

## ⚠️ 重要提示

**请先恢复被破坏的文件**：
```bash
git checkout strategy_a/trainer.py
git checkout strategy_a/training_monitor.py
```

---

## 修改步骤

### 📝 步骤1：修改 `training_monitor.py`

**文件**：`e:\projects\LDMLR-main\strategy_a\training_monitor.py`

**位置**：第 23-30 行

**查找**：
```python
def __init__(self, config: TrainingConfig):
    """
    初始化训练监控器
    """
    self.config = config
    self._setup_logging()
    self.best_accuracy = 0.0
    self.accuracies_history = []
```

**替换为**：
```python
def __init__(self, config: TrainingConfig):
    """
    初始化训练监控器
    """
    self.config = config
    self._setup_logging()
    self.best_accuracy = 0.0  # 最佳CE/WCDAS准确率
    self.best_label_shift_acc = 0.0  # 最佳Label Shift准确率
    self.accuracies_history = []
```

**变化**：添加一行 `self.best_label_shift_acc = 0.0`

---

### 📝 步骤2：修改 `trainer.py` - 保存逻辑

**文件**：`e:\projects\LDMLR-main\strategy_a\trainer.py`

**位置**：第 111-120 行

**查找**：
```python
self.monitor.log_validation(epoch, accuracy, test_loss, label_shift_acc, mmf_acc, mmf_acc_pc)

# 保存最佳模型
# 注意：这里的 accuracy 就是当前模式下最重要的指标
# WCDAS 模式下它就是 WCDAS Acc，CE 模式下它就是 CE Acc
if accuracy > self.monitor.best_accuracy:
    self._save_checkpoint(epoch, encoder, classifier, diffusion_model, optimizer, accuracy)

# 定期保存扩散模型
```

**替换为**：
```python
self.monitor.log_validation(epoch, accuracy, test_loss, label_shift_acc, mmf_acc, mmf_acc_pc)

# 保存最佳CE/WCDAS准确率模型
if accuracy > self.monitor.best_accuracy:
    self._save_checkpoint(epoch, encoder, classifier, diffusion_model, optimizer, accuracy, 'ce')

# 保存最佳Label Shift准确率模型
if label_shift_acc > self.monitor.best_label_shift_acc:
    self.monitor.best_label_shift_acc = label_shift_acc
    self._save_checkpoint(epoch, encoder, classifier, diffusion_model, optimizer, label_shift_acc, 'pc')

# 定期保存扩散模型
```

**变化**：
- 第一个保存添加参数 `'ce'`
- 新增 Label Shift 最佳模型保存逻辑（4行）

---

### 📝 步骤3：修改 `trainer.py` - `_save_checkpoint` 方法签名

**文件**：`e:\projects\LDMLR-main\strategy_a\trainer.py`

**位置**：第 551-564 行

**查找**：
```python
def _save_checkpoint(self, epoch, encoder, classifier, diffusion_model, optimizer, accuracy):
    checkpoint = {
        'epoch': epoch,
        'encoder': encoder.state_dict(),
        'classifier': classifier.state_dict(),
        'diffusion': diffusion_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': accuracy
    }
    # 文件名包含模式，方便区分
    mode = "wcdas" if self.config.use_wcdas else "ce"
    path = f"ckpt_strategy_A_{mode}_{self.config.dataset}_epoch_{epoch}.pth"
    torch.save(checkpoint, path)
    print(f"Saved best checkpoint to {path}")
```

**替换为**：
```python
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
        print(f"✅ Saved best CE/WCDAS checkpoint (epoch {epoch}, acc {accuracy:.4f}) to {path}")
    else:  # model_type == 'pc'
        path = f"ckpt_strategy_A_{mode}_best_pc.pth"
        print(f"✅ Saved best Label Shift checkpoint (epoch {epoch}, acc {accuracy:.4f}) to {path}")
    
    torch.save(checkpoint, path)
```

**变化**：
- 方法签名添加 `model_type='ce'` 参数
- 文件名从包含epoch改为 `best_ce.pth` / `best_pc.pth`
- 输出信息更详细

---

## ✅ 修改完成后的效果

训练时会自动保存两个模型：

1. **`ckpt_strategy_A_wcdas_best_ce.pth`**  
   - CE/WCDAS准确率达到最佳时保存
   - 对应原项目的 `ckpt_best_ce.checkpoint`

2. **`ckpt_strategy_A_wcdas_best_pc.pth`**  
   - Label Shift准确率达到最佳时保存
   - 对应原项目的 `ckpt_best_pc.checkpoint`

---

## 🔍 验证修改

修改后运行训练，应该看到类似输出：

```
✅ Saved best CE/WCDAS checkpoint (epoch 47, acc 0.8110) to ckpt_strategy_A_wcdas_best_ce.pth
✅ Saved best Label Shift checkpoint (epoch 93, acc 0.7807) to ckpt_strategy_A_wcdas_best_pc.pth
```

---

## 📊 与原项目的对应关系

| 原项目 | Strategy A | 说明 |
|--------|-----------|------|
| `ckpt_best_ce.checkpoint` | `ckpt_strategy_A_wcdas_best_ce.pth` | CE/WCDAS最佳 |
| `ckpt_best_pc.checkpoint` | `ckpt_strategy_A_wcdas_best_pc.pth` | Label Shift最佳 |
