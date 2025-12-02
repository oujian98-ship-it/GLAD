# Strategy A 双重评估系统实施方案

**目标**：在一次训练中同时评估 CE 和 WCDAS 方法，完整对比长尾分布下的性能差异

**版本**：v1.0  
**日期**：2025-12-02

---

## 📋 目录

1. [背景与目标](#1-背景与目标)
2. [评估指标体系](#2-评估指标体系)
3. [技术实现方案](#3-技术实现方案)
4. [代码修改详解](#4-代码修改详解)
5. [预期输出示例](#5-预期输出示例)
6. [验证与测试](#6-验证与测试)
7. [FAQ](#7-faq)

---

## 1. 背景与目标

### 1.1 当前问题

Strategy A 训练时只能看到一种方法（WCDAS）的效果，无法直接对比：
- WCDAS 的 tau 调整是否真的有效？
- WCDAS 相比标准 CE 提升了多少？
- Label Shift Compensation 对两种方法的影响有何不同？

### 1.2 解决方案

**在验证阶段同时计算 4 个核心指标**：

| 方法 | 原始准确率 | LSC修正后准确率 |
|------|-----------|----------------|
| **CE** | CE Acc | CE + LSC Acc |
| **WCDAS** | WCDAS Acc | WCDAS + LSC Acc |

### 1.3 核心价值

1. ✅ **节省时间** - 一次训练得到所有对比数据
2. ✅ **公平对比** - 相同特征、相同测试集
3. ✅ **诊断工具** - 量化 tau 的实际效果
4. ✅ **优化指引** - 判断是否需要调整 tau

---

## 2. 评估指标体系

### 2.1 四大核心指标

#### **指标1: CE Acc (Raw)**
```
含义：标准交叉熵，不带任何调整
特点：严重偏向Many类，Few类准确率低
用途：建立基准，展示原始问题
```

#### **指标2: CE + LSC Acc**
```
含义：CE预测 + Label Shift后处理修正
特点：通过后处理平衡分布
用途：对比基准 - 理论上应接近 WCDAS Acc
```

#### **指标3: WCDAS Acc (Raw)**
```
含义：带tau调整的WCDAS预测
特点：训练时已内化偏见修正
用途：核心指标 - 展示WCDAS效果
```

#### **指标4: WCDAS + LSC Acc**
```
含义：WCDAS预测 + Label Shift后处理
特点：双重修正
用途：诊断工具 - 检验WCDAS是否充分
```

---

### 2.2 关键对比

**如果 WCDAS 有效**：
```
CE Acc:          74.23%  ← 基准
CE + LSC:        77.50%  ← LSC修正+3.27%
WCDAS Acc:       76.86%  ← tau调整+2.63%
WCDAS + LSC:     77.20%  ← 仅+0.34% ✅

结论：WCDAS的LSC Gap (0.34%) << CE的LSC Gap (3.27%)
说明：tau已在训练时解决了大部分偏见问题
```

**如果 WCDAS 无效**：
```
CE Acc:          74.23%
CE + LSC:        77.50%
WCDAS Acc:       74.50%  ← 提升很小
WCDAS + LSC:     77.80%  ← 仍需大幅修正 ❌

结论：WCDAS的LSC Gap (3.30%) ≈ CE的LSC Gap (3.27%)
说明：tau没起作用，需要调整超参数
```

---

## 3. 技术实现方案

### 3.1 整体架构

```
验证阶段流程：
1. 提取特征 (共享)
   ↓
2. 分别计算 logits
   ├─→ logit_wcdas (带tau)
   └─→ logit_ce (标准)
   ↓
3. 计算4个准确率
   ├─→ CE Acc
   ├─→ CE + LSC Acc
   ├─→ WCDAS Acc
   └─→ WCDAS + LSC Acc
   ↓
4. 输出对比报告
```

### 3.2 关键设计

**设计原则1**：训练仍使用 WCDAS，只在验证时双重评估

**设计原则2**：共享特征提取，减少计算开销

**设计原则3**：LSC对两种方法使用相同参数（公平对比）

---

## 4. 代码修改详解

### 4.1 修改文件清单

| 文件 | 修改内容 | 代码行数 |
|------|---------|---------|
| `loss_calculator.py` | 返回双logits | +15行 |
| `trainer.py` | 双重评估逻辑 | +60行 |
| `training_monitor.py` | 双重日志输出 | +40行 |

---

### 4.2 修改 `loss_calculator.py`

**文件路径**: `strategy_a/loss_calculator.py`

**修改位置**: `WCDASLoss.forward` 方法（约第82-115行）

**修改前**：
```python
def forward(self, input, target=None, sample_per_class=None):
    input = normalized(input)
    cosine = self.normalized_fc(input)
    
    if self.config.use_wcdas and sample_per_class is not None:
        # ... WCDAS逻辑 ...
        logit = ...  # 带tau调整
        return logit
    else:
        return cosine
```

**修改后**：
```python
def forward(self, input, target=None, sample_per_class=None):
    input = normalized(input)
    
    # 1. 计算标准CE logits（不带tau）
    cosine = self.normalized_fc(input)
    logit_ce = cosine.clone()  # 保存CE版本
    
    # 2. 计算WCDAS logits（带tau）
    if self.config.use_wcdas and sample_per_class is not None:
        # Gamma计算
        gamma = self.g.sigmoid()
        cosine_scaled = cosine * gamma
        
        # Logit Adjustment
        prior = sample_per_class / sample_per_class.sum()
        prior = prior.to(input.device)
        log_prior = torch.log(prior + 1e-9).unsqueeze(0)
        tau = 1.5  # 或从config读取
        
        # 应用tau调整
        logit_wcdas = cosine_scaled + tau * log_prior
        
        # Scaling
        if self.s_trainable:
            logit_wcdas = self.s * logit_wcdas
        
        return logit_wcdas, logit_ce  # 返回双logits
    else:
        # 验证时或无sample_per_class，返回相同的CE
        return logit_ce, logit_ce
```

**关键点**：
- ✅ 始终计算 `logit_ce`（标准CE）
- ✅ WCDAS模式下额外计算 `logit_wcdas`
- ✅ 返回元组 `(logit_wcdas, logit_ce)`

---

### 4.3 修改 `trainer.py`

**文件路径**: `strategy_a/trainer.py`

#### **修改位置1**: `_validate` 方法（约第308-380行）

**完整的新 `_validate` 方法**：

```python
def _validate(self, encoder, classifier, test_set, dataset_info, train_class_counts):
    """
    验证函数 - 双重评估版本
    同时计算 CE 和 WCDAS 的 4 个核心指标
    """
    encoder.eval()
    classifier.eval()
    
    test_loss = 0
    total = 0
    
    # 准确率计数器
    correct_wcdas = 0
    correct_ce = 0
    
    # 概率收集器
    probs_wcdas_list = []
    probs_ce_list = []
    labels_list = []
    
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    cls_num_list = dataset_info['per_class_img_num']
    
    with torch.no_grad():
        for inputs, label in test_set:
            inputs = inputs.to(self.device)
            label = label.to(self.device)
            labels_list.append(label)
            
            # 提取特征（共享）
            features = encoder.forward_no_fc(inputs)
            
            # 获取双logits
            if hasattr(classifier, 'forward'):
                try:
                    outputs = classifier(
                        input=features,
                        target=label,
                        sample_per_class=train_class_counts
                    )
                    # 检查是否返回双logits
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        logits_wcdas, logits_ce = outputs
                    else:
                        # 向后兼容：如果只返回一个
                        logits_wcdas = logits_ce = outputs
                except TypeError:
                    # 如果classifier不支持参数
                    logits_wcdas = logits_ce = classifier(features)
            else:
                logits_wcdas = logits_ce = classifier(features)
            
            # 计算损失（使用WCDAS logits）
            test_loss += loss_fn(logits_wcdas, label).item()
            
            # ========== WCDAS 评估 ==========
            prob_wcdas = F.softmax(logits_wcdas, dim=1)
            probs_wcdas_list.extend(prob_wcdas.cpu().numpy())
            pred_wcdas = prob_wcdas.argmax(dim=1)
            correct_wcdas += (pred_wcdas == label).sum().item()
            
            # ========== CE 评估 ==========
            prob_ce = F.softmax(logits_ce, dim=1)
            probs_ce_list.extend(prob_ce.cpu().numpy())
            pred_ce = prob_ce.argmax(dim=1)
            correct_ce += (pred_ce == label).sum().item()
            
            total += label.size(0)
    
    # 转换为numpy数组
    probs_wcdas = np.array(probs_wcdas_list)
    probs_ce = np.array(probs_ce_list)
    labels = torch.cat(labels_list).cpu()
    
    # 计算基础准确率
    acc_wcdas = correct_wcdas / total
    acc_ce = correct_ce / total
    test_loss /= len(test_set)
    
    # ========== WCDAS 指标 ==========
    # 1. WCDAS Acc (Raw)
    _, mmf_wcdas = self._get_metrics(probs_wcdas, labels, cls_num_list)
    
    # 2. WCDAS + LSC Acc
    probs_wcdas_lsc = LSC(probs_wcdas, cls_num_list=cls_num_list)
    acc_wcdas_lsc, mmf_wcdas_lsc = self._get_metrics(
        probs_wcdas_lsc, labels, cls_num_list
    )
    
    # ========== CE 指标 ==========
    # 3. CE Acc (Raw)
    _, mmf_ce = self._get_metrics(probs_ce, labels, cls_num_list)
    
    # 4. CE + LSC Acc
    probs_ce_lsc = LSC(probs_ce, cls_num_list=cls_num_list)
    acc_ce_lsc, mmf_ce_lsc = self._get_metrics(
        probs_ce_lsc, labels, cls_num_list
    )
    
    # 返回完整结果字典
    return {
        'test_loss': test_loss,
        'wcdas': {
            'acc': acc_wcdas,
            'lsc_acc': acc_wcdas_lsc,
            'mmf': mmf_wcdas,
            'mmf_lsc': mmf_wcdas_lsc,
            'lsc_gap': acc_wcdas_lsc - acc_wcdas  # Gap指标
        },
        'ce': {
            'acc': acc_ce,
            'lsc_acc': acc_ce_lsc,
            'mmf': mmf_ce,
            'mmf_lsc': mmf_ce_lsc,
            'lsc_gap': acc_ce_lsc - acc_ce  # Gap指标
        }
    }
```

#### **修改位置2**: 主训练循环调用（约第108-115行）

**修改前**：
```python
test_loss, accuracy, label_shift_acc, mmf_acc, mmf_acc_pc = \
    self._validate(encoder, classifier, test_set, dataset_info, train_class_counts)

self.monitor.log_validation(epoch, accuracy, test_loss, label_shift_acc, mmf_acc, mmf_acc_pc)
```

**修改后**：
```python
# 验证（返回字典结构）
val_results = self._validate(encoder, classifier, test_set, dataset_info, train_class_counts)

# 记录双重验证结果
self.monitor.log_validation_dual(epoch, val_results)

# 保存最佳模型（使用WCDAS准确率）
wcdas_acc = val_results['wcdas']['acc']
wcdas_lsc = val_results['wcdas']['lsc_acc']

if wcdas_acc > self.monitor.best_accuracy:
    self._save_checkpoint(epoch, encoder, classifier, diffusion_model, 
                         optimizer, wcdas_acc, 'ce')

if wcdas_lsc > self.monitor.best_label_shift_acc:
    self.monitor.best_label_shift_acc = wcdas_lsc
    self._save_checkpoint(epoch, encoder, classifier, diffusion_model, 
                         optimizer, wcdas_lsc, 'pc')
```

---

### 4.4 修改 `training_monitor.py`

**文件路径**: `strategy_a/training_monitor.py`

**添加新方法**: `log_validation_dual`

```python
def log_validation_dual(self, epoch: int, val_results: dict):
    """
    记录双重验证结果
    
    Args:
        epoch: 当前epoch
        val_results: 验证结果字典，包含ce和wcdas的所有指标
    """
    # 提取数据
    test_loss = val_results['test_loss']
    
    wcdas_acc = val_results['wcdas']['acc']
    wcdas_lsc = val_results['wcdas']['lsc_acc']
    wcdas_mmf = val_results['wcdas']['mmf']
    wcdas_mmf_lsc = val_results['wcdas']['mmf_lsc']
    wcdas_gap = val_results['wcdas']['lsc_gap']
    
    ce_acc = val_results['ce']['acc']
    ce_lsc = val_results['ce']['lsc_acc']
    ce_mmf = val_results['ce']['mmf']
    ce_mmf_lsc = val_results['ce']['mmf_lsc']
    ce_gap = val_results['ce']['lsc_gap']
    
    # 更新最佳准确率
    if wcdas_acc > self.best_accuracy:
        self.best_accuracy = wcdas_acc
    
    # 保存历史
    self.accuracies_history.append({
        'epoch': epoch,
        'wcdas_acc': wcdas_acc,
        'wcdas_lsc': wcdas_lsc,
        'ce_acc': ce_acc,
        'ce_lsc': ce_lsc
    })
    
    # ========== 输出格式化日志 ==========
    print(f"\n{'='*70}")
    print(f"Epoch {epoch:3d} - Dual Evaluation Results")
    print(f"{'='*70}")
    print(f"Test Loss: {test_loss:.4f}\n")
    
    # WCDAS结果
    print(f"🔷 WCDAS Method (tau={getattr(self.config, 'tau', 1.5)})")
    print(f"   Raw Acc:      {100*wcdas_acc:6.2f}%")
    print(f"   + LSC:        {100*wcdas_lsc:6.2f}%  (Gap: {100*wcdas_gap:+.2f}%)")
    print(f"   MMF (Raw):    {[f'{x:.1f}' for x in wcdas_mmf]}")
    print(f"   MMF (LSC):    {[f'{x:.1f}' for x in wcdas_mmf_lsc]}")
    
    # CE结果
    print(f"\n🔶 Standard CE Method (no tau)")
    print(f"   Raw Acc:      {100*ce_acc:6.2f}%")
    print(f"   + LSC:        {100*ce_lsc:6.2f}%  (Gap: {100*ce_gap:+.2f}%)")
    print(f"   MMF (Raw):    {[f'{x:.1f}' for x in ce_mmf]}")
    print(f"   MMF (LSC):    {[f'{x:.1f}' for x in ce_mmf_lsc]}")
    
    # 对比分析
    print(f"\n📊 Comparison Analysis:")
    diff_raw = wcdas_acc - ce_acc
    diff_lsc = wcdas_lsc - ce_lsc
    gap_reduction = (ce_gap - wcdas_gap) / ce_gap * 100 if ce_gap > 0 else 0
    
    print(f"   WCDAS vs CE (Raw):    {100*diff_raw:+.2f}%  {'✅ Better' if diff_raw > 0 else '❌ Worse'}")
    print(f"   WCDAS vs CE (LSC):    {100*diff_lsc:+.2f}%")
    print(f"   LSC Gap Reduction:     {gap_reduction:.1f}%  {'✅ Effective' if gap_reduction > 60 else '⚠️ Limited'}")
    
    if gap_reduction > 60:
        print(f"   💡 WCDAS successfully reduced bias!")
    elif gap_reduction > 30:
        print(f"   ⚠️  WCDAS partially reduced bias, consider adjusting tau")
    else:
        print(f"   ❌ WCDAS ineffective, tau adjustment not working")
    
    print(f"{'='*70}\n")
    
    # 记录到日志文件
    logging.info(f"Epoch {epoch} - WCDAS: {100*wcdas_acc:.2f}% (LSC: {100*wcdas_lsc:.2f}%), "
                f"CE: {100*ce_acc:.2f}% (LSC: {100*ce_lsc:.2f}%)")
    logging.info(f"Gap Reduction: {gap_reduction:.1f}%")
```

---

## 5. 预期输出示例

### 5.1 训练日志输出

```
======================================================================
Epoch 110 - Dual Evaluation Results
======================================================================
Test Loss: 1.4297

🔷 WCDAS Method (tau=1.5)
   Raw Acc:       76.86%
   + LSC:         77.20%  (Gap: +0.34%)
   MMF (Raw):    ['81.7', '56.9', '58.5']
   MMF (LSC):    ['82.1', '68.2', '70.5']

🔶 Standard CE Method (no tau)
   Raw Acc:       74.23%
   + LSC:         77.50%  (Gap: +3.27%)
   MMF (Raw):    ['79.8', '52.3', '51.2']
   MMF (LSC):    ['81.2', '66.5', '68.9']

📊 Comparison Analysis:
   WCDAS vs CE (Raw):    +2.63%  ✅ Better
   WCDAS vs CE (LSC):    -0.30%
   LSC Gap Reduction:     89.6%  ✅ Effective
   💡 WCDAS successfully reduced bias!
======================================================================
```

### 5.2 最终训练总结

```
Training Complete - Strategy A with Dual Evaluation

Best WCDAS Accuracy:     76.86% (Epoch 110)
Best WCDAS + LSC:        77.20% (Epoch 110)
Best CE Accuracy:        74.23% (Epoch 110)
Best CE + LSC:           77.50% (Epoch 110)

Key Findings:
✅ WCDAS improved raw accuracy by 2.63% over CE
✅ LSC gap reduced by 89.6% (3.27% → 0.34%)
✅ tau=1.5 effectively internalized bias correction
```

---

## 6. 验证与测试

### 6.1 功能验证清单

- [ ] **基础验证**：代码无语法错误，能正常启动
- [ ] **输出验证**：每个epoch输出4个准确率
- [ ] **数值验证**：WCDAS Acc > CE Acc（正常情况）
- [ ] **LSC验证**：CE的LSC Gap > WCDAS的LSC Gap
- [ ] **MMF验证**：输出Many/Medium/Few准确率
- [ ] **保存验证**：生成两个最佳模型文件

### 6.2 预期行为

**正常情况**：
```
CE Acc < WCDAS Acc < CE+LSC ≈ WCDAS+LSC
74.23% < 76.86% < 77.50% ≈ 77.20%
```

**异常情况（需要调查）**：
```
情况1: WCDAS Acc ≤ CE Acc
      → tau设置可能有问题

情况2: WCDAS+LSC >> WCDAS Acc (Gap仍很大)
      → tau调整未生效

情况3: CE+LSC >> WCDAS+LSC
      → 训练可能有问题
```

### 6.3 测试步骤

1. **修改代码**（按上述方案）
2. **运行1个epoch** - 检查输出格式
3. **运行10个epoch** - 观察趋势
4. **完整训练** - 获取最终结果

---

## 7. FAQ

**Q1: 训练Loss使用哪个？**
A: 使用WCDAS logits计算loss（保持训练不变）

**Q2: 保存哪个模型？**
A: 根据WCDAS Acc保存最佳模型（主要指标）

**Q3: 会增加多少训练时间？**
A: 验证阶段增加约10-15%（一次forward，两次softmax）

**Q4: 如果只想看WCDAS结果？**
A: 可以只输出WCDAS相关指标，CE作为参考

**Q5: LSC对两种方法用同样参数？**
A: 是的，都使用训练集分布，保证公平对比

**Q6: 能否只在特定epoch评估？**
A: 可以，添加条件：`if epoch % 10 == 0: 双重评估`

**Q7: CE和WCDAS差异不大怎么办？**
A: 增大tau值（如1.5→2.0）或检查WCDAS实现

---

## 8. 实施检查表

### 准备工作
- [ ] 备份现有代码
- [ ] 阅读本文档
- [ ] 理解4指标含义

### 代码修改
- [ ] 修改 `loss_calculator.py`（返回双logits）
- [ ] 修改 `trainer.py` （_validate方法）
- [ ] 修改 `trainer.py` （调用处）
- [ ] 修改 `training_monitor.py` （新增log方法）

### 测试验证
- [ ] 代码无语法错误
- [ ] 运行1个epoch测试
- [ ] 检查输出格式
- [ ] 验证数值合理性

### 正式训练
- [ ] 开始完整训练
- [ ] 监控LSC Gap变化
- [ ] 保存结果分析

---

## 附录：快速参考

### 核心公式

```python
# 1. CE Logits
logit_ce = normalized_fc(features)

# 2. WCDAS Logits  
logit_wcdas = logit_ce * gamma + tau * log_prior

# 3. LSC修正
prob_lsc = prob * (q_y / p_y)  # q_y=uniform, p_y=train_dist
```

### 关键指标

```
LSC Gap = LSC_Acc - Raw_Acc
Gap Reduction = (CE_Gap - WCDAS_Gap) / CE_Gap * 100%
```

### 判断标准

```
Excellent:  Gap Reduction > 80%
Good:       Gap Reduction > 60%
Fair:       Gap Reduction > 30%
Poor:       Gap Reduction < 30%  → 需要调整tau
```

---

**文档版本**: v1.0  
**最后更新**: 2025-12-02  
**作者**: Strategy A Team
