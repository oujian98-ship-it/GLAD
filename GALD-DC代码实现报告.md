# GALD-DC代码实现报告

## 符号与整体预先设定

**Idea 内容：**
- 训练集：$D = (x_i, y_i)_{i=1}^N$，$y_i \in Y = \{1, \dots, K\}$。
- 头/尾部类划分：$C_{head} = \{y | n_y \geq \tau\}$, $C_{tail} = \{y | n_y < \tau\}$。
- 模型组件：编码器 $E(\cdot; \theta_E)$，分类头 $G(\cdot; \theta_G)$，噪声预测网络 $\epsilon_\theta(z_t, t, y)$。
- 统计量：类原型 $\mu_y$，观测半径 $r_{obs}$，校准半径 $r_{cal}$，头部全局先验半径 $r_{prior}$。

**代码：**
- **潜空间数据集Dz**：train_set(DataLoader)。[gald_dc/trainer.py:L48](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L48)
- **类别划分**：head_mask = class_counts >= tau。[gald_dc/loss_calculator.py:L604](file:///e:/projects/LDMLR-main/gald_dc/loss_calculator.py#L604)
- **模型实现**：encoder, classifier, diffusion_model。[gald_dc/trainer.py:L66-72](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L66-72)
- **统计量变量**：class_mu , r_obs ， r_cal , r_prior。[gald_dc/trainer.py:L93-113](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L93-113)

**代码片段：**

```python
# gald_dc/trainer.py & loss_calculator.py
#潜空间数据集构造
train_set, test_set, num_classes, dataset_info = self._load_data(cfg)
# 类别划分 Chead / Ctail
head_mask = class_counts >= tau  # C_head
tail_mask = class_counts < tau   # C_tail
# 模型组件初始化
encoder, classifier, diffusion_model, _ = self.model_manager.initialize_models(...)
# 统计量变量
class_mu = self._initialize_class_mu(num_classes, feature_dim) # μy
r_obs = self._compute_r_obs_from_real_features(...) # robs
r_prior = self.loss_calculator.compute_head_class_prior(...) # rprior
```

---

## 1. Stage 1：表征学习 + 头部先验统计

### 1.1 编码器 + 分类头训练 ($L_{CE}$)
**Idea 内容：**
- 训练 $\theta_E, \theta_G$ 最小化交叉熵损失：$L_{CE} = -E(x,y) \sim D [\log G(E(x; \theta_E); \theta_G)_y]$。
- 更新规则：$(\theta_E, \theta_G) \leftarrow (\theta_E, \theta_G) - \eta \nabla_{\theta_E, \theta_G} L_{CE}$。

**代码：**
- **损失计算**：F.cross_entropy(real_logits, labels)。[gald_dc/loss_calculator.py:L130](file:///e:/projects/LDMLR-main/gald_dc/loss_calculator.py#L130)
- **更新规则**：Stage 1 优化器包含 encoder 和 classifier 参数。[gald_dc/trainer.py:L77-81](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L77-81)

**代码片段：**
```python
# gald_dc/loss_calculator.py 损失计算
real_logits = classifier(real_features)
loss = F.cross_entropy(real_logits, labels, label_smoothing=0.1)

# gald_dc/trainer.py 更新规则
optimizer = optim.SGD([
    {'params': encoder.parameters(), 'lr': self.config.lr * 0.01},
    {'params': classifier.parameters(), 'lr': self.config.lr},
    {'params': diffusion_model.parameters(), 'lr': self.config.lr}
], momentum=0.9, nesterov=True, weight_decay=self.config.weight_decay)
```

### 1.1.2 潜空间数据集构造 ($D_z$)
**Idea 内容：**
- 训练收敛后，冻结 $\theta_E^{(0)}$ 用于后续 Stage 2。
- 构造潜空间数据集：$D_z = \{(z_i, y_i)_{i=1}^N, z_i = E^{(0)}(x_i)\}$。
- 【解释】：对样本 $x_i$ 做前向传播得到潜空间特征向量 $z_i$，收集起来形成 latent 数据集。

**代码：**

- **代码位置**：[gald_dc/trainer.py:L282-284](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L282-284)
- **实现方式**：代码采用 **On-the-fly** 提取。在 Stage 2 中，通过 with torch.no_grad(): real_features = encoder.forward_no_fc(inputs) 实时生成 $z_i$，在逻辑上等价于构造了 $D_z$。

**代码片段：**
```python
# gald_dc/trainer.py
# Stage 2 & Stage 3-S: Encoder 冻结，动态提取潜空间特征
with torch.no_grad():
    real_features = encoder.forward_no_fc(inputs) # 提取逻辑上的 Dz
```

### 1.2 头部类全局半径先验 ($r_{prior}$)
**Idea 内容：**
- 估计“健康类内多样性”：$\mu_k = \text{mean}(z_i)$, $r_k = \text{mean}(\|z_i - \mu_k\|_2)$。
- 全局先验：$r_{prior} = \frac{1}{|C_{head}|} \sum_{k \in C_{head}} r_k$。

**代码：**
- **类内半径计算**：[gald_dc/trainer.py:L894-924](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L894-924) (_compute_r_obs_from_real_features)。
- **全局先验计算**：[gald_dc/loss_calculator.py:L616-636](file:///e:/projects/LDMLR-main/gald_dc/loss_calculator.py#L616-636) (compute_head_class_prior)。

**代码片段：**
```python
# gald_dc/trainer.py 类内半径计算 (rk)
dists = torch.norm(cls_feats - cls_mu[cls_idx], p=2, dim=1)
rk = dists.mean()

# gald_dc/loss_calculator.py 全局先验计算 (rprior)
r_prior = observed_radii[head_mask].mean().item()
```

---

## 2. Stage 2：几何感知 + 分布校准 + 判别 margin 的潜空间扩散

### 2.1 正向加噪
**Idea 内容：**

- 噪声调度 $\{\alpha_t\}$，定义 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。
- 对任意 $(z_0, y)$，加噪：$z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$。

**代码：**

- **代码位置**：[gald_dc/loss_calculator.py:L310-L318](file:///e:/projects/LDMLR-main/gald_dc/loss_calculator.py#L318) (q_sample)

**代码片段：**

```python
# gald_dc/loss_calculator.py
# zt = sqrt(alphabar)*z0 + sqrt(1-alphabar)*eps
# 1. 生成时间步 t (如果未传入)
        if t is None:
            t = torch.randint(0, self.config.diffusion_steps, (b,), device=device)
        # 2. 生成噪声
        noise = torch.randn_like(x_start)
        # 3. 前向加噪 (q_sample) - 只加一次
        # 使用 diffusion_model 内部的 q_sample，它会正确使用 sqrt_alphas_cumprod
        x_noisy = diffusion_model.q_sample(x_start=x_start, t=t, noise=noise)
```

### 2.2 噪声预测损失
**Idea 内容：**

- 噪声预测网络 $\epsilon_\theta(z_t, t, y)$ 尝试重构 $\epsilon$。
- DDIM 损失：$L_{LDM} = E_{z,y,t,\epsilon} [\|\epsilon - \epsilon_\theta(z_t, t, y)\|^2]$。

**代码：**

- **代码位置**：[gald_dc/loss_calculator.py:L333-346](file:///e:/projects/LDMLR-main/gald_dc/loss_calculator.py#L333-346)

**代码片段：**

```python
# gald_dc/loss_calculator.py
# 预测目标 target = noise
model_output = diffusion_model.model(x_noisy, t, labels)
loss = F.mse_loss(model_output, target, reduction='none')
```

### 2.3 去噪估计与 EMA 统计量更新
**Idea 内容：**
- 去噪估计：$\tilde{z}_0(z_t, t, y) = \frac{z_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$。
- 对每个类别 $y$，维护 $\mu_y, r^{obs}_y$。
- EMA 更新：$\mu_y \leftarrow (1-\beta)\mu_y + \beta \tilde{z}_0$，$r^{obs}_y \leftarrow (1-\beta)r_{obs\_y} + \beta \|\tilde{z}_0 - \mu_y\|_2$。

**代码：**
- **代码位置**：[gald_dc/trainer.py:L766-L800](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L766) (去噪估计), [gald_dc/trainer.py:L619-652](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L619-652) (EMA 更新)

**代码片段：**

```python
# gald_dc/trainer.py
#去噪估计
predictions = diffusion_model.model_predictions(noisy_features_reshaped, t, labels)
estimated_clean_features = predictions.pred_x_start.squeeze(1)
return estimated_clean_features

# EMA 更新原型和观测半径
new_proto = (1 - lambda_ema) * old_proto + lambda_ema * cls_mean
new_radius = (1 - beta_radius) * old_radius + beta_radius * current_avg_radius
```

### 2.4 分布校准：校准半径 $r_{cal}$
**Idea 内容：**

- 引入校准半径：$r^{cal}_y = r^{obs}_y$ (if $y \in C_{head}$), $r^{cal}_y = \lambda r^{obs}_y + (1-\lambda) r_{prior}$ (if $y \in C_{tail}$)。
- 作用：防止 Tail 类生成坍塌，用 $r_{prior}$ 撑开分布到与head接近的水平

**代码：**

- **代码位置**：[gald_dc/loss_calculator.py:L577-614](file:///e:/projects/LDMLR-main/gald_dc/loss_calculator.py#L577-614)

**代码片段：**
```python
# gald_dc/loss_calculator.py
# 尾部类混合校准
r_cal[tail_mask] = lambda_cal * observed_radii[tail_mask] + (1 - lambda_cal) * r_prior
```

### 2.5 语义原型约束与校准半径约束
**Idea 内容：**
- 原型约束：$L_{proto} = E [\|\tilde{z}_0 - \mu_y\|^2]$。
- 校准半径约束：$L^{cal}_{rad} = E [(\|\tilde{z}_0 - \mu_y\|_2 - r_{cal\_y})^2]$。

**代码：**

- **代码位置**：[gald_dc/loss_calculator.py:L348-398](file:///e:/projects/LDMLR-main/gald_dc/loss_calculator.py#L348-398),（原型约束） [gald_dc/loss_calculator.py:L400-479](file:///e:/projects/LDMLR-main/gald_dc/loss_calculator.py#L400-479)（校准半径）

**代码片段：**
```python
# gald_dc/loss_calculator.py
# L_proto (MSE)
# 裁剪估计特征，防止数值爆炸
estimated_clean_features = torch.clamp(estimated_clean_features, 
                                             min=-self.config.feature_clamp_max,
                                             max=self.config.feature_clamp_max)
# 裁剪估计特征，防止数值爆炸
clamped_prototype = torch.clamp(class_mu[cls_idx].detach(),
                                              min=-self.config.feature_clamp_max,
                                              max=self.config.feature_clamp_max)
loss_item = F.mse_loss(estimated_clean_features[i], clamped_prototype)

# L_rad
# 计算特征到类中心的欧几里得距离 ||~z_0 - μ_y||_2
distance = torch.norm(feature - class_mu, p=2)

loss_rad = (distance - target_radius) ** 2
```

### 2.6 判别 margin 约束
**Idea 内容：**
- 选取最近负类原型：$y^- = \arg \min_{y' \neq y} \|\tilde{z}_0 - \mu_{y'}\|^2$。
- Margin 损失：$L_{margin} = E [ (m + \|\tilde{z}_0 - \mu_y\|_2^2 - \|\tilde{z}_0 - \mu_{y^-}\|_2^2)_+ ]$。(·)+ 选正部

**代码：**
- **代码位置**：[gald_dc/loss_calculator.py:L502-575](file:///e:/projects/LDMLR-main/gald_dc/loss_calculator.py#L502-575)

**代码片段：**
```python
# gald_dc/loss_calculator.py
# 最近负类距离
dists = torch.cdist(estimated_clean, prototype_matrix, p=2) ** 2

neg_dists = dists[i, neg_mask]
min_neg_dist = neg_dists.min() # 最近负类距离
#Margin 损失
loss_item = F.relu(margin + dist_to_pos - min_neg_dist)
margin_loss = torch.stack(margin_losses).mean()
```

### 2.7 Stage 2 总损失与更新
**Idea 内容：**
- 总损失：$L_{Stage2} = L_{LDM} + \eta_p L_{proto} + \eta_r L_{cal\_rad} + \eta_m L_{margin}$。
- 更新：扩散模型参数 $\theta$、统计量 $\{\mu_y, r^{obs}_y\}$。
- 冻结：$\theta_E^{(0)}, \theta_G^{(0)}$。

**代码：**
- **代码位置**：[gald_dc/trainer.py:L404-411](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L404-411) (总损失), [gald_dc/trainer.py:L232-240](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L232-240) (更新策略)

**代码片段：**
```python
# gald_dc/trainer.py
# 总损失加权求和
L_semantic = (L_ldm +  self.config.eta_p * L_proto + self.config.eta_r * L_rad + self.config.eta_m * L_margin)
# 更新扩散模型，冻结编码器和分类器
elif stage == 2:
            # Stage 2: 冻结 Encoder + Classifier，只训练 Diffusion
            encoder.eval()
            classifier.eval()
            diffusion_model.train()
            for param in encoder.parameters(): param.requires_grad = False
            for param in classifier.parameters(): param.requires_grad = False
            for param in diffusion_model.parameters(): param.requires_grad = True
```

---

## 3. Stage 3：On-the-fly 伪特征驱动的微调

### 3.0 On-the-fly 采样与采样流程
**Idea 内容：**

- 从 $z_T \sim N(0, I)$ 开始迭代去噪得到 $\hat{z}_0 \sim p_\theta(z_0 | y)$。
- 伪特征只在当前 iteration 存在，动态生成。
- 采样公式：$$\hat{z}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{\hat{z}t - \sqrt{1 - \bar{\alpha}t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}} \right)  \sigma_t \epsilon_t$$，得到$\hat{z}_0$。这里发现❌️错误了，之前代码导致没有正常迭代，现在已经更正

**代码：**

- **采样逻辑**：_ddim_sample 函数。[gald_dc/trainer.py:L654-682](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L654-682)
- **动态性保证**：_compute_generation_loss [gald_dc/trainer.py:L470-L556](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L470-L556)在每个批次被调用，不保存静态数据。

**代码片段：**

```python
# gald_dc/trainer.py 采样逻辑与动态生成
def _compute_generation_loss(...):
    # 动态生成 fake_labels 和 fake_features
    fake_labels = torch.multinomial(sample_weights, batch_size, ...)
    fake_features = torch.randn(batch_size, feature_dim, ...)
    # 反向采样过程
    fake_features = self._ddim_sample(...)
#_ddim_sample(...) trainer.py:L654-682
    batch_size = fake_features.size(0)
    estimated_clean = fake_features  
        
    # 严格执行 ddim_steps 次迭代，确保到达 t=0
    for i in range(self.config.ddim_steps):
        current_step = time_steps[i]
        next_step = time_steps[i+1] # 这里的索引最多到 ddim_steps，对应 value 0
        
        fake_features_reshaped = fake_features.unsqueeze(1)
        batched_times = torch.full((batch_size,), current_step, device=self.device, dtype=torch.long)
            
        # 预测干净特征估计 (hat z0)
        predictions = diffusion_model.model_predictions(fake_features_reshaped, batched_times, fake_labels)
        estimated_clean = predictions.pred_x_start.squeeze(1)
        predicted_noise = predictions.pred_noise.squeeze(1)
            
        estimated_clean = torch.clamp(estimated_clean, -self.config.feature_clamp_max, self.config.feature_clamp_max)
            
        # 如果还有下一步 (包括 next_step 为 0)，更新容器特征
        if next_step >= 0:
           alpha_next = sqrt_alphas[next_step]
           sigma = torch.sqrt(max(torch.tensor(0.0, device=self.device), 1 - alpha_next**2))
           fake_features = alpha_next * estimated_clean + sigma * predicted_noise
        
        return estimated_clean
```

### 3.B Stage 3（Hybrid+Consistency 版 H）：受控微调 Encoder
**Idea 内容：**

- 目标：在保持 DM / 几何统计基本对齐的前提下，让 encoder 得到一定程度的“纠偏”，同时利用校准伪特征强化分类边界。
- 引入一份冻结副本 $E^{(0)}$ 仅做参考，以及当前可训练的 $E^{(t)}$。

#### 3.B.1 真实数据损失（更新 Encoder + Classifier）
**Idea 内容：**
- $z_{real}^{(t)} = E^{(t)}(x)$，$L_{real}^{(H)} = E_{(x,y) \sim B_{real}} [\log G(z_{real}^{(t)}; \theta_G)_y]$。
- 梯度流向 $\theta_E^{(t)}, \theta_G$。

**代码：**
- **特征提取**：real_features = encoder.forward_no_fc(inputs)[gald_dc/trainer.py:L273](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L273)
- **损失计算**：L_real计算逻辑。[gald_dc/trainer.py:L416-421](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L416-421)

**代码片段：**

```python
# gald_dc/trainer.py
# Stage 3-H: Encoder 需要梯度
real_features = encoder.forward_no_fc(inputs)
# 计算真实数据分类损失
L_real = self.loss_calculator.compute_real_loss(classifier, real_features, labels, ...)
```

#### 3.B.2 特征一致性正则 (Feature Consistency)
**Idea 内容：**
- 使用冻结编码器 $E^{(0)}$ 给出的“旧特征”作为锚点：$z_{real}^{(0)} = E^{(0)}(x)$。
- 一致性损失：$L_{cons} = E_{x \sim B_{real}} [\|E^{(t)}(x) - \text{detach}(E^{(0)}(x))\|_2^2]$。

**代码：**
- **锚点提取**：frozen_features = self.frozen_encoder.forward_no_fc(inputs)。[gald_dc/trainer.py:L275](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L275)
- **损失实现**：compute_consistency_loss，[gald_dc/loss_calculator.py:L638-656](file:///e:/projects/LDMLR-main/gald_dc/loss_calculator.py#L638-656)

**代码片段：**

```python
# gald_dc/trainer.py & loss_calculator.py
# 冻结特征作为锚点
with torch.no_grad():
    frozen_features = self.frozen_encoder.forward_no_fc(inputs)
# 计算一致性损失 Lcons
L_cons = self.loss_calculator.compute_consistency_loss(real_features, frozen_features)
```

#### 3.B.3 伪特征损失（仅更新分类头）
**Idea 内容：**
- 与稳定版类似，对伪特征只更新 $\theta_G$。
- $L_{ge}^{(H)} = -E_{(\hat{z},y) \sim B_z} [\log G(\hat{z}_{det}; \theta_G)_y]$。$\hat{z}_{det} = \text{detach}(\hat{z})$ 不对 $\theta$ 和 $\theta_E$ 反传。

**代码：**
- **实现方式**：在 _compute_generation_loss中使用 fake_features.detach()。[gald_dc/trainer.py:L538-551](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L538-551)

**代码片段：**
```python
# gald_dc/trainer.py
# 对伪特征阻断梯度，确保只更新分类头 θG
fake_logits = classifier(input=fake_features.detach(), ...)
L_ge = F.cross_entropy(fake_logits, fake_labels)
```

### 3.B.4 Hybrid+Consistency 总损失与更新
**Idea 内容：**

- 总损失：$L_{Stage3}^{(H)} = L_{real}^{(H)} + \gamma L_{ge}^{(H)} + \beta L_{cons}$。
- 梯度更新规则：
  - 对分类头：$\theta_g \leftarrow \theta_g - \eta_g \nabla_{\theta_g} L_{Stage3}^{(H)}$
  - 对编码器：$\theta_\epsilon^{(t)} \leftarrow \theta_\epsilon^{(t)} - \eta_\epsilon \nabla_{\theta_\epsilon^{(t)}} (L_{real}^{(H)} + \beta L_{cons})$
  - 扩散模型完全冻结：$\nabla_\theta L_{Stage3}^{(H)} = 0$
- 梯度关系：$\nabla_{\theta_G} L_{Stage3}^{(H)} \neq 0$, $\nabla_{\theta_E} L_{Stage3}^{(H)} = \nabla_{\theta_E} (L_{real}^{(H)} + \beta L_{cons})$, $\nabla_\theta L_{Stage3} = 0$。

**代码：**

- **代码位置**：[gald_dc/trainer.py:L440-443](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L440-443) (总损失)，[gald_dc/trainer.py:L243-251](file:///e:/projects/LDMLR-main/gald_dc/trainer.py#L243-251) （更新策略)

**代码片段：**
```python
# gald_dc/trainer.py
# 总损失计算
total_loss = (L_real + gamma_pseudo * L_ge + beta_cons * L_cons)

# (L243-251) 梯度更新策略
if self.config.stage3_mode == 'hybrid':
    encoder.train(); classifier.train(); diffusion_model.eval()
    for param in encoder.parameters(): param.requires_grad = True # 更新 θE
    for param in classifier.parameters(): param.requires_grad = True # 更新 θG
    for param in diffusion_model.parameters(): param.requires_grad = False # 冻结 θ

# 通过 .detach()，将生成特征从计算图中剥离
# 这样 L_ge 的梯度只能回传给分类器 θg，无法穿透回 Encoder θE
fake_logits = classifier(input=fake_features.detach(), ...) #L552
L_ge = F.cross_entropy(fake_logits, fake_labels)
```
