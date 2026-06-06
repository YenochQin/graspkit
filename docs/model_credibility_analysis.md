# 迭代式主动学习模型可信度分析

> **分析日期**: 2026-02-27
> **代码版本**: graspkit 2.9.1
> **分析范围**: `ml_trainer.py`, `ml_initializer.py`, `neural_network.py`

---

## 目录

- [一、当前实现的工作机制](#一当前实现的工作机制)
- [二、模型可信度问题诊断](#二模型可信度问题诊断)
- [三、改进建议](#三改进建议)
- [四、可信度量化评估](#四可信度量化评估)
- [五、快速验证建议](#五快速验证建议)
- [六、结论](#六结论)

---

## 一、当前实现的工作机制

### 1.1 数据增长模式

**累积式数据累积**（`ml_initializer.py:462-577`）：

```
Loop 1:
  Training Data = Current Loop CSFs Only
  Size: cal_csfs_count (如 8,500)

Loop 2+:
  Training Data = Loop 1 ∪ Loop 2 ∪ ... ∪ Loop N
  Size: 累积并集（单调递增）
  Merge策略: 索引取并集，CI系数以最新轮次为准
```

**具体实现**：

```python
# merge_historical_ci_data() 函数 (ml_initializer.py:462-519)
merged_idxs = np.union1d(previous_idxs, current_idxs)

# 以转置视图 (n_merged, n_levels) 按行赋值
merged_ci_t = np.empty((len(merged_idxs), previous_ci_squared.shape[0]))
merged_ci_t[prev_pos] = previous_ci_squared.T
merged_ci_t[curr_pos] = current_ci_squared.T  # 当前数据覆盖交集
merged_ci_squared = merged_ci_t.T
```

**数据流向**：

```
Loop N-1 累积数据
    ↓ (加载 accumulated_idxs_ci_path.npz)
合并历史数据 (merge_historical_ci_data)
    ↓
Loop N 累积数据
    ↓ (保存 accumulated_idxs_ci_path.npz)
供 Loop N+1 使用
```

### 1.2 模型加载与训练

**当前实现**（`ml_trainer.py:94-154`）：

```python
if config.cal_settings.cal_loop_num == 1:
    # 第一轮：创建新模型
    model = ANNClassifier(
        input_size=X_train.shape[1],
        output_size=n_correct_levels,
        hidden_size=96 if not torch.cuda.is_available() else 128,
        learning_rate=0.001,
        class_weights=class_weights,
        model_architecture="tensornet",
    )
    logger.info("创建新模型")
else:
    # 后续轮次：加载历史模型
    model_path = (
        config.cal_path.models_path
        / f"{config.target.conf}_{config.cal_settings.cal_loop_num - 1}.pkl"
    )
    model = joblib.load(model_path)
    logger.info(f"加载已有模型: {model_path}")

# 但是！训练时完全相同：
model.fit(X_resampled, y_resampled,
           batch_size=4096,
           max_epochs=150)  # 始终150 epochs
```

**关键问题**：
- 加载的模型仅作为**初始权重起点**
- 训练参数与第一轮**完全相同**
- 没有针对warm-start的特殊处理

### 1.3 CSF采样策略

**智能动态选择机制**（`ml_trainer.py:499-575`）：

```python
# 1. 计算当前重要组态数量
current_important_count = len(verified_important_idxs)

# 2. 获取扩展比例
expansion_ratio = getattr(config.cal_settings, "expansion_ratio", 2)
new_sampling_CSFs_num = math.ceil(expansion_ratio * current_important_count)

# 3. 在未选择CSF中预测
X_unselected_for_prediction = raw_csfs_descriptors[unselected_idxs]
y_unselected_probability = model.predict_proba_batch(
    X_unselected_for_prediction,
    batch_size=10_000_000,
)

# 4. 选择策略
if len(ml_predicted_important_local_idxs) >= new_sampling_CSFs_num:
    # 情况1：ML预测充足，按概率排序选择top-k
    ml_predicted_important_probabilities = np.max(
        y_unselected_probability[ml_predicted_important_local_idxs], axis=1
    )
    probability_sorted_idxs = np.argsort(ml_predicted_important_probabilities)[::-1]
    ml_sampled_idxs = unselected_idxs[probability_sorted_idxs[:new_sampling_CSFs_num]]
else:
    # 情况2：ML预测不足，全部采用
    ml_sampled_idxs = ml_predicted_important_global_idxs
```

---

## 二、模型可信度问题诊断

### 2.1 第一轮模型：可信度极低

**数据特征**（`ml_trainer.py:76-83`）：

```python
# 从日志中看到的典型数据
positive_count = 50-200      # 正样本数
negative_count = 8000+        # 负样本数
avg_positive_ratio = 0.5%-2.0%  # 正样本比例
```

**问题**：

1. **极端不平衡**：1:50~1:100的比例，模型很难学到正类特征
2. **样本量不足**：50-200个正样本不足以训练稳定特征
3. **预测偏向**：模型可能倾向预测负类（损失最小化策略）

**实际表现**（从`ml_trainer.py:247-295`的阈值调整逻辑可以看出）：

```python
# 预测正样本过多时（超过真实3倍）→ 提高阈值
if predicted_positive_ratio > positive_ratio * 3:
    logger.warning("预测正样本过多，尝试提高阈值")
    # ... 自动搜索最优阈值

# 没有任何正样本时 → 降低阈值
elif np.sum(y_prediction) == 0:
    logger.warning("模型没有预测任何正样本，尝试使用自适应阈值")
    threshold_percentile = 90
    adaptive_threshold = np.percentile(y_probability, threshold_percentile)
```

**结论**：这说明模型要么预测过度，要么预测不足，都不稳定。

### 2.2 Warm-start机制效果有限

**当前实现的问题**：

| 方面 | 当前实现 | 应有的增量学习实践 |
|------|---------|-----------------|
| 学习率 | 固定0.001 | 应衰减至0.0005-0.0001 |
| 训练轮数 | 固定150 epochs | 应减少至50-80 epochs |
| 知识保护 | 无 | 应有正则化或知识蒸馏 |
| 收敛判断 | 仅基于loss | 应监控历史知识保留 |

**为什么效果有限？**

```python
# neural_network.py:258-371 - fit() 方法
def fit(self, X_train, y_train, ...):
    # 无论是否warm-start，都执行相同流程：
    for epoch in range(max_epochs):  # 150轮
        loss = self._train_epoch(...)  # 完整训练
        # 没有特殊的warm-start逻辑
        # 没有历史知识保护机制
```

相当于每次都"半重新训练"，历史权重可能被覆盖。

### 2.3 类别权重处理

**当前实现**（`neural_network.py:424-543`）：

```python
def _calculate_dynamic_weights(self, y_train, ...):
    # 针对"找出重要组态"任务优化的权重计算
    pos_counts = y_tensor.sum(dim=0)
    neg_counts = n_samples - pos_counts
    pos_ratios = pos_counts / n_samples

    # 自适应权重逻辑
    if focus_on_recall:  # 默认True
        # 情况A: 早期迭代/少数类 (Ratio <= 0.5)
        if pos_ratios <= 0.5:
            raw_weight = 1.0 / (pos_ratios + 1e-6)
            calculated_weights = 1.0 + (raw_weight - 1.0) * adaptive_strength

        # 情况B: 后期迭代/多数类 (Ratio > 0.5)
        else:
            decay_factor = (pos_ratios - 0.5) / 0.5
            current_weights = min_positive_weight * (1.0 - 0.3 * decay_factor)
            calculated_weights = torch.max(current_weights, min_positive_weight)
```

**评估**：
- ✅ 设计合理：考虑了早期/后期迭代的不同策略
- ⚠️ 早期轮次仍极端：即使有自适应，50:1的比例仍可能导致训练不稳定
- ⚠️ 缺乏样本级权重：只有类别级权重，没有对历史/当前数据的差异化处理

### 2.4 数据累积的正面作用

**有效方面**：
- Loop 3之后，训练数据包含数千~数万个CSF
- 正样本数量累积增长（尽管比例仍低）
- 特征空间覆盖更全面

**但仍有问题**：
- 负样本累积更快，不平衡可能加剧
- 没有对历史数据的"重要性加权"
- 缺乏对重复样本的处理策略

---

## 三、改进建议

### 3.1 立即可实施的改进（最小改动）

#### A. 动态调整训练参数

**位置**：`ml_trainer.py:94-154`

```python
# 添加动态参数计算
if config.cal_settings.cal_loop_num == 1:
    max_epochs = 150
    learning_rate = 0.001
    logger.info("第一轮训练：初始模式")
else:
    # 后续轮次：微调而非重训
    # 根据轮次衰减epochs和学习率
    decay_factor = max(0.5, 1.0 / (config.cal_settings.cal_loop_num - 1))
    max_epochs = max(50, int(150 * decay_factor))
    learning_rate = 0.001 * decay_factor  # 衰减学习率

    logger.info(f"Warm-start模式 (Loop {config.cal_settings.cal_loop_num}): "
                f"epochs={max_epochs}, lr={learning_rate:.6f}, decay={decay_factor:.2f}")

# 传递给模型创建
model = ANNClassifier(
    ...,
    learning_rate=learning_rate,
)
```

#### B. 添加模型质量评估

**位置**：`ml_trainer.py:112-138`

```python
# 在加载模型后，评估初始性能
if model_path.exists():
    model = joblib.load(model_path)

    # 快速评估加载模型的初始性能
    logger.info("评估加载模型的初始性能...")

    # 使用测试集快速评估（避免过拟合到训练集）
    X_quick_test = X_test[:min(500, len(X_test))]
    y_quick_test = y_test[:min(500, len(y_test))]

    model.eval()
    with torch.no_grad():
        initial_probs = model.predict_proba(X_quick_test)
        initial_preds = (initial_probs > 0.5).astype(int)

        # 计算快速指标
        from sklearn.metrics import f1_score
        initial_f1 = f1_score(y_quick_test, initial_preds, average='micro')
        initial_recall = (initial_preds == 1).sum() / (y_quick_test == 1).sum()

    logger.info(f"加载模型初始F1: {initial_f1:.4f}, Recall: {initial_recall:.4f}")

    # 如果模型质量太差，重新训练
    quality_threshold = getattr(config.ml_config, "warm_start_quality_threshold", 0.3)
    if initial_f1 < quality_threshold:
        logger.warning(
            f"历史模型质量较差 (F1={initial_f1:.4f} < {quality_threshold})，"
            f"重新创建模型"
        )
        model = ANNClassifier(
            input_size=X_train.shape[1],
            output_size=n_correct_levels,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            class_weights=class_weights,
            model_architecture="tensornet",
        )
    else:
        logger.info(f"历史模型质量可接受，继续warm-start训练")
```

#### C. 历史数据重要性加权

**位置**：`ml_initializer.py:582-657`（`generate_train_csfs_descriptors`函数）

```python
def generate_train_csfs_descriptors(
    config: MLCalConfig,
    raw_csfs_descriptors: NDArray[np.float64],
    current_loop_idxs: NDArray[np.int64],  # 新增参数：当前轮次索引
    logger: logging.Logger
) -> np.ndarray:
    # ... 现有代码 ...

    # 区分历史数据和当前数据
    is_current = np.isin(accumulated_idxs, current_loop_idxs)
    is_historical = ~is_current

    logger.info(f"训练数据构成：当前轮次={np.sum(is_current)}, "
                f"历史数据={np.sum(is_historical)}")

    # 应用样本权重（在训练时使用）
    # 当前数据权重=1.0，历史数据权重=0.5~0.7
    historical_weight = getattr(config.ml_config, "historical_weight", 0.7)
    current_weight = 1.0

    sample_weights = np.where(
        is_current, current_weight, historical_weight
    )

    # 保存权重供fit()使用
    np.save(
        config.cal_path.cal_loop_path / f"{config.cal_path.loop_file_name}_sample_weights.npy",
        sample_weights
    )

    # ... 其余代码不变 ...
```

**修改`train_model()`以使用样本权重**：

```python
# 在 ml_trainer.py 中
# 加载样本权重
sample_weights_path = (
    config.cal_path.cal_loop_path
    / f"{config.cal_path.loop_file_name}_sample_weights.npy"
)

if sample_weights_path.exists():
    sample_weights = np.load(sample_weights_path)
    logger.info(f"使用样本权重：当前数据均值={sample_weights[is_current].mean():.3f}, "
                f"历史数据均值={sample_weights[is_historical].mean():.3f}")
else:
    sample_weights = None

# 修改模型训练
model.fit(
    X_resampled,
    y_resampled,
    sample_weights=sample_weights,  # 添加样本权重
    batch_size=batch_size_optimized,
    max_epochs=max_epochs_optimized,
)
```

### 3.2 中期改进（需要重构）

#### A. 知识蒸馏防止遗忘

**新建模块**：`ml_module/distillation.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    结合任务损失和蒸馏损失，防止catastrophic forgetting
    """

    def __init__(self, alpha=0.5, temperature=2.0):
        super().__init__()
        self.alpha = alpha  # 任务损失权重
        self.temperature = temperature  # 蒸馏温度

    def forward(self, student_logits, teacher_logits, y_true):
        # 学生模型的任务损失
        task_loss = nn.BCEWithLogitsLoss()(student_logits, y_true)

        # 教师模型的软目标
        with torch.no_grad():
            teacher_probs = torch.sigmoid(teacher_logits / self.temperature)
        student_probs = torch.sigmoid(student_logits / self.temperature)

        # KL散度损失（蒸馏损失）
        distill_loss = F.kl_div(
            student_probs.log(),
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # 组合损失
        return self.alpha * task_loss + (1 - self.alpha) * distill_loss
```

**修改训练循环**（在`neural_network.py:373-400`）：

```python
# 在 _train_epoch() 中添加蒸馏支持
def _train_epoch_with_distillation(
    self, X_train, y_train, batch_size,
    teacher_model=None, distillation_criterion=None
) -> float:
    """训练一个epoch（带知识蒸馏）"""
    self.model.train()
    total_loss = 0.0
    num_batches = 0

    permutation = torch.randperm(X_train.size(0))

    for i in range(0, X_train.size(0), batch_size):
        idxs = permutation[i : i + batch_size]
        batch_X, batch_y = X_train[idxs], y_train[idxs]

        self.optimizer.zero_grad()
        student_outputs = self.model(batch_X)

        if teacher_model is not None and distillation_criterion is not None:
            # 知识蒸馏模式
            with torch.no_grad():
                teacher_outputs = teacher_model(batch_X)

            loss = distillation_criterion(
                student_outputs, teacher_outputs, batch_y
            )
        else:
            # 标准训练模式
            loss = self.criterion(student_outputs, batch_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches
```

#### B. Uncertainty-Based主动采样

**新建函数**：`ml_module/uncertainty_sampling.py`

```python
import numpy as np
from scipy.stats import entropy

def uncertainty_sampling(model, X_candidates, budget, method='entropy'):
    """
    基于不确定性的主动采样

    Args:
        model: 训练好的模型
        X_candidates: 候选样本特征 (n_candidates, n_features)
        budget: 需要采样的样本数量
        method: 不确定性计算方法 ('entropy', 'margin', 'least_confident')

    Returns:
        selected_idxs: 选中样本的索引
    """
    # 计算所有候选样本的预测概率
    probs = model.predict_proba(X_candidates)  # (n_candidates, n_labels)

    if method == 'entropy':
        # 熵不确定性：越高的熵表示越不确定
        entropies = np.zeros(len(probs))
        for i in range(len(probs)):
            entropies[i] = entropy(probs[i])

        # 选择熵最高的样本
        selected_idxs = np.argsort(entropies)[-budget:]

    elif method == 'margin':
        # 边距不确定性：预测概率最高和第二高的差距
        sorted_probs = np.sort(probs, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        selected_idxs = np.argsort(margins)[:budget]

    elif method == 'least_confident':
        # 最置信不确定性：选择概率最接近0.5的样本
        confidences = np.abs(probs[:, 1] - 0.5)
        selected_idxs = np.argsort(confidences)[-budget:]

    else:
        raise ValueError(f"Unknown method: {method}")

    return selected_idxs
```

**修改`predict_model()`**（`ml_trainer.py:537-565`）：

```python
# 添加配置项
use_uncertainty_sampling = getattr(
    config.ml_config, "use_uncertainty_sampling", False
)

if len(ml_predicted_important_local_idxs) >= new_sampling_CSFs_num:
    if use_uncertainty_sampling:
        # 不确定性采样策略
        logger.info("使用不确定性采样策略")

        # 从ML预测重要的样本中选择高不确定性样本
        uncertain_idxs = uncertainty_sampling(
            model,
            X_unselected_for_prediction[ml_predicted_important_local_idxs],
            budget=new_sampling_CSFs_num,
            method='entropy'
        )

        ml_sampled_idxs = ml_predicted_important_local_idxs[uncertain_idxs]
    else:
        # 原有的概率排序策略
        logger.info("使用概率排序策略")

        ml_predicted_important_probabilities = np.max(
            y_unselected_probability[ml_predicted_important_local_idxs],
            axis=1
        )
        probability_sorted_idxs = np.argsort(
            ml_predicted_important_probabilities
        )[::-1]
        top_k_local_idxs = ml_predicted_important_local_idxs[
            probability_sorted_idxs[:new_sampling_CSFs_num]
        ]
        ml_sampled_idxs = unselected_idxs[top_k_local_idxs]
else:
    # ML预测组态不足，全部采用
    ml_sampled_idxs = ml_predicted_important_global_idxs
```

### 3.3 长期改进（架构级）

#### A. 使用PEFT/LoRA适配器

**优势**：
- 训练参数量减少90-99%
- 防止catastrophic forgetting（冻结基础模型）
- 支持多任务适配器切换

**实现示例**：

```python
from peft import LoraConfig, get_peft_model

# 创建LoRA配置
lora_config = LoraConfig(
    r=8,  # 低秩维度（1%训练参数）
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["linear"],  # 可训练层
    bias="none",
)

# 包装模型
model = get_peft_model(base_model, lora_config)
logger.info(f"使用PEFT: 可训练参数量 = {model.num_parameters(only_trainable=True)}")

# 训练时只更新LoRA参数
for name, param in model.named_parameters():
    if param.requires_grad:
        logger.info(f"可训练: {name}, shape={param.shape}")
```

#### B. Replay Buffer机制

**新建模块**：`ml_module/replay_buffer.py`

```python
import numpy as np
from collections import deque

class ReplayBuffer:
    """
    Reservoir Sampling Replay Buffer
    保持历史代表性样本，防止遗忘
    """

    def __init__(self, capacity=10000, seed=42):
        self.capacity = capacity
        self.rng = np.random.RandomState(seed)
        self.buffer = deque(maxlen=capacity)
        self.sample_weights = deque(maxlen=capacity)
        self.added_count = 0

    def add(self, X_new, y_new, sample_weight=1.0):
        """
        Reservoir采样：保持代表性分布

        Args:
            X_new: 新样本特征 (n_samples, n_features)
            y_new: 新样本标签 (n_samples, n_labels)
            sample_weight: 样本权重（如CI系数大小）
        """
        n_new = len(X_new)

        for i in range(n_new):
            self.added_count += 1

            if len(self.buffer) < self.capacity:
                # 缓冲区未满，直接添加
                self.buffer.append((X_new[i], y_new[i], sample_weight))
                self.sample_weights.append(sample_weight)
            else:
                # 缓冲区已满，随机替换
                # 每个样本被保留的概率 = capacity / added_count
                replace_prob = self.capacity / self.added_count
                if self.rng.rand() < replace_prob:
                    replace_idx = self.rng.randint(0, self.capacity)
                    self.buffer[replace_idx] = (X_new[i], y_new[i], sample_weight)
                    self.sample_weights[replace_idx] = sample_weight

    def sample(self, batch_size, weighted=True):
        """
        从缓冲区采样

        Args:
            batch_size: 采样数量
            weighted: 是否使用样本权重

        Returns:
            X_sampled, y_sampled
        """
        if len(self.buffer) == 0:
            return np.array([]), np.array([])

        buffer_array = np.array(self.buffer)
        X = np.stack(buffer_array[:, 0])
        y = np.stack(buffer_array[:, 1])

        if weighted:
            weights = np.array(self.sample_weights)
            weights = weights / weights.sum()  # 归一化
            idxs = self.rng.choice(
                len(X), size=batch_size, p=weights, replace=True
            )
        else:
            idxs = self.rng.choice(len(X), size=batch_size, replace=True)

        return X[idxs], y[idxs]

    def __len__(self):
        return len(self.buffer)
```

**集成到训练循环**：

```python
# 在训练时混合新数据和replay buffer
replay_buffer = ReplayBuffer(capacity=10000)

def train_with_replay(self, X_new, y_new, batch_size):
    # 添加新数据到replay buffer
    replay_buffer.add(X_new, y_new, sample_weight=ci_coefficient)

    # 从buffer采样历史数据
    X_replay, y_replay = replay_buffer.sample(
        batch_size // 2, weighted=True
    )

    # 混合新数据和历史数据
    X_mixed = np.vstack([X_new[:batch_size//2], X_replay])
    y_mixed = np.vstack([y_new[:batch_size//2], y_replay])

    # 训练
    self.fit(X_mixed, y_mixed, batch_size=batch_size)
```

---

## 四、可信度量化评估

### 4.1 可信度评分标准

| 维度 | 权重 | 评分依据 |
|------|------|---------|
| 数据质量 | 30% | 累积数据覆盖率、正样本增长趋势 |
| 模型稳定性 | 30% | 过拟合/欠拟合风险、轮次间波动 |
| 预测可靠性 | 25% | 历史表现、测试集指标 |
| 架构合理性 | 15% | 与工业界最佳实践对比 |

### 4.2 各轮次可信度评估

| 轮次 | 训练数据量 | 正样本数 | 正样本比例 | 模型可信度 | 主要风险 | 可靠性描述 |
|------|------------|----------|------------|------------|---------|-----------|
| **Loop 1** | ~8,500 | 50-200 | 0.5%-2.0% | **低 (0.2-0.4)** | 严重过拟合/欠拟合 | 不可信：仅用于初始化 |
| **Loop 2** | ~17,000 | 100-400 | 0.6%-2.5% | **中低 (0.4-0.5)** | 遗忘历史知识 | 谨慎使用：参考为主 |
| **Loop 3** | ~34,000 | 200-800 | 0.7%-3.0% | **中 (0.5-0.6)** | 不平衡加剧 | 可部分信任：需验证 |
| **Loop 4+** | >50,000 | >500 | 1.0%-4.0% | **中高 (0.6-0.7)** | 收敛速度慢 | 相对可信：稳定后使用 |

### 4.3 当前实现整体评分

| 方面 | 评分 | 说明 |
|------|------|------|
| **数据累积策略** | ✅ 8/10 | 有效的累积机制，但缺乏重要性加权 |
| **Warm-start实现** | ⚠️ 4/10 | 仅加载权重，未真正优化 |
| **不平衡处理** | ✅ 7/10 | 动态权重计算合理，但早期仍极端 |
| **采样策略** | ✅ 6/10 | 概率排序合理，但缺乏不确定性采样 |
| **整体可信度** | ⚠️ **5.5/10** | 中等偏低，需要改进 |

---

## 五、快速验证建议

### 5.1 实验对比设计

**并行运行3个版本**：

```bash
# 版本A：当前实现（基线）
# - 每轮150 epochs
# - 固定学习率0.001
# - 标准概率排序采样

# 版本B：简单warm-start
# - Loop 2+：50-80 epochs（根据轮次衰减）
# - 学习率衰减至0.0005
# - 添加模型质量评估

# 版本C：完整改进
# - Loop 2+：知识蒸馏 + 80 epochs + 动态lr
# - 不确定性采样策略
# - 样本权重（当前=1.0, 历史=0.7）
```

### 5.2 评估指标

| 指标类别 | 具体指标 | 说明 |
|----------|----------|------|
| **收敛速度** | 达到目标F1(0.7)的轮次 | 越少越好 |
| **最终质量** | 测试集F1, AUC, Precision, Recall | 越高越好 |
| **预测稳定性** | 轮次间F1波动标准差 | 越小越好 |
| **遗忘指标** | 历史数据性能下降幅度 | <10%为优秀 |
| **采样效率** | 每轮新增重要CSF数量/采样数 | 越高越好 |

### 5.3 验证脚本框架

```python
# 文件：experiments/validate_improvements.py
import mlflow
import logging
from graspkit.ml_module import train_model, predict_model

def run_experiment(config, version_name):
    """运行单个实验版本"""
    mlflow.set_experiment("graspkit_model_credibility")
    mlflow.start_run(run_name=f"loop{config.cal_settings.cal_loop_num}_{version_name}")

    try:
        # 训练
        model, X_train, X_test, y_train, y_test = train_model(
            config, ..., logger
        )

        # 评估
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

        # 记录
        mlflow.log_metrics({
            "train_f1": metrics["train_f1"],
            "test_f1": metrics["test_f1"],
            "test_auc": metrics["test_auc"],
            "train_samples": len(X_train),
        })

        return metrics

    finally:
        mlflow.end_run()

# 对比实验
versions = ["baseline", "warm_start", "full_improvements"]
results = {}

for version in versions:
    results[version] = run_experiment(config, version)

# 生成对比报告
generate_comparison_report(results)
```

---

## 六、结论

### 6.1 核心发现

1. **第一轮模型几乎不可信** - 数据太少（<200正样本），建议谨慎使用其预测
2. **Warm-start效果有限** - 需要实现真正的增量学习策略（当前仅为权重加载）
3. **数据累积有效但不够** - 需要配合更好的训练策略才能发挥价值
4. **缺乏知识保护机制** - 没有防止catastrophic forgetting的措施

### 6.2 优先级建议

| 优先级 | 改进项 | 预期效果 | 实施难度 |
|-------|--------|----------|---------|
| **P0（立即）** | 动态调整训练参数 | ⭐⭐⭐ | ⭐ 简单 |
| **P0（立即）** | 添加模型质量评估 | ⭐⭐ | ⭐ 简单 |
| **P1（短期）** | 历史数据重要性加权 | ⭐⭐⭐ | ⭐⭐ 中等 |
| **P2（中期）** | 不确定性采样策略 | ⭐⭐ | ⭐⭐ 中等 |
| **P3（长期）** | 知识蒸馏 | ⭐⭐⭐⭐ | ⭐⭐⭐ 复杂 |
| **P4（长期）** | PEFT/LoRA适配器 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ 很难 |

### 6.3 风险提示

**高风险场景**：
- 🔴 **Loop 1预测直接用于生产** - 强烈建议至少等待Loop 2
- 🔴 **数据分布突变** - 如果物理参数大幅调整，需重新训练
- 🔴 **正样本比例<0.1%** - 极端不平衡，模型几乎无效

**监控指标**：
- 🟡 `overfitting_gap > 0.15` - 过拟合风险高
- 🟡 `test_f1 < 0.4` - 模型质量差，不应使用预测
- 🟡 `predicted_positive_ratio / positive_ratio > 5` - 模型严重偏差

### 6.4 最终建议

**短期行动（1-2周）**：
1. 实施"动态调整训练参数"（3.1.A）
2. 实施"模型质量评估"（3.1.B）
3. 运行对比实验验证效果

**中期规划（1-2月）**：
1. 实施"历史数据重要性加权"（3.1.C）
2. 引入不确定性采样（3.2.B）
3. 建立完整的实验对比框架

**长期优化（3-6月）**：
1. 研究知识蒸馏可行性（3.2.A）
2. 评估PEFT/LoRA适配（3.3.A）
3. 建立持续监控和A/B测试机制

---

## 附录

### A. 相关代码文件

| 文件 | 关键函数/行数 | 说明 |
|------|--------------|------|
| `ml_trainer.py` | 39-337: `train_model()` | 主训练逻辑 |
| `ml_trainer.py` | 449-575: `predict_model()` | CSF采样逻辑 |
| `ml_initializer.py` | 462-519: `merge_historical_ci_data()` | 数据累积 |
| `ml_initializer.py` | 582-657: `generate_train_csfs_descriptors()` | 训练数据生成 |
| `neural_network.py` | 258-371: `fit()` | 模型训练循环 |
| `neural_network.py` | 424-543: `_calculate_dynamic_weights()` | 类别权重计算 |

### B. 配置文件建议

在`config.toml`中添加：

```toml
[ml_config]
# 模型质量阈值
warm_start_quality_threshold = 0.3  # F1 < 0.3时重新训练

# 历史数据权重
historical_weight = 0.7  # Loop 2+中历史数据权重

# 训练参数
initial_learning_rate = 0.001
warm_start_learning_rate_decay = 0.5

# 采样策略
use_uncertainty_sampling = false
uncertainty_method = "entropy"  # entropy, margin, least_confident

# 实验对比
run_baseline = true
run_warm_start = true
run_full_improvements = false
```

### C. 参考文献

1. **Incremental Learning**:
   - Parisi, G. I., et al. (2019). "Continual lifelong learning with neural networks: A review." *Neural Networks*, 113, 54-71.

2. **Catastrophic Forgetting**:
   - Kemker, R., et al. (2018). "Gradient Episodic Memory for Continual Learning." *NeurIPS*.

3. **Active Learning**:
   - Settles, B. (2009). "Active Learning Literature Survey." *University of Wisconsin-Madison*.

4. **Knowledge Distillation**:
   - Hinton, G., et al. (2015). "Distilling the Knowledge in a Neural Network." *arXiv*.

---

**文档版本**: v1.0
**最后更新**: 2026-02-27
**作者**: Sisyphus AI Agent
**审核**: 待审核
