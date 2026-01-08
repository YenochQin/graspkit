# 多标签分类支持 - 修改日志

**日期**: 2025-01-08
**版本**: 3.0dev2
**作者**: Claude (YenochQin)

## 概述

本次修改为机器学习模块添加了**多标签分类（Multi-Label Classification）**支持，使神经网络能够同时预测每个CSF在多个能级上的重要性，而不是将多个能级的信息压缩成单一标量。

## 背景

### 原始设计（单标签分类）
- 每个CSF只有一个标签（0/1）
- 多个能级的CI系数通过取最大值压缩成一个标量
- 输出维度：`output_size = 2`（二分类）
- 损失函数：`CrossEntropyLoss` + softmax

### 新设计（多标签分类）
- 每个CSF有 `n_correct_levels` 个独立标签（每个能级一个）
- 保留所有能级的独立信息
- 输出维度：`output_size = n_correct_levels`
- 损失函数：`BCEWithLogitsLoss` + sigmoid

## 核心修改

### 1. 数据标签转置 (`ml_initializer.py:764-765`)

**问题**：`accumulated_ci_squared` 形状为 `(n_correct_levels, n_current_csfs)`，但描述符的行维度是 `n_current_csfs`

**解决**：转置标签以匹配描述符的行维度

```python
# 修改前
important_csfs_mask = accumulated_ci_squared >= cutoff_value

# 修改后
# 转置以匹配描述符的行维度: (n_current_csfs, n_correct_levels)
important_csfs_mask = (accumulated_ci_squared >= cutoff_value).T
```

**数据流变化**：
```python
# 之前
caled_csfs_descriptors: (n_current_csfs, descriptor_features + 1)
important_csfs_mask: (n_correct_levels, n_current_csfs)  # 维度不匹配！

# 现在
caled_csfs_descriptors: (n_current_csfs, descriptor_features + n_correct_levels)
important_csfs_mask: (n_current_csfs, n_correct_levels)  # 维度匹配 ✓
```

### 2. ANNClassifier 多标签支持 (`neural_network.py`)

#### 2.1 自动识别机制

**核心逻辑** (`neural_network.py:158-161`)：
```python
# output_size = 1: 单标签分类（单能级）
# output_size > 1: 多标签分类（多能级）
if multi_label is None:
    self.multi_label = output_size > 1
else:
    self.multi_label = multi_label  # 允许手动覆盖
```

#### 2.2 损失函数适配 (`neural_network.py:182-196`)

```python
if self.multi_label:
    # 多标签分类：使用 BCEWithLogitsLoss
    if self.class_weights is not None:
        pos_weight = torch.tensor([self.class_weights[1]], dtype=torch.float32).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        self.criterion = nn.BCEWithLogitsLoss()
else:
    # 单标签分类：使用 CrossEntropyLoss
    if self.class_weights is not None:
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
    else:
        self.criterion = nn.CrossEntropyLoss()
```

#### 2.3 激活函数适配 (`neural_network.py:549-555`)

```python
with torch.no_grad():
    model_outputs = self.model(X_tensor)
    if self.multi_label:
        # 多标签分类：每个标签独立的 sigmoid 概率
        outputs = torch.sigmoid(model_outputs)
    else:
        # 单标签分类：softmax 概率
        outputs = torch.softmax(model_outputs, dim=1)
```

#### 2.4 标签数据类型 (`neural_network.py:287-289`)

```python
# 多标签分类需要 float32 类型，单标签需要 long 类型
y_dtype = torch.float32 if self.multi_label else torch.long
y_train_tensor = torch.tensor(y_train, dtype=y_dtype).to(self.device)
```

#### 2.5 预测逻辑 (`neural_network.py:633-640`)

```python
if self.multi_label:
    # 多标签分类：使用 sigmoid 和阈值
    y_probability = torch.sigmoid(outputs).cpu().numpy()
    y_pred = (y_probability > 0.5).astype(int)
else:
    # 单标签分类：使用 argmax
    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
    y_probability = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
```

#### 2.6 评估指标 (`neural_network.py:642-659`)

```python
if self.multi_label:
    # 多标签评估指标（使用 average='samples'）
    metrics = {
        "accuracy": accuracy_score(y.flatten(), y_pred.flatten()),
        "f1_score": f1_score(y, y_pred, average='samples'),
        "precision": precision_score(y, y_pred, average='samples'),
        "recall": recall_score(y, y_pred, average='samples'),
    }
else:
    # 单标签评估指标
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_probability)
    }
```

### 3. train_model 函数更新 (`ml_trainer.py`)

#### 3.1 数据提取逻辑 (`ml_trainer.py:45-54`)

```python
# 从 correct_levels_ci 推断能级数量
n_correct_levels = correct_levels_ci.shape[0] if correct_levels_ci.ndim == 2 else 1
descriptor_features = caled_csfs_descriptors.shape[1] - n_correct_levels

X = caled_csfs_descriptors[:, :descriptor_features]
y = caled_csfs_descriptors[:, descriptor_features:]  # 多标签：所有能级的标签
```

#### 3.2 数据平衡性检查 (`ml_trainer.py:59-75`)

```python
# 统计所有能级的正负样本
positive_count = np.sum(y_train == 1)
negative_count = np.sum(y_train == 0)

# 计算每个能级的正样本比例
per_level_positive_ratio = np.mean(y_train, axis=0)
avg_positive_ratio = np.mean(per_level_positive_ratio)

logger.info(f"多标签分类 - {n_correct_levels} 个能级")
logger.info(f"各能级正样本比例: {np.array2string(per_level_positive_ratio, precision=4)}")
```

#### 3.3 模型初始化 (`ml_trainer.py:87-94`)

```python
model = ANNClassifier(
    input_size=X_train.shape[1],
    output_size=n_correct_levels,  # 自动根据 output_size>1 启用多标签分类
    hidden_size=hidden_size,
    learning_rate=0.001,
    class_weights=class_weights,
    model_architecture="tensornet",
    # 不再需要显式传入 multi_label=True
)
```

#### 3.4 预测和评估 (`ml_trainer.py:215-322`)

```python
# 多标签分类：predict_proba 返回 (n_samples, n_labels)，取平均概率用于分析
y_probability = model.predict_proba(X_test).mean(axis=1)

# 多标签：统计至少在一个能级上被预测为重要的样本数
positive_samples_test = np.any(y_prediction == 1, axis=1)

# 模型评估：传入完整的概率矩阵而不是平均概率
y_probability_matrix_test = model.predict_proba(X_test)
f1, roc_auc, accuracy, precision, recall = ANNClassifier.model_evaluation(
    y_test, y_prediction, y_probability_matrix_test
)
```

## 技术细节

### 维度对照表

| 组件 | 单标签分类 | 多标签分类 |
|------|----------|----------|
| `output_size` | 2 | n_correct_levels |
| `y` 形状 | (n_samples,) | (n_samples, n_correct_levels) |
| 模型输出 | (n_samples, 2) | (n_samples, n_correct_levels) |
| 激活函数 | softmax | sigmoid |
| 损失函数 | CrossEntropyLoss | BCEWithLogitsLoss |
| 标签类型 | torch.long | torch.float32 |
| 预测方法 | argmax | 阈值判断 |

### 自动识别逻辑

```python
n_correct_levels = correct_levels_ci.shape[0] if correct_levels_ci.ndim == 2 else 1

# n_correct_levels = 1: 单能级 → 单标签分类 (multi_label = False)
# n_correct_levels > 1: 多能级 → 多标签分类 (multi_label = True)
```

### 兼容性保证

1. **向后兼容**：旧代码中显式传入 `multi_label=True` 仍然有效
2. **自动迁移**：加载旧模型时会验证 `output_size`，不匹配时自动创建新模型
3. **手动覆盖**：可以通过 `multi_label` 参数手动覆盖自动判断

## 推理阶段改进

### 原始流程
```
多个能级 → 取最大值压缩 → 单个预测 → 选择CSF
```

### 新流程
```
多个能级 → 独立预测 → 综合决策 → 选择CSF
```

在推理（Inference）阶段，模型会输出每个CSF在各个能级上的重要性概率，然后可以基于以下策略做最终决策：
- 任意能级重要 → 选择该CSF（`any()`）
- 多数能级重要 → 选择该CSF（投票）
- 加权组合 → 选择该CSF（自定义权重）

## 文件修改清单

1. **src/graspkit/ml_module/ml_initializer.py**
   - `generate_train_csfs_descriptors()`: 添加 `.T` 转置

2. **src/graspkit/ml_module/neural_network.py**
   - `__init__()`: 添加自动识别逻辑、参数类型改为 `bool | None`
   - 损失函数初始化: 添加 `multi_label` 分支
   - `fit()`: 标签数据类型适配
   - `predict_proba()`: 激活函数适配
   - `predict_proba_batch()`: 激活函数适配
   - `evaluate()`: 预测和评估指标适配
   - `plot_roc_curve()`: 激活函数适配

3. **src/graspkit/ml_module/ml_trainer.py**
   - `train_model()`: 数据提取、平衡性检查、模型初始化、预测评估全面适配

## 测试建议

1. **单能级场景**：
   ```python
   n_correct_levels = 1
   → multi_label = False
   → 验证单标签分类正常工作
   ```

2. **多能级场景**：
   ```python
   n_correct_levels = 3
   → multi_label = True
   → 验证多标签分类正常工作
   ```

3. **模型加载**：
   ```python
   # 验证跨轮次的模型加载和继续训练
   # 验证 output_size 不匹配时的自动重建
   ```

## 性能影响

- **内存**：多标签分类需要存储更多标签，内存占用略微增加
- **计算**：`BCEWithLogitsLoss` 比 `CrossEntropyLoss` 略快（无需 softmax）
- **准确率**：保留更多能级信息，预期准确率提升

## 未来改进方向

1. **自适应阈值**：为每个能级学习独立的最优阈值
2. **能级权重**：为不同能级设置不同的重要性权重
3. **注意力机制**：让模型学习关注更重要的能级
4. **多任务学习**：同时优化能级分类和其他辅助任务

## 相关文档

- [PyTorch BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [Scikit-learn Multi-label Classification](https://scikit-learn.org/stable/modules/multiclass.html#multi-label-classification)
- [多标签分类评估指标](https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics)

## Changelog

- **2025-01-08**: 初始版本，添加多标签分类支持
- **待定**: 性能优化和测试完善
