# 回归模型模块变更记录

**日期**：2026-02-27
**分支**：3.1dev1
**类型**：新增功能（不修改任何原有文件）

---

## 背景

原 ML 模块使用二分类（`BCEWithLogitsLoss`）判断 CSF 重要性，通过 `cutoff_value` 阈值将连续的 CI² 值离散化为 0/1 标签，丢失了 CI² 的数值大小信息。

新增的回归方案直接预测 `log₁₀(CI²)` 值，保留完整排序信息：

```
分类（现有）：
  accumulated_ci_squared  →  (ci² ≥ cutoff)?  →  0/1 label  →  BCELoss  →  prob [0,1]  →  top-k

回归（新增）：
  accumulated_ci_squared  →  log₁₀(max(ci², 1e-15))  →  连续 label [-15, 0]  →  HuberLoss  →  predicted_log_ci²  →  top-k
```

---

## 新增文件

### 1. `src/graspkit/ml_module/ml_regression_model.py`

新增 `ANNRegressor` 类。

#### 与 `ANNClassifier` 的对比

| 项目 | `ANNClassifier`（原有） | `ANNRegressor`（新增） |
|------|------------------------|------------------------|
| 网络结构 | Linear → LayerNorm → ReLU → Dropout（×2）→ Linear | 相同 |
| 输出激活 | 无（BCEWithLogitsLoss 内含 sigmoid） | 无（线性输出） |
| 损失函数 | `BCEWithLogitsLoss` | `HuberLoss(delta=1.0)` |
| 动态权重 | 支持（正负样本比例自适应） | 不需要 |
| 标签范围 | `{0, 1}` | `[-15, 0]`（`log₁₀ CI²`） |
| 主要指标 | F1 / ROC-AUC | MAE / RMSE / Spearman ρ |

#### 构造函数

```python
ANNRegressor(
    input_size: int,
    hidden_size: int = 150,
    output_size: int = 1,       # 等于能级数量
    learning_rate: float = 0.001,
    huber_delta: float = 1.0,   # HuberLoss delta 参数
    device: str | None = None,
    random_seed: int | None = None,
)
```

#### 主要方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `fit` | `(X_train, y_train, X_val, y_val, ...)` | 训练，支持早停和学习率调度 |
| `predict` | `(X) → ndarray (n, n_levels)` | 返回 `log₁₀(CI²)` 预测值 |
| `predict_batch` | `(X, batch_size) → ndarray` | 分批预测，适合大数据集 |
| `evaluate` | `(X, y) → (mae, rmse, spearman_rho)` | 回归评估三指标 |
| `plot_curve` | `(X, y, filename, level_titles)` | 各能级预测值 vs 真实值散点图 |
| `save_model` | `(path)` | 保存完整模型状态（含超参数） |
| `load_model` | `(path, device)` | 加载已保存模型 |

---

### 2. `src/graspkit/ml_module/ml_regression_trainer.py`

新增 4 个函数，接口设计与 `ml_trainer.py` 保持一致，可直接替换使用。

#### `generate_regression_train_descriptors`

```python
def generate_regression_train_descriptors(
    raw_csfs_descriptors: np.ndarray,   # shape: (total_csfs, n_features)
    accumulated_idxs: np.ndarray,        # shape: (n_accumulated,)
    accumulated_ci_squared: np.ndarray,  # shape: (n_levels, n_accumulated)
    min_clip_value: float = 1e-15,
) -> np.ndarray:                         # shape: (n_accumulated, n_features + n_levels)
```

将 CI² 值转换为 `log₁₀` 连续标签。`min_clip_value` 对 CI² 进行下限截断，避免 `log(0)`。
返回 `column_stack([descriptors, log_ci_labels])`，与分类版的 `generate_train_csfs_descriptors` 格式一致。

#### `train_regression_model`

```python
def train_regression_model(
    train_data: np.ndarray,
    config: MLCalConfig,
    loop_num: int,
    logger: logging.Logger | None = None,
) -> tuple[ANNRegressor, dict]:
```

训练回归模型，自动处理 CPU 线程优化、验证集早停。
返回 `(model, metrics_dict)`，`metrics_dict` 包含：

```python
{
    "mae": float,              # 测试集 MAE
    "rmse": float,             # 测试集 RMSE
    "spearman_rho": float,     # 测试集 Spearman ρ
    "overfitting_gap": float,  # 训练ρ - 测试ρ（过拟合检测）
    "train_mae": float,
    "train_rmse": float,
    "train_spearman_rho": float,
}
```

#### `evaluate_regression_model`

```python
def evaluate_regression_model(
    model: ANNRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float]:  # (mae, rmse, spearman_rho)
```

`model.evaluate()` 的薄封装，与 `ml_trainer.py` 中 `evaluate_model` 的调用风格对应。

#### `predict_regression_model`

```python
def predict_regression_model(
    model: ANNRegressor,
    unselected_csf_descriptors: np.ndarray,
    unselected_idxs: np.ndarray,
    verified_important_idxs: np.ndarray,
    config: MLCalConfig,
    train_data_counts: MLDataCounts,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MLDataCounts]:
```

接口与 `predict_model()` 完全一致，返回相同的四元组：

```python
(ml_sampled_idxs, verified_important_idxs, y_predicted_log_ci, train_data_counts)
```

**选择逻辑**（替代分类版的概率阈值）：

```python
# 取各能级预测值的最大值，代表"至少在一个能级上重要"
max_predicted_log_ci = np.max(y_predicted_log_ci, axis=1)
# 降序排列，选 top-k
ranked_idxs = np.argsort(max_predicted_log_ci)[::-1]
ml_sampled_idxs = unselected_idxs[ranked_idxs[:new_target]]
```

---

## 修改文件

### `src/graspkit/ml_module/__init__.py`

在原有导入末尾追加：

```python
from .ml_regression_model import ANNRegressor
from .ml_regression_trainer import (
    generate_regression_train_descriptors,
    train_regression_model,
    evaluate_regression_model,
    predict_regression_model,
)
```

`__all__` 中新增 5 个符号：

```python
"ANNRegressor",
"generate_regression_train_descriptors",
"train_regression_model",
"evaluate_regression_model",
"predict_regression_model",
```

### `src/graspkit/__init__.py`

在 `from .ml_module import (...)` 块末尾追加相同的 5 个符号，并同步更新 `__all__`。

---

## 文件清单

| 文件 | 操作 |
|------|------|
| `src/graspkit/ml_module/ml_regression_model.py` | **新建** |
| `src/graspkit/ml_module/ml_regression_trainer.py` | **新建** |
| `src/graspkit/ml_module/__init__.py` | **追加**（未修改原有内容） |
| `src/graspkit/__init__.py` | **追加**（未修改原有内容） |

---

## 验证

```bash
# 功能测试
python -c "
import numpy as np
from graspkit import ANNRegressor, generate_regression_train_descriptors

# 测试模型训练与评估
X = np.random.randn(200, 24).astype(np.float32)
y = np.random.uniform(-15, 0, (200, 3)).astype(np.float32)
model = ANNRegressor(input_size=24, hidden_size=96, output_size=3)
model.fit(X[:160], y[:160], X[160:], y[160:])
mae, rmse, rho = model.evaluate(X[160:], y[160:])
print(f'MAE={mae:.4f}, RMSE={rmse:.4f}, Spearman rho={rho:.4f}')

# 测试描述符生成
raw_desc = np.random.randn(500, 24)
idxs = np.arange(200)
ci_sq = np.random.uniform(0, 1, (3, 200))
train_data = generate_regression_train_descriptors(raw_desc, idxs, ci_sq)
print(f'train_data shape: {train_data.shape}')  # (200, 27)
"
```
