# 多能级预测概率和绘图修复 - 修改日志

**日期**: 2025-01-15
**版本**: 2.0dev1
**作者**: Claude (YenochQin)

## 概述

本次修改修复了多能级情况下机器学习预测概率处理和绘图的两个关键问题：

1. **预测概率维度错误**：`predict_model` 函数输出的 `y_current_cal_probability` 只保留第1列，应为二维数组 `(n_csfs, n_levels)`
2. **绘图逻辑错误**：`save_and_plot_results` 函数中的索引切片 `correct_levels_ci[:, caled_csfs_idxs_array]` 导致越界

## 背景

### 问题1: 预测概率维度错误

**原始设计**：
- 模型是多标签分类，输出 `(n_samples, n_levels)` 的概率矩阵
- `predict_proba` 返回 `(n_samples, n_levels)` 二维数组
- 但代码中错误地只取第1列：`y_current_cal_probability = model.predict_proba(X_current_calc)[:, 1]`

**问题影响**：
- 丢失了其他能级的预测概率信息
- 绘图时无法为每个能级生成对应的图表

### 问题2: 绘图索引越界

**错误代码**：
```python
current_cal_ci = correct_levels_ci[:, caled_csfs_idxs_array]
```

**错误原因**：
- `correct_levels_ci` 是 `(n_levels, n_current_csfs)`，只包含当前计算的CSFs
- `caled_csfs_idxs_array` 是当前计算的CSF在**总池**中的索引
- 使用总池索引去索引当前CSF数组导致越界

**错误示例**：
- `correct_levels_ci` shape: `(3, 256896)` - axis 1 大小 256896
- `caled_csfs_idxs_array` 最大值: 257579
- 结果：`index 257579 is out of bounds for axis 1 with size 256896`

## 核心修改

### 1. predict_model 函数修复 (`ml_trainer.py`)

#### 1.1 未选择CSF预测修复 (第468行)

**修改前**：
```python
# 只取第1列（假设是二分类的正类概率）
y_unselected_prediction = (y_unselected_probability[:, 1] > 0.5).astype(int)
```

**修改后**：
```python
# 多能级情况：在所有能级中取最大概率值，然后与阈值比较
y_unselected_prediction = (np.max(y_unselected_probability, axis=1) > 0.5).astype(int)
```

**逻辑说明**：
- `y_unselected_probability` shape: `(n_unselected_csfs, n_levels)`
- `np.max(..., axis=1)` 对每个CSF取所有能级中的最大概率
- 如果最大概率 > 0.5，则认为该CSF重要

#### 1.2 当前计算CSF预测概率修复 (第476行)

**修改前**：
```python
# 只取第1列
y_current_cal_probability = model.predict_proba(X_current_calc)[:, 1]
# shape: (n_current_csfs,)
```

**修改后**：
```python
# 多能级情况：保留所有能级的预测概率
y_current_cal_probability = model.predict_proba(X_current_calc)
# shape: (n_current_csfs, n_levels)
```

**数据流变化**：
```python
# 之前
y_current_cal_probability: (n_current_csfs,)
# 只包含第2个能级的概率（或二分类的正类概率）

# 现在
y_current_cal_probability: (n_current_csfs, n_levels)
# 包含所有能级的预测概率
# level_probability = y_current_cal_probability[:, level_idx]
```

#### 1.3 ML选择概率排序修复 (第537-541行)

**修改前**：
```python
# 获取ML预测重要组态的正类概率（只取第1列）
ml_predicted_important_probabilities = y_unselected_probability[
    ml_predicted_important_local_idxs, 1
]
```

**修改后**：
```python
# 获取ML预测重要组态的最大概率（在所有能级中取最大值）
ml_predicted_important_probabilities = np.max(
    y_unselected_probability[ml_predicted_important_local_idxs],
    axis=1
)
```

### 2. save_and_plot_results 函数重构 (`ml_results_analyzer.py`)

#### 2.1 函数签名修改

**修改前**：
```python
def save_and_plot_results(
    logger: logging.Logger,
    evaluation_results,
    model,
    path_cfg,
    correct_levels_ci: np.ndarray,
    caled_csfs_idxs_array: np.ndarray = np.array([], dtype=int),
    y_current_cal_probability=None,
    save_model: bool = True,
    save_data: bool = True,
    plot_curves: bool = True,
    spectral_term: list | None = None,
):
```

**修改后**：
```python
def save_and_plot_results(
    config,
    logger: logging.Logger,
    evaluation_results,
    model,
    correct_levels_ci: np.ndarray,
    caled_csfs_idxs_array: np.ndarray = np.array([], dtype=int),
    y_current_cal_probability=None,
    save_model: bool = True,
    save_data: bool = True,
    plot_curves: bool = True,
):
    # 内部定义path_cfg
    path_cfg = config.cal_path
```

**改动说明**：
- `config` 参数放在最开头，保持文件中函数定义的一致性
- 移除 `path_cfg` 参数，改为内部定义 `path_cfg = config.cal_path`
- 移除 `spectral_term` 参数，改为内部获取 `config.cal_settings.spectral_term`

#### 2.2 绘图逻辑重构

**修改前**（错误逻辑）：
```python
# 错误的索引切片导致越界
if caled_csfs_idxs_array.size > 0:
    if len(correct_levels_ci.shape) > 1:
        current_cal_ci = correct_levels_ci[:, caled_csfs_idxs_array]  # 越界！
        cal_mix_coeff_list = np.sqrt(np.sum(current_cal_ci**2, axis=0))

# 只绘制一幅图
ANNClassifier.plot_curve(
    cal_mix_coeff_list,  # 压缩后的CI系数
    y_prob_current_cal,
    ...
)
```

**修改后**（多能级支持）：
```python
# 生成latex格式的谱项符号列表
latex_form_spectral_term: list[str] = []
spectral_term = config.cal_settings.spectral_term
if spectral_term is not None:
    for term in spectral_term:
        _formatted_conf, format_LS_coupling = ConfigurationFormatter(term).conf_format()
        latex_form_spectral_term.append(format_LS_coupling)

# 确定能级数量
if len(correct_levels_ci.shape) > 1:
    n_levels = correct_levels_ci.shape[0]  # 多能级情况
else:
    n_levels = 1  # 单能级情况

# 为每个能级绘制图表
for level_idx in range(n_levels):
    # 提取当前能级的CI系数（无需索引切片，直接取对应行）
    if len(correct_levels_ci.shape) > 1:
        level_ci = np.abs(correct_levels_ci[level_idx, :])
    else:
        level_ci = np.abs(correct_levels_ci)

    # 提取当前能级的预测概率
    if y_current_cal_probability is not None:
        if len(y_current_cal_probability.shape) > 1:
            level_probability = y_current_cal_probability[:, level_idx]
        else:
            level_probability = y_current_cal_probability
    else:
        # 回退逻辑
        ...

    # 生成图表标题
    if level_idx < len(latex_form_spectral_term):
        level_title = latex_form_spectral_term[level_idx]
    else:
        level_title = f"Level {level_idx}"

    # 绘制当前能级的ROC和PR曲线
    plot_file = path_cfg.roc_curves_path / f"{path_cfg.loop_file_name}_level{level_idx}_roc_pr_curves.png"
    ANNClassifier.plot_curve(
        level_ci,
        level_probability,
        evaluation_results["true_labels"]["y_test"],
        evaluation_results["probabilities"]["y_probability_test"],
        str(plot_file),
        level_title=level_title,  # 使用能级对应的谱项符号
    )
```

**关键改进**：
1. **移除错误的索引切片**：直接使用 `correct_levels_ci[level_idx, :]` 获取对应能级的CI系数
2. **支持多能级绘图**：为每个能级单独绘制一幅图
3. **使用LaTeX谱项符号**：图表标题使用格式化的谱项符号（如 `3d^8 4s^2 ^1S_0`）

### 3. plot_curve 函数增强 (`neural_network.py`)

#### 3.1 添加 level_title 参数

**修改前**：
```python
@staticmethod
def plot_curve(
    cal_mix_coeff_List: np.ndarray,
    y_probability_all: np.ndarray,
    y_test: np.ndarray,
    y_probability: np.ndarray,
    filename: str
):
```

**修改后**：
```python
@staticmethod
def plot_curve(
    cal_mix_coeff_List: np.ndarray,
    y_probability_all: np.ndarray,
    y_test: np.ndarray,
    y_probability: np.ndarray,
    filename: str,
    level_title: str = "Ci Values vs Predicted Probability"
):
```

#### 3.2 使用 level_title 设置图表标题

**修改前**：
```python
plt.title('Ci Values vs Predicted Probability')
plt.title('Ci Values vs Predicted Probability (Data Mismatch)')
plt.title(f'Ci Values vs Predicted Probability\n(n={total_count}, zeros={zero_count})')
```

**修改后**：
```python
plt.title(level_title)
plt.title(f'{level_title} (Data Mismatch)')
plt.title(f'{level_title}\n(n={total_count}, zeros={zero_count})')
```

### 4. train.py 调用修改

**修改前**：
```python
gk.save_and_plot_results(
    logger=logger,
    evaluation_results=evaluation_results,
    model=model,
    path_cfg=config.cal_path,
    correct_levels_ci=correct_levels_ci,
    caled_csfs_idxs_array=caled_csfs_idxs_array,
    y_current_cal_probability=y_current_cal_probability,
    save_model=True,
    save_data=True,
    plot_curves=True,
)
```

**修改后**：
```python
gk.save_and_plot_results(
    config=config,  # config 放在最开头
    logger=logger,
    evaluation_results=evaluation_results,
    model=model,
    correct_levels_ci=correct_levels_ci,
    caled_csfs_idxs_array=caled_csfs_idxs_array,
    y_current_cal_probability=y_current_cal_probability,
    save_model=True,
    save_data=True,
    plot_curves=True,
)
```

### 5. 导入依赖添加 (`ml_results_analyzer.py`)

**添加导入**：
```python
from ..grasp_data_extractor.asfs_data_processor import ConfigurationFormatter
```

## 修改文件清单

| 文件路径 | 修改内容 |
|---------|---------|
| `src/graspkit/ml_module/ml_trainer.py` | predict_model 函数：修复预测概率维度和处理逻辑 |
| `src/graspkit/ml_module/ml_results_analyzer.py` | save_and_plot_results 函数：重构为支持多能级绘图 |
| `src/graspkit/ml_module/neural_network.py` | plot_curve 函数：添加 level_title 参数 |
| `ml_CSFs_selection_scripts/ml_csf_choosing/train.py` | 更新 save_and_plot_results 调用 |

## 数据流变化

### 之前（单能级/压缩）
```
correct_levels_ci: (n_levels, n_current_csfs)
       ↓ 取最大值压缩
cal_mix_coeff_list: (n_current_csfs,)
       ↓
y_current_cal_probability: (n_current_csfs,)  # 只取第1列
       ↓
1幅图表: {loop}_roc_pr_curves.png
```

### 现在（多能级/保留）
```
correct_levels_ci: (n_levels, n_current_csfs)
       ↓ 按能级分离
level_ci[0]: (n_current_csfs,)  ← level 0
level_ci[1]: (n_current_csfs,)  ← level 1
...
       ↓
y_current_cal_probability: (n_current_csfs, n_levels)
       ↓ 按能级分离
level_probability[:, 0]: (n_current_csfs,)  ← level 0
level_probability[:, 1]: (n_current_csfs,)  ← level 1
...
       ↓
n幅图表:
  - {loop}_level0_roc_pr_curves.png (标题: latex_form_spectral_term[0])
  - {loop}_level1_roc_pr_curves.png (标题: latex_form_spectral_term[1])
  - ...
```

## 兼容性

- **向后兼容**：单能级情况（`n_levels = 1`）自动适配，绘制1幅图
- **向前兼容**：支持任意数量的能级，自动生成对应数量的图表

## 测试建议

1. **单能级测试**：`spectral_term = ["3d^8 4s^2"]` → 生成1幅图
2. **多能级测试**：`spectral_term = ["3d^8 4s^2", "3d^9 4s^1", "3d^10"]` → 生成3幅图
3. **图表标题验证**：检查每幅图的第四个子图标题是否为正确的LaTeX谱项符号

## 相关问题修复

| 问题 | 描述 | 状态 |
|------|------|------|
| 问题1 | Target is multilabel-indicator but average='binary' | 已修复（之前解决） |
| 问题2 | index 257579 is out of bounds for axis 1 with size 256896 | 本次修复 |
