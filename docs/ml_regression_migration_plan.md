# ML 模型坍缩修复：回归路径迁移实施计划

> 文档日期：2026-03-18
> 基于诊断文档：`docs/ml_model_collapse_diagnosis.md`
> 代码库版本：2.9dev2（分支 3.2dev1）

---

## 一、背景与现状

### 问题现象

第 4 轮迭代后模型退化为全正预测，无法选出新组态：

```
正样本数量: 226315 (占比: 0.4629)
Epoch [150/150], Loss: 0.864200   ← Loss 卡死
AUC: 0.5114, Precision: 0.4629, Recall: 1.0000  ← 坍缩特征
```

### 根因链

```
cutoff_value = 1e-9（绝对阈值，作用于 ci²）
    ↓
所有参与 RCI 展开的 CSF 均满足 ci² ≥ 1e-9
    ↓
正样本比例 46%~95%（随迭代累积单调递增）
    ↓
pos_weight = 1.58（不足以阻止坍缩）
    ↓
BCEWithLogitsLoss 均衡点：z ≈ 0.311 → σ(z) > 0.5 → 全正预测
    ↓
梯度归零 → Loss 卡死 → 无法选出新组态
```

### 关键发现：回归基础设施已存在

代码库中已有完整的回归路径实现，但**尚未接入主训练流程**：

| 文件 | 状态 |
|------|------|
| `ml_module/ml_regression_model.py` | ✅ 已完备（`ANNRegressor`） |
| `ml_module/ml_regression_trainer.py` | ✅ 已完备（训练/预测函数） |
| `ml_module/__init__.py` | ✅ 已导出，但主流程未调用 |
| 主训练流程 | ❌ 仍走分类路径（`ANNClassifier`） |

**核心任务**：将已有的回归路径接入主训练流程，并加入配置开关。

---

## 二、改动范围与优先级

| 阶段 | 方案 | 文件 | 优先级 |
|------|------|------|--------|
| Phase 1 | 主路径切换为多输出回归 | `ml_initializer.py`、`__init__.py`、调用方 | P0（根本修复）|
| Phase 1 | 坍缩检测与 CRITICAL 日志 | `ml_trainer.py` | P1（0 副作用）|
| Phase 2 | config 开关 | `graspkit_config/ml_config_models.py` | P1 |
| Phase 2 | 诊断日志与评估对接 | `ml_initializer.py`、`ml_results_analyzer.py` | P2 |
| Phase 3 | 自适应截断兜底（分类路径） | `ml_initializer.py` | P2 |
| Phase 3 | 相对排名标签（分类路径） | `ml_initializer.py` | P2 |
| Phase 3 | 动态权重修复 | `neural_network.py` | P3 |

**最小改动路径（推荐先执行）**：Phase 1 + Phase 1.5（坍缩检测），验证回归路径稳定后再推进 Phase 2/3。

---

## 三、改动文件汇总

```
src/graspkit/ml_module/
├── ml_initializer.py         ← 新增函数；Phase3 修改标签生成逻辑
├── ml_trainer.py             ← 追加坍缩检测（Phase 1.5）
├── ml_regression_trainer.py  ← 已完备，无需改动
├── ml_regression_model.py    ← 已完备，无需改动
├── __init__.py               ← 暴露新函数
└── ml_types.py               ← 可选：新增 RegressionMetrics TypedDict

src/graspkit_config/
└── ml_config_models.py       ← 新增配置字段

外部调用方（ml_CSFs_selection_scripts/）
└── 主训练入口                ← 添加路径分支逻辑
```

---

## 四、详细实施步骤

### Step 1  添加配置字段

**文件**：`src/graspkit_config/ml_config_models.py`

在 `MLConfig`（或等效配置类）中追加字段：

```python
use_regression_model: bool = True    # True=多输出回归，False=保留分类路径
regression_min_clip: float = 1e-15  # log₁₀ 截断下限，避免 log(0)
target_positive_ratio: float = 0.10 # 分类路径相对排名目标正样本比例（Phase3 备用）
```

对应 `config.toml` 新增节：

```toml
[ml_config]
use_regression_model = true
regression_min_clip = 1e-15
# target_positive_ratio = 0.10  # Phase3 启用时取消注释
```

---

### Step 2  在 `ml_initializer.py` 新增回归描述符生成函数

**文件**：`src/graspkit/ml_module/ml_initializer.py`

在现有 `generate_train_csfs_descriptors()`（约 756 行）之后新增同级函数。
该函数复用 `ci_idx_data_processor` 已积累的数据，调用已有的
`generate_regression_train_descriptors()`（`ml_regression_trainer.py:27`）。

```python
def generate_regression_descriptors_from_config(
    config: MLCalConfig,
    raw_csfs_descriptors: NDArray[np.float64],
    logger: logging.Logger,
) -> np.ndarray:
    """
    从配置读取累积 CI 数据，生成回归用训练描述符（log₁₀(CI²) 标签）。
    接口与 generate_train_csfs_descriptors 保持一致，可直接替换。

    Returns:
        shape: (n_accumulated, n_features + n_levels)
        后 n_levels 列为 log₁₀(CI²) 连续标签
    """
    from .ml_regression_trainer import generate_regression_train_descriptors

    accumulated_idxs_ci_path = config.cal_path.accumulated_idxs_ci_path

    if config.cal_settings.cal_loop_num > 1 and not accumulated_idxs_ci_path.exists():
        raise FileNotFoundError(f"累积CI系数文件不存在: {accumulated_idxs_ci_path}")

    accumulated_idxs, accumulated_ci_squared = csfs_idxs_ci_loader(accumulated_idxs_ci_path)

    logger.info("[回归路径] 加载累积CI系数数据")
    logger.info(f"[回归路径] CSF总数（累积）: {len(accumulated_idxs)}")
    logger.info(f"[回归路径] 当前轮次: {config.cal_settings.cal_loop_num}")

    # ── 诊断日志：各能级 CI² 分布 ──
    for level_idx in range(accumulated_ci_squared.shape[0]):
        ci2_level = accumulated_ci_squared[level_idx]
        nonzero = ci2_level[ci2_level > 0]
        if len(nonzero) > 0:
            log_ci2 = np.log10(nonzero)
            top5_sum = np.sort(nonzero)[::-1][:5].sum()
            logger.info(
                "[诊断] Level %d: 非零CSF数=%d, "
                "log₁₀(CI²) 范围=[%.1f, %.1f], "
                "top5覆盖=%.1f%%",
                level_idx, len(nonzero),
                log_ci2.min(), log_ci2.max(),
                top5_sum / nonzero.sum() * 100,
            )

    min_clip_value = config.ml_config.regression_min_clip
    caled_csfs_descriptors = generate_regression_train_descriptors(
        raw_csfs_descriptors=raw_csfs_descriptors,
        accumulated_idxs=accumulated_idxs,
        accumulated_ci_squared=accumulated_ci_squared,
        min_clip_value=min_clip_value,
    )

    descriptor_path = (
        config.cal_path.cal_loop_path
        / f"{config.cal_path.loop_file_name}_regression_full"
    )
    save_descriptors(caled_csfs_descriptors, descriptor_path, "npy")
    logger.info(
        "[回归路径] 保存训练描述符（log₁₀(CI²) 标签）: %s.npy",
        descriptor_path,
    )

    return caled_csfs_descriptors
```

---

### Step 3  暴露新函数到 `__init__.py`

**文件**：`src/graspkit/ml_module/__init__.py`

```python
from .ml_initializer import (
    setup_logging,
    setup_directories,
    training_data_loader,
    ci_idx_data_processor,
    check_configuration_coupling,
    check_reference_energy_agreement,
    check_energy_convergence,
    evaluate_calculation_convergence,
    generate_train_csfs_descriptors,
    generate_regression_descriptors_from_config,   # ← 新增
    get_stay_descriptors,
)

__all__ = [
    ...
    "generate_regression_descriptors_from_config",  # ← 新增
]
```

---

### Step 4  修改主训练入口，添加路径分支

**文件**：外部调用方（`ml_CSFs_selection_scripts/ml_csf_choosing/train.py` 或等效入口）

将现有线性调用改为条件分支：

```python
if config.ml_config.use_regression_model:
    # ── 回归路径（推荐）──
    caled_descriptors = generate_regression_descriptors_from_config(
        config, raw_csfs_descriptors, logger
    )
    reg_model, reg_metrics = train_regression_model(caled_descriptors, config, logger)
    logger.info(
        "[回归路径] 训练完成 - MAE=%.4f, RMSE=%.4f, Spearman ρ=%.4f",
        reg_metrics["mae"], reg_metrics["rmse"], reg_metrics["spearman_rho"],
    )
    ml_sampled_idxs, verified_idxs, y_pred_log_ci, counts = predict_regression_model(
        reg_model,
        unselected_csf_descriptors,
        unselected_idxs,
        verified_important_idxs,
        config,
        train_data_counts,
        logger,
    )
else:
    # ── 分类路径（保留，过渡期兜底）──
    caled_descriptors = generate_train_csfs_descriptors(
        config, raw_csfs_descriptors, logger, selected_energy_data
    )
    model, X_train, y_train = train_model(
        config, caled_descriptors, correct_levels_ci, logger
    )
    eval_results, pred_outputs = evaluate_model(model, X_train, y_train, config, logger)
    ml_sampled_idxs, verified_idxs, y_pred_proba, counts = predict_model(
        model, unselected_csf_descriptors, unselected_idxs,
        verified_important_idxs, config, train_data_counts, logger,
    )
```

---

### Step 4.5  坍缩检测（Phase 1.5，0 副作用）

**文件**：`src/graspkit/ml_module/ml_trainer.py`
**位置**：`evaluate_model()` 函数内，约第 365 行（`logger.info("注意：以上指标...")` 之后）

```python
# ── 坍缩检测：Precision ≈ 正样本比例 且 Recall ≈ 1.0 ──
avg_positive_ratio = float(np.mean(y_labeled))
is_collapsed = (
    abs(labeled_precision - avg_positive_ratio) < 0.02
    and labeled_recall > 0.95
)
if is_collapsed:
    logger.critical(
        "检测到模型坍缩（全正预测）: "
        "Precision=%.4f ≈ 正样本比例=%.4f, Recall=%.4f。"
        "建议在 config.toml 中设置 use_regression_model=true。",
        labeled_precision, avg_positive_ratio, labeled_recall,
    )
```

注：`labeled_precision`、`labeled_recall` 在函数内约 352 行已计算完毕，可直接引用。

---

### Step 5（Phase 3）  分类路径兜底修复

> 仅在 `use_regression_model = false` 时生效，或作为长期维护备选方案。

#### 5.1  方案 A：相对排名标签

**文件**：`src/graspkit/ml_module/ml_initializer.py:799`

```python
# 替换当前绝对阈值标签生成
# 旧代码：
# important_csfs_mask = (accumulated_ci_squared >= cutoff_value).T

# 新代码：按百分位排名，正样本比例始终稳定
target_positive_ratio = config.ml_config.target_positive_ratio  # 默认 0.10
per_level_threshold = np.percentile(
    accumulated_ci_squared,
    100.0 * (1.0 - target_positive_ratio),
    axis=1,
)  # shape: (n_levels,)
important_csfs_mask = (accumulated_ci_squared >= per_level_threshold[:, None]).T
```

#### 5.2  方案 B：自适应截断兜底

**文件**：`src/graspkit/ml_module/ml_initializer.py:873`，在 `positive_count / total_elements` 计算之后插入：

```python
MAX_POSITIVE_RATIO = 0.30
positive_ratio = positive_count / total_elements
if positive_ratio > MAX_POSITIVE_RATIO:
    target_ratio = 0.15
    max_ci_per_csf = accumulated_ci_squared.max(axis=0)
    nonzero_ci = max_ci_per_csf[max_ci_per_csf > 0]
    adaptive_cutoff = max(
        float(np.percentile(nonzero_ci, 100.0 * (1.0 - target_ratio))),
        cutoff_value,
    )
    if adaptive_cutoff > cutoff_value:
        logger.warning(
            "正样本比例过高(%.4f)，cutoff 自动从 %.2e 提升至 %.2e",
            positive_ratio, cutoff_value, adaptive_cutoff,
        )
        important_csfs_mask = (accumulated_ci_squared >= adaptive_cutoff).T
        positive_count = int(np.sum(important_csfs_mask))
        positive_ratio = positive_count / total_elements
```

#### 5.3  方案 D：动态权重修复

**文件**：`src/graspkit/ml_module/neural_network.py`，约 519 行 `_calculate_dynamic_weights()`

在 Case A（pos_ratios ≤ 0.5）分支末尾追加：

```python
# 当正样本比例超过 0.35 时，强制压低正类权重，防止坍缩
high_ratio_mask = mask_early & (pos_ratios > 0.35)
if high_ratio_mask.any():
    calculated_weights[high_ratio_mask] = torch.clamp(
        calculated_weights[high_ratio_mask], max=0.5
    )
```

---

## 五、实施顺序

```
Step 1   graspkit_config/ml_config_models.py   添加 use_regression_model 等字段
Step 2   ml_initializer.py                新增 generate_regression_descriptors_from_config()
Step 3   ml_module/__init__.py            暴露新函数
Step 4   主训练入口                        添加 if use_regression_model 分支
Step 4.5 ml_trainer.py                   添加坍缩检测日志（无破坏性）

─── 验证节点：跑一轮回归路径，检查以下指标 ───

Step 5   （按需）Phase 3：分类路径兜底修复（方案 A/B/D）
```

---

## 六、验证指标

实施 Phase 1 后，预期在第 4 轮迭代观察到：

| 指标 | 当前（坍缩） | 目标（回归路径） |
|------|-------------|-----------------|
| Loss 曲线 | 卡死在 0.864 | 单调下降，早停正常触发 |
| Spearman ρ | ~0.51（随机水平） | > 0.70 |
| MAE（log₁₀ 空间） | N/A | < 1.5 |
| 能选出新 CSF | 否 | 是 |
| 正样本比例膨胀 | 随轮次单调增大 | 无此概念（回归无阈值）|
| 多能级支持 | 多标签 0/1（信息损失大） | 每能级独立 log₁₀(CI²) 输出 |

---

## 七、设计决策说明

### 为什么是多输出回归而非相对排名分类

| | 相对排名分类（方案 A） | 多输出回归（主方案）|
|---|---|---|
| 多能级支持 | 需要合并标签，丢失差异信息 | 每能级独立输出，天然支持 |
| 头部 CSF 区分度 | top-few 与中等贡献者同为正类 | 完整保留 15 个数量级的排序信息 |
| 推理方式 | 概率阈值 + 选样 | max(log-CI²) 直接 top-k，无需阈值 |
| 差动贡献支持 | 不直接支持 | 可在选样分数中叠加差动项 |

CI² 数据实测（来自诊断文档中对 `rmix_data.npy` 的分析）：
- 单能级 CI² 从 ~1e-20 跨至 ~0.16，共 **15 个数量级**
- top-5 CSF 吃掉 **~50% 总 CI²**，二值化损失极大
- 两能级 corr(CI²) ≈ 0.789，有共性但差异显著，必须保留各自输出

### 为什么保留分类路径开关

- 回归路径在某些极小样本场景（第 1 轮，CSF 数 < 1000）行为未经充分验证
- `use_regression_model = false` 提供回退能力，便于逐步迁移和 A/B 对比

---

## 八、相关代码位置速查

| 组件 | 文件 | 关键行号 |
|------|------|---------|
| 绝对阈值标签生成（问题根源）| `ml_module/ml_initializer.py` | ~799 |
| 正样本比例统计与日志 | `ml_module/ml_initializer.py` | ~873–896 |
| 正样本比例 WARNING（无处理）| `ml_module/ml_trainer.py` | ~117–121 |
| 评估函数（坍缩检测插入点）| `ml_module/ml_trainer.py` | ~345–391 |
| 动态权重计算 | `ml_module/neural_network.py` | ~466–584 |
| 历史 CI 数据合并 | `ml_module/ml_initializer.py` | ~634–690 |
| 回归标签生成（已有）| `ml_module/ml_regression_trainer.py` | 27–57 |
| 回归训练函数（已有）| `ml_module/ml_regression_trainer.py` | 60–181 |
| 回归预测函数（已有）| `ml_module/ml_regression_trainer.py` | 203–289 |
| 回归模型定义（已有）| `ml_module/ml_regression_model.py` | 35–416 |

---

## 九、代码库验证与计划审查

> 审查日期：2026-03-20
> 基于对代码库实际文件的逐一核查

### 9.1 基础设施声明验证

| 声明 | 验证结果 | 说明 |
|------|----------|------|
| `ANNRegressor` 存在且完备 | ✅ 已确认 | 35–416 行，含 HuberLoss、早停、评估、序列化 |
| `generate_regression_train_descriptors()` 完备 | ✅ 已确认 | 27–57 行，签名与计划描述一致 |
| `train_regression_model()` 完备 | ✅ 已确认 | 60–181 行，返回 `(ANNRegressor, metrics_dict)` |
| `predict_regression_model()` 完备 | ✅ 已确认 | 203–289 行，支持动态选样 |
| `__init__.py` 已导出回归函数 | ✅ 已确认 | 38–44 行，4 个函数均已列出 |
| 绝对阈值标签生成为根因 | ✅ 已确认 | `ml_initializer.py:799`，`>=cutoff_value` |
| 正样本比例统计在 ~873 行 | ✅ 已确认 | 873–896 行，仅日志无处理 |
| 正样本比例 WARNING 在 ~117 行 | ✅ 已确认 | 117–121 行，阈值 0.35，仅警告 |
| `evaluate_model()` 在 ~345–391 行 | ✅ 已确认 | 实际 330–392 行，轻微偏移 |
| `_calculate_dynamic_weights()` 在 ~466–584 行 | ✅ 已确认 | 完整实现，无高正样本比例保护 |
| 配置类中已有 `use_regression_model` 字段 | ❌ 不存在 | `MlConfig` 类（200–206 行）中无此字段 |
| 配置类中已有 `regression_min_clip` 字段 | ❌ 不存在 | `CalSettings` 和 `MlConfig` 均无此字段 |
| 主训练入口已有回归分支 | ❌ 不存在 | 当前仅分类路径 |

**结论**：计划中关于已有基础设施的声明全部准确，需新增的部分也已正确识别。

### 9.2 发现的问题与修订建议

#### 问题 1：配置字段归属不清晰（中等严重度）

**现状**：计划将 `use_regression_model` 放入 `MLConfig`（或等效配置类），但 `MlConfig`（200–206 行）当前存放的是模型调参字段（`overfitting_threshold` 等），而训练模式决策字段（`cutoff_value`、`sampling_ratio`）在 `CalSettings`（24–48 行）中。

**修订**：`use_regression_model` 和 `regression_min_clip` 应放入 `CalSettings`，与 `cutoff_value` 同级，保持"训练策略"与"模型调参"的关注点分离。`target_positive_ratio`（Phase 3）同理归入 `CalSettings`。

#### 问题 2：Step 4 引用的外部入口文件不在仓库内（中等严重度）

**现状**：`ml_CSFs_selection_scripts/ml_csf_choosing/train.py` 是用户侧编排脚本，不属于包的一部分。

**修订**：Step 4 应明确标注此文件为"外部调用方示例"，并补充说明：若用户通过其他方式调用 `train_model()`，需自行添加等效分支逻辑。可考虑在包内提供一个统一入口函数（如 `run_training_pipeline(config, ...)`）封装分支逻辑，减少用户侧改动。

#### 问题 3：描述符生成函数存在逻辑重复（低-中严重度）

**现状**：新增的 `generate_regression_descriptors_from_config()` 重复了 `generate_train_csfs_descriptors()` 中的 CI 数据加载、日志输出、描述符保存逻辑。

**修订**：建议重构为以下结构：
- 抽取公共前置逻辑（CI 数据加载 + 诊断日志）为内部函数 `_load_accumulated_ci_data(config, logger)`
- `generate_train_csfs_descriptors()` 调用公共函数 → 二值标签路径
- `generate_regression_descriptors_from_config()` 调用公共函数 → 回归标签路径

或者在现有函数中添加 `mode: Literal["classification", "regression"]` 参数，内部按模式分支。

#### 问题 4：首轮迭代缺少保护（中等严重度）

**现状**：计划在"设计决策说明"中提到回归路径在极小样本场景（第 1 轮，CSF < 1000）未经验证，但未在实施步骤中添加防护。

**修订**：在 Step 2 的 `generate_regression_descriptors_from_config()` 开头或 Step 4 的分支逻辑中添加：

```python
MIN_REGRESSION_SAMPLES = 1000
if len(accumulated_idxs) < MIN_REGRESSION_SAMPLES:
    logger.warning(
        "[回归路径] 累积样本数 (%d) 不足 %d，回退至分类路径",
        len(accumulated_idxs), MIN_REGRESSION_SAMPLES,
    )
    # 回退至分类路径
    ...
```

#### 问题 5：Phase 3 三方案缺少优先级排序（低严重度）

**现状**：方案 A（相对排名）、B（自适应截断）、D（动态权重）并列展示，未指明推荐默认方案。

**修订**：推荐 **方案 B（自适应截断）** 作为分类路径默认兜底：
- 实现最简单（仅在正样本比例越界时动态提升 cutoff）
- 不改变标签语义（仍为绝对阈值），对下游评估逻辑零影响
- 方案 A（相对排名）和 D（动态权重）标记为实验性备选

#### 问题 6：坍缩检测阈值硬编码（低严重度）

**现状**：Step 4.5 中 `abs(precision - pos_ratio) < 0.02 and recall > 0.95` 为硬编码常数。

**修订**：可接受作为初始实现。后续可考虑将阈值提取为配置字段或使用更鲁棒的指标（如预测熵），但不应阻塞 Phase 1 实施。

#### 问题 7：缺少回退时的文件清理策略（低严重度）

**现状**：若回归路径效果不佳需回退至分类路径，计划未说明已保存的回归描述符文件（`*_regression_full.npy`）和回归模型文件如何处理。

**修订**：回退时无需删除回归产物，两路径输出文件命名不冲突（回归使用 `_regression_full` 后缀）。建议在文档中明确说明此点即可。

### 9.3 修订后的实施顺序

```
Step 1   graspkit_config/ml_config_models.py   在 CalSettings（非 MlConfig）中添加
                                            use_regression_model / regression_min_clip
Step 2   ml_initializer.py                 新增 generate_regression_descriptors_from_config()
                                            ＋ 首轮样本数保护（< 1000 时回退分类路径）
                                            建议：抽取公共 CI 加载逻辑，避免重复
Step 3   ml_module/__init__.py             暴露新函数
Step 4   主训练入口（外部调用方示例）        添加 if use_regression_model 分支
                                            ＋ 注明此为用户侧改动，非包内代码
Step 4.5 ml_trainer.py                     添加坍缩检测日志（无破坏性）

─── 验证节点 ───

Step 5   （按需）Phase 3 默认启用方案 B（自适应截断），
         方案 A / D 标记为实验性备选
```

### 9.4 总体评价

本计划**技术方案合理、根因分析准确、分阶段策略得当**。核心思路——激活已有回归基础设施而非大幅重写——是正确的最小改动路径。上述 7 项修订建议均为改善性调整，不影响整体方案的可行性。建议在完成 Step 1–4.5 后，以第 4 轮迭代数据为基准进行 A/B 对比验证。
