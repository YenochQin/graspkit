# ML 模型坍缩诊断与优化计划

> 诊断日期：2025-03
> 问题现象：第 4 轮迭代后模型无法预测出新组态，训练退化为全正预测

## ML 模型描述符构造

使用 rcsfs 本地库 （"/Users/yiqin/Documents/ProjectFiles/rCSFs"）将计算的csfs转换为描述符的X，将csfs通过rmcdhf方法或rci方法计算得到的每个csfs的ci^2是否大于阈值作为y，两者合为总体的描述符。

---

## 一、问题现象（日志摘录）

```
正样本数量: 226315 (占比: 0.4629)
Epoch [50/150],  Loss: 0.864200
Epoch [100/150], Loss: 0.864200
Epoch [150/150], Loss: 0.864200
[DEBUG] Epoch 149 - w1/b1/w2/b2/w3 梯度为0
AUC: 0.5114, F1: 0.6329, Accuracy: 0.3844
Precision: 0.4629, Recall: 1.0000
```

特征：Loss 不收敛、梯度全为 0、Recall=1.0、Precision≈正样本比例 → **模型坍缩为全正预测**。

---

## 二、根本原因链

```
cutoff_value = 1e-09（对 ci² 的阈值，对应 ci ≥ 3.16e-5，物理上几乎为零）
        ↓
所有参与 RCI 展开的 CSF 均满足 ci² ≥ 1e-09
        ↓
正样本比例 = 46.29%（且随迭代累积持续增大）
        ↓
动态权重 pos_weight = 1.58（不足以阻止模型坍缩）
        ↓
BCEWithLogitsLoss 的数学均衡点：输出常数 z ≈ 0.311，σ(z) ≈ 0.577 > 0.5
        ↓
所有样本均预测为正 → 梯度归零 → Loss 卡死 → 无法选出新组态
```

### 数学验证

均衡损失 = `pos_weight × p × (−ln σ) + (1−p) × (−ln(1−σ))`
= `1.58 × 0.463 × (−ln 0.577) + 0.537 × (−ln 0.423)`
= `0.401 + 0.462 = 0.863 ≈ 0.864` ✓ 与日志完全吻合

---

## 三、各层原因详解

### 3.1 `cutoff_value = 1e-09` 的物理含义

`cutoff_value` 作用于 **ci²**（`ml_initializer.py:799`）：

```python
important_csfs_mask = (accumulated_ci_squared >= cutoff_value).T
```

| ci² 阈值 | 对应 ci | 物理意义 |
|----------|---------|---------|
| `1e-09`（当前）| ≥ 3.16×10⁻⁵ | 波函数概率贡献 < 0.000003%，数值噪声级 |
| `1e-06` | ≥ 0.001 | 贡献 0.0001%，边缘有意义 |
| `1e-04` | ≥ 0.01  | 贡献 0.01%，通常认为的最低可信阈值 |
| `1e-03` | ≥ 0.032 | 贡献 0.1%，明确重要 |

注意：日志中 `混合系数 ≥ 1e-09` 措辞具有误导性，实际比较的是 **ci²**，不是原始 ci 值。

### 3.2 正样本比例随迭代膨胀

每轮 `ci_idx_data_processor` 将当前轮 CSF 合并进累积集，由于阈值极低，新增 CSF 几乎全部标记为正样本，导致正样本比例随轮次单调递增，无法收敛。

### 3.3 动态权重计算在高正样本比例时失效

`neural_network.py: _calculate_dynamic_weights`，当 `pos_ratio = 0.463`（≤ 0.5，进入 Case A）：

```python
raw_weight = 1.0 / 0.463 = 2.16
calculated_weights = 1.0 + (2.16 - 1.0) × 0.5 = 1.58  # 不足以阻止坍缩
```

代码在 `ml_trainer.py:117` 触发了 WARNING，但没有任何纠正行动：

```python
if avg_positive_ratio > 0.35:
    logger.warning("当前正样本比例过高(%.4f)，模型可能退化...")  # 只警告，不处理
```

### 3.4 绝对阈值的结构性缺陷

在 <1% 子空间的 RCI 计算中，CSF 之间相互"竞争"波函数权重，子空间越小，每个 CSF 分配到的 ci² 越大。随着迭代扩大子空间，ci² 被摊薄，**相同的绝对阈值在不同轮次对应的物理意义不一致**。

---

## 四、优化方案

### 方案 A：相对排名标签（推荐，根本性修复）

**位置：`ml_initializer.py: generate_train_csfs_descriptors`**

将绝对阈值改为每轮按百分位排名定义正样本，确保正样本比例始终稳定：

```python
# 当前（绝对阈值，不稳定）
important_csfs_mask = (accumulated_ci_squared >= cutoff_value).T

# 建议（相对排名，稳定）
# 对每个能级，取 ci² 排名前 target_positive_ratio 的 CSF 作为正样本
target_positive_ratio = 0.10  # 可配置，建议 10%
per_level_threshold = np.percentile(
    accumulated_ci_squared, 100.0 * (1 - target_positive_ratio), axis=1
)  # shape: (n_levels,)
important_csfs_mask = (accumulated_ci_squared >= per_level_threshold[:, None]).T
```

优点：
- 正样本比例与子空间大小无关，始终稳定
- 模型学习"相对重要性"，随迭代标签质量自然提升
- 无需手动调整 `cutoff_value`

### 方案 B：自适应截断保护（防御性修复，可与 A 并用）

**位置：`ml_initializer.py: generate_train_csfs_descriptors`，约第 873 行**

当正样本比例超过上限时，自动提升截断值：

```python
MAX_POSITIVE_RATIO = 0.30
if positive_ratio > MAX_POSITIVE_RATIO:
    target_ratio = 0.15
    max_ci_per_csf = accumulated_ci_squared.max(axis=0)
    nonzero_ci = max_ci_per_csf[max_ci_per_csf > 0]
    percentile_rank = 100.0 * (1.0 - target_ratio)
    adaptive_cutoff = max(float(np.percentile(nonzero_ci, percentile_rank)), cutoff_value)
    if adaptive_cutoff > cutoff_value:
        logger.warning("正样本比例过高(%.4f)，cutoff 自动从 %.2e 提升至 %.2e",
                       positive_ratio, cutoff_value, adaptive_cutoff)
        important_csfs_mask = (accumulated_ci_squared >= adaptive_cutoff).T
        # 重新统计
        positive_count = int(np.sum(important_csfs_mask))
        positive_ratio = positive_count / total_elements
```

### 方案 C：模型坍缩检测与中止（紧急保护）

**位置：`ml_trainer.py: evaluate_model` 或 `train.py`**

训练后若检测到坍缩，记录 CRITICAL 并触发回退：

```python
# 坍缩判定：Precision ≈ 正样本比例 且 Recall ≈ 1.0
is_collapsed = (
    abs(labeled_precision - avg_positive_ratio) < 0.02
    and labeled_recall > 0.95
)
if is_collapsed:
    logger.critical("检测到模型坍缩（全正预测）：Precision=%.4f ≈ 正样本比例=%.4f，"
                    "Recall=%.4f。标签定义需要修正。",
                    labeled_precision, avg_positive_ratio, labeled_recall)
    # 可在此处触发：重置模型、提高 cutoff、或中止本轮
```

### 方案 D：修复 `_calculate_dynamic_weights` 高正样本比例分支

**位置：`neural_network.py: _calculate_dynamic_weights`，约第 519 行**

当前 Case A（pos_ratio ≤ 0.5）在 ratio=0.46 时给出 pos_weight=1.58，不足以阻止坍缩。建议增加高比例保护：

```python
# 在 Case A 末尾追加：当正样本比例过高时强制提升负类权重
if mask_early.any():
    # ... 现有逻辑 ...
    # 当比例超过 0.35 时，额外对负类加权（等价于降低正类权重）
    high_ratio_mask = mask_early & (pos_ratios > 0.35)
    if high_ratio_mask.any():
        # 强制 pos_weight < 1.0，让模型偏向负类学习
        calculated_weights[high_ratio_mask] = torch.clamp(
            calculated_weights[high_ratio_mask], max=0.5
        )
```

---

## 五、优先级与实施顺序

| 优先级 | 方案 | 改动位置 | 风险 |
|--------|------|---------|------|
| P0（最高）| A：相对排名标签 | `ml_initializer.py` | 改变训练标签定义，需验证收敛行为 |
| P1 | C：坍缩检测 | `ml_trainer.py` | 只增加检测，无副作用 |
| P2 | B：自适应截断 | `ml_initializer.py` | 防御性，A 实施后可作为兜底 |
| P3 | D：权重修复 | `neural_network.py` | A 实施后正样本比例正常，此项影响变小 |

**最小改动路径**：先只实施 A + C，验证效果后再考虑 B、D。

---

## 六、相关代码位置速查

| 问题点 | 文件 | 行号（近似） |
|--------|------|------------|
| 标签生成（绝对阈值） | `ml_module/ml_initializer.py` | ~799 |
| 正样本比例统计与日志 | `ml_module/ml_initializer.py` | ~873–896 |
| 正样本比例 WARNING（无处理）| `ml_module/ml_trainer.py` | ~117–121 |
| 动态权重计算 | `ml_module/neural_network.py` | ~466–584 |
| 坍缩评估（诊断入口）| `ml_module/ml_trainer.py` | ~345–391 |
| 历史 CI 数据合并 | `ml_module/ml_initializer.py` | ~634–690 |

---

## 七、背景：为何不能简单提高 `cutoff_value`

提高绝对阈值（如 `cutoff_value = 1e-4`）虽然能减少正样本，但：

1. **正样本可能确实变少**：在 244K 样本中如果只有 1-2% 正样本（~2,400 个），类别失衡依然存在，只是方向相反
2. **跨轮次的物理含义不一致**：小子空间时 ci² 被压缩放大，大子空间时被摊薄，绝对阈值无法适应这种变化
3. **根本解决方案是相对排名（方案 A）**，而不是寻找某个"正确"的绝对阈值

---

## 八、验证指标

实施方案 A 后，预期观察到：

- 正样本比例：稳定在目标值（如 10%），不随轮次膨胀
- AUC：> 0.70（高于 0.5114 的随机水平）
- Precision 与 Recall 解耦：不再出现 Precision=class_ratio、Recall=1.0 的坍缩特征
- Loss 曲线：应随 epoch 单调下降，不再卡在固定值


##################

这份数据反而更支持把主方案切到“多输出回归”，不是削弱这个结论。

我直接看了 /Users/yiqin/Documents/PythonProjects/graspkit-tools/tests/rmix_data/rmix_data.npy。它是 shape=(2, 393887)，两个能级各自一条 CI 序列。关键结论是：

1. CI² 的跨度极大，单能级上从约 1e-20 到 1.6e-1。
2. max(CI²) 的 log10 分布大约从 -16 到 -0.8，跨了 15 个数量级。
3. cutoff=1e-9 时，正样本比例已经很高：
    - level 0: 0.7518
    - level 1: 0.8287
    - 两个能级任一满足: 0.9493
4. 每个能级的 CI² 质量极度头部集中：
    - top 5 个 CSF 就吃掉约 50% 总 CI²
    - top 26/31 个就吃掉约 90%
    - 但要到 1.3 万到 1.6 万个才到 99%
5. 两个能级不能合并成一个简单单标量来训：
    - corr(CI²) ≈ 0.789，说明有共性
    - 但支配比 max/min 的中位数都有 2.5，99 分位到 3.6e3
    - 也就是大量 CSF 对两个能级的重要性差很多，必须保留“每个能级一个输出”

所以你的这个担心是对的：如果直接在原始 CI² 空间做统一阈值分类，会被数量级分布和多能级差异一起放大。

更合理的具体做法

用现有的多输出回归链路，目标定义为每个能级各自预测：

y_level = log10(max(CI^2, eps))

仓库里已经有这个实现方向：

- ml_regression_trainer.py:27
- ml_regression_trainer.py:203:203

这正好适配你这个数据形态，因为它：

- 压缩 15 个数量级，训练更稳定
- 不丢掉强弱排序信息
- 天然支持多能级输出，不需要把两个能级粗暴合并成一个 0/1 标签

两个能级应该怎么一起分析

不要先把两个能级合成单标签。应该分两层处理：

1. 训练层：
    - 输出 shape 保持 (n_csfs, n_levels)
    - 每个能级各自学习 log10(CI²)
2. 选样层：
    - 基础版用 max(pred_log_ci2_across_levels) 排序选 top-k
    - 如果你的目标更偏能级间距，可以再加一个差动分数，例如
    score = max_pred + lambda * max_pairwise_gap
    - 这样既保留“对任一能级重要”的 CSF，也保留“对能级差有差动贡献”的 CSF

为什么现在不建议优先做分位数分类

分位数分类比固定阈值强，但对你这类数据仍然有两个缺点：

- top few 个和中等贡献者都会被压成同一个正类
- 两个能级的重要性差异会被二值化吃掉

在你这份数据里，头部太尖，尾部太长，二值化损失很大。

我现在的最终建议

最合理方案仍然是：

- 主训练目标改为多输出回归 log10(CI² + eps)
- 推理改为按分数直接 top-k 选样
- 保留每个能级单独输出，不要先合并标签
- 若要兼顾能级间距，再在选样分数里叠加差动项，而不是回到硬阈值分类

如果你要继续，我下一步可以直接基于这份分析，把当前迭代主流程改成“多能级 log-CI² 回归 + top-k 采样”，并补上针对两能级分布的诊断日志与测试。