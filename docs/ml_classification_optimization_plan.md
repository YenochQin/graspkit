# ML 分类任务优化计划

日期：2026-03-18

## 一、目标

基于以下现有文档与代码现状，整理分类路线的独立优化计划：

- `docs/ml_model_collapse_implementation_plan.md`
- `docs/ml_csf_training_issue_summary.md`

本计划聚焦两个核心问题：

1. 多能级任务中，不同能级的重要 CSF 可能并不相同，当前推理阶段过早合并，导致能级特异性丢失。
2. 当前标签仍通过绝对阈值定义：

```python
important_csfs_mask = (accumulated_ci_squared >= cutoff_value).T
```

这一定义过于依赖 `cutoff_value`，且不符合 CI² 头部高度集中的实际分布。

因此，分类路线的优化目标是：

- 将任务改造成真正的 level-aware classification
- 将标签从“绝对阈值”改造为“按 CI² 排序并按累计贡献定义重要”

---

## 二、当前问题概述

### 2.1 多能级训练与推理目标不一致

当前模型虽然输出多能级结果，但在推理和采样时仍会做全局合并：

- 用 `np.max(y_unselected_probability, axis=1)` 合并各能级概率
- 用 `np.any(correct_levels_ci_squared >= cutoff_value, axis=0)` 合并物理重要性

这会导致：

- level 1 的强信号可能掩盖 level 2 的关键组态
- 不同能级的重要组态结构被压扁成一个“全局重要性”
- 最终采样结果更偏向“对任一能级有点重要”的组态，而不是“对每个目标能级都有效覆盖”的组态

### 2.2 标签定义过于依赖绝对阈值

当前分类标签使用：

```python
important_csfs_mask = (accumulated_ci_squared >= cutoff_value).T
```

这会带来几个问题：

- 正样本比例高度受 `cutoff_value` 影响
- 随迭代累积，正样本容易膨胀
- 标签不能反映“少数头部 CSF 承担主要 CI² 贡献”的物理事实
- 相邻强弱不同的 CSF 会被压成同一个 0/1 标签

---

## 三、总体优化方向

分类路线建议拆成两部分推进：

1. 多能级任务优化
2. 标签定义优化

原则：

- 先修任务定义，再修模型
- 先修推理与采样逻辑，再调网络结构
- 训练标签必须尽量贴近最终 screening 目标

---

## 四、Phase A：多能级任务优化

### 4.1 目标

把当前“多标签训练 + 全局合并推理”改成真正的 per-level classification pipeline。

### 4.2 核心设计

#### A1. 保留每个能级独立的重要性标签

每个能级都应有自己的正样本定义：

- `CSF_i` 对 level 1 是否重要
- `CSF_i` 对 level 2 是否重要
- ...

不能在训练标签阶段就把所有能级合并成单一“重要 / 不重要”概念。

#### A2. 推理阶段不再使用全局 `max(probability)` 作为主决策

当前逻辑的问题在于：

- 若某个 CSF 只对某一个能级概率很高，就可能在全局 `max` 下被优先选中
- 但它可能对其他关键能级几乎没有帮助

建议改为：

- 每个能级分别输出概率排序
- 每个能级分别选择 top-k 或按该能级阈值筛选
- 再把多个能级的候选集合做并集

#### A3. 采样策略改为 per-level selection

建议采样时支持：

- `top_k_per_level`
- `threshold_per_level`
- `budget_per_level`

最终：

- 各能级独立选取
- 再 union
- 再做去重和覆盖性检查

#### A4. 评估指标改为 per-level

新增：

- 每能级正样本比例
- 每能级预测正例比例
- 每能级 Recall@k
- 每能级累计 CI² 覆盖率
- union 后最终采样集的去重率和覆盖率

### 4.3 实施步骤

#### Step A1

修改 `predict_model()`：

- 不再使用 `max + 0.5` 作为主采样逻辑
- 保留 per-level probability matrix
- 对每个 level 独立排序

#### Step A2

修改“已验证重要组态”的组织方式：

- 从单一 `verified_important_idxs` 拆分为 per-level verified masks 或 per-level idx sets

#### Step A3

实现 per-level selection sampler：

- 每个能级单独选 top-k
- 对结果取并集

#### Step A4

补充 per-level 日志和评估输出。

---

## 五、Phase B：标签定义优化

### 5.1 目标

把当前“CI² 超过绝对阈值就算重要”的标签定义改成：

- 对每个能级按 `CI²` 从大到小排序
- 按累计贡献达到目标阈值来定义重要组态

### 5.2 新标签定义

对于每个能级：

1. 将该能级对应的所有 CSF 按 `CI²` 降序排列
2. 计算累计和
3. 找到使累计贡献首次达到目标比例的最小前缀集合
4. 将这个集合中的 CSF 标为正样本

例如：

- `cumulative_ci_ratio = 0.90`
- 含义是：累计 CI² 达到 90% 的最小 CSF 集合作为“重要 CSF”

### 5.3 相比绝对阈值的优势

- 更符合 CI² 头部集中的真实物理分布
- 不依赖单一固定 `cutoff_value`
- 正样本数量会根据不同能级自适应变化
- 更贴近 screening 任务对“主贡献组态”的定义

### 5.4 需要控制的风险

1. 若累计阈值设得太高，例如 99%，正样本仍可能过多
2. 不同能级达到 90% / 95% 所需 CSF 数可能差别很大
3. 训练标签口径与物理验收口径需要分离

建议区分：

- `training_positive_mask`
- `physics_verified_mask`

前者用于训练，后者用于物理分析和报告。

### 5.5 建议新增配置

建议在 `CalSettings` 中新增：

- `label_mode: str = "cumulative_ci"`
- `cumulative_ci_ratio: float = 0.90`
- `cumulative_label_min_count: int = 5`
- `cumulative_label_max_ratio: float = 0.20`

同时保留：

- `label_mode="absolute_cutoff"`
- `label_mode="relative_rank"`
- `label_mode="cumulative_ci"`

其中建议默认主方案改为：

- `cumulative_ci`

### 5.6 实施步骤

#### Step B1

在 `ml_initializer.py` 中实现统一 label builder：

- `absolute_cutoff`
- `relative_rank`
- `cumulative_ci`

#### Step B2

对每个能级记录：

- 达到累计阈值所需 CSF 数量
- top 5 / top 10 / top 30 覆盖率
- 最后一个入选 CSF 的 CI² 值

#### Step B3

将差动 CI 增广标签与新标签做 OR 合并，但单独记录标签来源。

#### Step B4

在历史轮次上比较三种标签模式：

- 正样本比例
- 模型稳定性
- 下一轮新重要 CSF 的发现率

---

## 六、分类路线的评估方案

不再只看全局分类指标，而应重点看：

1. 每能级 Recall@k
2. 每能级累计 CI² 覆盖率
3. 每能级 top-N 重要组态召回率
4. 下一轮 RCI 验证后的命中率
5. 最终采样规模与预算利用率
6. union 后采样集对目标能级的覆盖是否均衡

---

## 七、建议实施顺序

建议按以下顺序推进：

1. 先修推理阶段的多能级合并问题
2. 再实现 per-level selection
3. 再实现 `cumulative_ci` 标签
4. 最后对比 `absolute_cutoff / relative_rank / cumulative_ci`

原因是：

- 当前最直接的问题是多能级信息在推理阶段被压平
- `cumulative_ci` 标签则是下一步从任务定义上修正分类目标

---

## 八、最终建议

分类路线如果继续保留，必须完成两件关键改造：

1. 从“全局重要性分类”改成“按能级的重要性分类”
2. 从“绝对阈值打标”改成“按累计 CI² 贡献打标”

否则，即使网络结构继续调整，任务定义本身仍会与真实 screening 目标错位。
