# ML 模型坍缩修复实施计划

日期：2026-03-18

## 一、目标

基于 [`docs/ml_model_collapse_diagnosis.md`](/Users/yiqin/Documents/PythonProjects/GraspKit/docs/ml_model_collapse_diagnosis.md) 与当前仓库实现，制定一份可直接落地的实施计划，解决以下问题：

- 当前多标签分类训练在高正样本比例下坍缩为近似全正预测
- 推理阶段仍以固定 `0.5` 阈值为主，放大了训练坍缩问题
- 仓库虽已具备回归模块，但主迭代流程尚未接入
- 现有评估偏训练内诊断，缺少对“跨轮发现新重要 CSF”能力的约束

本计划分为两个层次：

- 短期止血：修复分类标签与采样逻辑，阻止模型继续坍缩
- 中期切换：将主流程逐步切换到回归排序方案

## 二、当前代码现状

### 2.1 标签生成仍使用绝对阈值

当前训练标签在 [`src/graspkit/ml_module/ml_initializer.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_initializer.py#L756) 的 `generate_train_csfs_descriptors()` 中生成：

```python
important_csfs_mask = (accumulated_ci_squared >= cutoff_value).T
```

这意味着：

- 标签定义完全依赖 `cutoff_value`
- 当 `cutoff_value` 过低时，正样本比例会随迭代膨胀
- 训练集标签分布会偏离“重要 CSF 稀有”的真实目标

### 2.2 训练阶段只报警，不干预

[`src/graspkit/ml_module/ml_trainer.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_trainer.py#L63) 的 `train_model()` 已统计正样本比例，并在过高时告警：

```python
if avg_positive_ratio > 0.35:
    logger.warning(...)
```

但当前没有：

- 自动收紧标签
- 中止训练
- 回退采样策略

### 2.3 推理阶段仍以固定阈值为主

[`src/graspkit/ml_module/ml_trainer.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_trainer.py#L395) 的 `predict_model()` 先做：

```python
y_unselected_prediction = (max_unselected_probability > 0.5).astype(int)
```

然后：

- 若阈值筛出的组态数足够，再按概率取 top-k
- 若阈值筛出为空，才 fallback 到纯排序

这会导致：

- 一旦模型概率整体偏高，几乎所有未选 CSF 都会进入“重要”集合
- 一旦模型概率整体偏低，又会筛不出任何组态
- 固定阈值被用于主决策，不适合当前排序检索任务

### 2.4 动态权重不是根因修复

[`src/graspkit/ml_module/neural_network.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/neural_network.py#L466) 的 `_calculate_dynamic_weights()` 会按正样本比例计算 `pos_weight`，但当前逻辑无法从根本上修复标签定义失真导致的坍缩。

结论：

- 根因在标签与采样策略
- 权重调整只能作为缓冲，不能替代标签修复

### 2.5 回归模块已存在但未接入主流程

仓库中已有以下文件：

- [`src/graspkit/ml_module/ml_regression_model.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_regression_model.py)
- [`src/graspkit/ml_module/ml_regression_trainer.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_regression_trainer.py)
- [`docs/regression_module_changes.md`](/Users/yiqin/Documents/PythonProjects/GraspKit/docs/regression_module_changes.md)

这些模块已经支持：

- 用 `log10(CI²)` 连续标签训练回归器
- 按预测分数排序选择 top-k CSF

但当前主流程仍未提供“分类 / 回归”模式切换入口。

## 三、总体实施策略

建议分三阶段实施：

1. 先修分类标签与推理策略，阻止模型继续坍缩
2. 再加入坍缩检测、训练保护和回退机制
3. 最后把回归模块接入主流程，并逐步转为默认方案

推荐原则：

- 先改数据与采样，再调模型
- 先保证流程不坏，再追求指标提升
- 每一步都必须带测试与可回放诊断

## 四、Phase 1：分类止血修复

### 4.1 配置层新增标签策略选项

修改文件：

- [`src/graspkit/data_IO/ml_cal_config_module.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/data_IO/ml_cal_config_module.py)

建议在 `CalSettings` 中新增以下字段：

- `label_mode: str = "relative_rank"`
- `target_positive_ratio: float = 0.10`
- `max_positive_ratio: float = 0.30`
- `adaptive_cutoff_enabled: bool = True`
- `collapse_detection_enabled: bool = True`

建议语义：

- `label_mode="absolute_cutoff"`：保留当前绝对阈值标签
- `label_mode="relative_rank"`：按每个能级的 CI² 排名取 top-q% 为正样本
- `target_positive_ratio`：目标正样本比例
- `max_positive_ratio`：超过后视为标签失真
- `adaptive_cutoff_enabled`：在绝对阈值模式下允许自动收紧

同时补充字段校验：

- `target_positive_ratio` 和 `max_positive_ratio` 必须在 `(0, 1)` 内
- `target_positive_ratio <= max_positive_ratio`

### 4.2 在标签生成阶段实现相对排名标签

修改文件：

- [`src/graspkit/ml_module/ml_initializer.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_initializer.py#L756)

改动目标：

- 把当前硬编码的绝对阈值标签改为策略分支
- 默认采用 `relative_rank`

建议实现逻辑：

1. 读取 `accumulated_ci_squared`
2. 若 `label_mode == "relative_rank"`：
   - 对每个能级单独计算百分位阈值
   - 生成每个能级自己的正样本 mask
3. 若 `label_mode == "absolute_cutoff"`：
   - 保留当前 `cutoff_value` 逻辑
4. 最后再与 `diff_ci_cutoff` 生成的增广标签做 OR 合并

实施注意事项：

- `relative_rank` 必须按能级独立处理，不能用全局统一阈值
- 需要考虑 `CI²` 为全零或极端稀疏时的边界情况
- 应保留每能级阈值到日志中，便于复盘

### 4.3 在标签生成阶段补充诊断统计

修改文件：

- [`src/graspkit/ml_module/ml_initializer.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_initializer.py#L756)

新增日志内容：

- 每个能级的正样本比例
- 平均正样本比例
- `CI²` 的 `min/max/mean`
- `p50/p90/p95/p99/p99.9`
- 若仍使用绝对阈值，则输出几个候选阈值下的正样本占比

目的：

- 为后续调参和回放提供依据
- 快速判断标签是否再次发生膨胀

### 4.4 在绝对阈值模式下加入自适应保护

修改文件：

- [`src/graspkit/ml_module/ml_initializer.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_initializer.py#L756)

逻辑：

- 仅当 `label_mode="absolute_cutoff"` 且 `adaptive_cutoff_enabled=True` 时启用
- 若平均正样本比例超过 `max_positive_ratio`
- 则基于当前 `accumulated_ci_squared` 的分布自动抬高阈值
- 重新生成标签并重新统计

这一机制的定位是：

- 兜底保护
- 不是主方案

## 五、Phase 2：采样与评估逻辑修复

### 5.1 推理改为“排序优先”

修改文件：

- [`src/graspkit/ml_module/ml_trainer.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_trainer.py#L395)

建议改法：

- 保留 `max_unselected_probability = max(probability across levels)`
- 不再先用 `> 0.5` 进行主筛选
- 直接按 `max_unselected_probability` 降序排序
- 直接取 `new_sampling_CSFs_num` 个 top-k

保留内容：

- 仍可记录 `> 0.5` 的数量作为诊断指标
- 但不再让固定阈值决定主采样结果

这样做的收益：

- 与排序任务本质一致
- 能减少概率校准偏移带来的不稳定性
- 与回归版 `predict_regression_model()` 的行为对齐

### 5.2 拆分“训练正样本”和“物理验证正样本”

当前风险：

- 如果训练标签改成 `relative_rank`，但 `verified_important_idxs` 仍只按 `cutoff_value` 计算，训练标准和评估标准会混杂

建议拆分两个概念：

- `training_positive_mask`：供模型训练使用
- `physics_verified_idxs`：按物理阈值定义的重要组态，供汇报和分析使用

涉及文件：

- [`src/graspkit/ml_module/ml_initializer.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_initializer.py)
- [`src/graspkit/ml_module/ml_trainer.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_trainer.py#L395)

目标：

- 训练策略可以演化
- 物理评价口径保持稳定

### 5.3 增加训练后坍缩检测

修改文件：

- [`src/graspkit/ml_module/ml_trainer.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_trainer.py#L324)

建议在 `evaluate_model()` 中加入以下规则：

- `abs(precision - avg_positive_ratio) < eps`
- `recall > 0.95`
- 预测正例比例极高，或概率分布方差过低

触发后行为建议：

- 记录 `CRITICAL`
- 在结果数据中写入 `is_collapsed`
- 返回一个上层可识别的状态位

### 5.4 增加坍缩后的回退动作

建议在主流程中预留回退行为：

- 若模型坍缩，则本轮不使用阈值分类结果扩样
- 回退为按概率 top-k 采样或物理已验证组态扩样
- 必要时强制切换为更严格标签配置重新训练

这里的原则是：

- 检测不能只是日志
- 必须影响本轮采样行为

## 六、Phase 3：回归模块接入主流程

### 6.1 增加任务模式开关

修改文件：

- [`src/graspkit/data_IO/ml_cal_config_module.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/data_IO/ml_cal_config_module.py)

建议新增：

- `ml_task_mode: str = "classification"`

可选值：

- `classification`
- `regression`

### 6.2 建立统一的主流程分派

需要在主迭代调用链中增加统一分派逻辑。

目标行为：

- 当 `ml_task_mode="classification"`：
  - 调用 `generate_train_csfs_descriptors()`
  - 调用 `train_model()`
  - 调用 `evaluate_model()`
  - 调用 `predict_model()`
- 当 `ml_task_mode="regression"`：
  - 调用 `generate_regression_train_descriptors()`
  - 调用 `train_regression_model()`
  - 调用 `evaluate_regression_model()`
  - 调用 `predict_regression_model()`

涉及模块：

- [`src/graspkit/ml_module/ml_initializer.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_initializer.py)
- [`src/graspkit/ml_module/ml_trainer.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_trainer.py)
- [`src/graspkit/ml_module/ml_regression_trainer.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_regression_trainer.py)
- 以及实际启动迭代训练的入口文件

### 6.3 回归模式的接入原则

回归模式要保持与现有累计数据机制兼容：

- 继续使用累积的 `accumulated_idxs`
- 继续使用累积的 `accumulated_ci_squared`
- 标签改为 `log10(max(CI², eps))`
- 推理输出保留每个能级的预测值
- 主排序分数取各能级预测值最大值

### 6.4 回归模式的评估重点

不建议只保留静态随机切分下的：

- MAE
- RMSE
- Spearman ρ

还应补充跨轮指标：

- 下一轮新发现重要 CSF 的 top-k 命中率
- 相对随机采样的富集倍数
- 对目标能级间距改善的贡献

## 七、Phase 4：动态权重逻辑的保底修补

修改文件：

- [`src/graspkit/ml_module/neural_network.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/neural_network.py#L466)

定位：

- 只做保底修补
- 不作为主修复方案

建议：

- 对高正样本比例分支增加显式保护
- 当 `pos_ratio > 0.35` 时记录异常状态
- 可限制 `pos_weight` 上界或切换到更保守模式

但该项优先级低于：

- 标签修复
- 排序采样
- 坍缩检测

## 八、测试计划

当前已有测试覆盖了迭代训练的基础接口，但还没有覆盖本次修复的关键风险点。

建议新增测试文件或扩展 [`tests/test_iterative_training_mode.py`](/Users/yiqin/Documents/PythonProjects/GraspKit/tests/test_iterative_training_mode.py)。

至少应包含以下测试：

- `test_relative_rank_labels_keep_target_ratio`
- `test_absolute_cutoff_triggers_adaptive_cutoff_when_positive_ratio_too_high`
- `test_predict_model_uses_top_k_ranking_instead_of_fixed_threshold`
- `test_evaluate_model_flags_all_positive_collapse`
- `test_training_positive_and_physics_verified_are_separated`
- `test_generate_regression_train_descriptors_outputs_log_ci_squared`
- `test_predict_regression_model_returns_top_ranked_unselected_csfs`

建议再增加一个小型伪数据集成测试：

- 构造 2 个能级、几百个 CSF 的 CI² 数据
- 让少量 CSF 明显高于背景
- 验证旧分类阈值路径容易退化
- 验证相对排名与回归排序都能把高贡献样本排到前面

## 九、推荐实施顺序

建议按以下顺序推进：

1. 在配置层新增标签策略与任务模式字段
2. 在 `ml_initializer.py` 中实现 `relative_rank` 标签与分布诊断
3. 在 `ml_trainer.py` 中将采样逻辑改为排序优先
4. 在 `ml_trainer.py` 中加入坍缩检测与状态输出
5. 补齐单元测试与小型集成测试
6. 用历史第 3/4/5 轮数据做离线回放验证
7. 若分类止血方案有效，再将回归模块接入主流程
8. 对比分类与回归在历史轮次上的 top-k 命中率和富集能力
9. 若回归明显优于分类，则将 `ml_task_mode` 默认切到 `regression`

## 十、验收标准

本问题不能只看训练内 AUC/F1，建议同时检查以下指标：

### 10.1 分类止血阶段

- 平均正样本比例稳定在目标区间，例如 `8% ~ 15%`
- 不再出现 `precision ≈ class_ratio` 且 `recall ≈ 1.0`
- Loss 曲线不再长时间卡死在固定值
- 训练后不会再频繁出现梯度全零

### 10.2 采样效果阶段

- top-k 新采样 CSF 中，后续真实超过物理阈值的比例提升
- 相比随机采样有显著富集
- 在相同采样预算下，能更快发现新重要 CSF

### 10.3 回归切换阶段

- 回归排序在历史回放中优于分类排序
- `Spearman ρ` 稳定为正且优于随机基线
- top-k 命中率和能级间距改善优于分类方案

## 十一、结论

从当前代码和已有文档看，最优先的工作不是继续调神经网络结构，而是修复：

- 标签定义
- 采样决策逻辑
- 坍缩检测与回退机制

短期最小可行方案是：

- 使用相对排名标签
- 将推理改为排序 top-k
- 增加坍缩检测

中期建议则是：

- 将已存在的回归模块正式接入主流程
- 逐步从“阈值分类”迁移到“连续分数排序”

这一路线与当前仓库结构兼容，且能最大程度复用现有代码资产。
