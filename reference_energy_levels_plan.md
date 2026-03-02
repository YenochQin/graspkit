# 计划：在 `cal_settings` 中添加参考能级参数以调控多目标能级间距

## 背景与问题分析

### 问题的根本原因

当 `spectral_term` 列表包含多个目标能级时（如同时优化 `3F` 和 `1G` 两个谱项），各迭代轮次之间的**能级间距**（即目标能级相互之间的相对能量差）会发生无约束的浮动。

**原因链路：**
1. 每轮 CI 计算使用不同的 CSF 子集，相关效应（correlation effects）的描述程度不同
2. 不同谱项对关联效应的响应速度不同，导致能级之间的相对能量差在迭代中自由漂移
3. 现有收敛判定（`evaluate_calculation_convergence`）仅检查**绝对能量**（`EnergyTotal`，单位 a.u.）在 3 轮间的标准差，未约束能级间的**相对能量差**
4. `check_energy_convergence` 同样是逐能级比较绝对能量，不涉及能级间相对关系

### 现有数据流（关键理解）

```
check_configuration_coupling()
    → selected_energy_data (Polars DataFrame)
        列: No, Pos, J, Parity, EnergyTotal (a.u.), EnergyLevel (cm⁻¹), splitting, configuration_raw
    → 保存为 {conf}_{loop_num}_correct_levels.csv

evaluate_calculation_convergence()
    → 读取最近3轮的 _correct_levels.csv
    → 以 EnergyTotal 计算跨轮标准差
    → 以 CSF 数量相对标准差 × 双重收敛判定
```

**关键发现**：`selected_energy_data` 中的 `EnergyLevel` 列已经是 cm⁻¹ 单位，由 mix_coef_loader 计算为
`EnergyLevel = (E_level - E_base) × Rydberg × 2`，即相对于计算输出中最低能级的值。
NIST 参考数据通常也以 cm⁻¹ 相对于原子基态给出，两者之差可直接比较（取差值后不受零点选择影响）。

---

## 方案设计

### 新增 config.toml 参数

```toml
[cal_settings]
spectral_term = [
    "2s(2).2p(6).3s(2).3p(6).3d(8)3F.4s(2)_3F",
    "2s(2).2p(6).3s(2).3p(6).3d(8)1G.4s(2)_1G",
]
# 与 spectral_term 一一对应，来自实验值（如NIST），单位 cm⁻¹
# 以数组中最小值为相对零点，比较各能级间的相对能量差
# 空列表 [] 表示不启用此功能
reference_energy_levels = [0.0, 13521.35]
# 允许计算值与参考值间距偏差的最大值（cm⁻¹）
# 0.0 = 仅监控记录，不影响收敛判定
reference_energy_threshold = 200.0
```

### "调控"的含义

- **监控（monitoring）**：每轮计算后，输出计算值与参考值的能级间距对比表（始终执行）
- **收敛门控（convergence gate）**：当 `reference_energy_threshold > 0` 时，
  在 `evaluate_calculation_convergence` 中增加第三个收敛判据：
  只有当所有能级间距与参考值偏差均 `< reference_energy_threshold` 时，才允许判定为收敛停止。
  否则继续迭代，直到能级间距向参考值收敛。

不引入回退（rollback）机制——能级间距偏大通常是早期迭代的正常现象，不应触发回退。

---

## 具体实现步骤

### 文件1：`CalSettings` 模型
**路径**：`/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/data_IO/ml_cal_config_module.py`

在 `CalSettings` 类的 `spectral_term` 字段后添加：
```python
reference_energy_levels: list[float] = []   # cm⁻¹, 与 spectral_term 等长
reference_energy_threshold: float = 0.0     # cm⁻¹, 0 = 仅监控
```

添加 `model_validator(mode="after")` 验证：
```python
@model_validator(mode="after")
def validate_reference_energy_levels(self) -> "CalSettings":
    if self.reference_energy_levels:
        if len(self.reference_energy_levels) != len(self.spectral_term):
            raise ValueError(
                f"reference_energy_levels 长度({len(self.reference_energy_levels)}) "
                f"必须与 spectral_term 长度({len(self.spectral_term)}) 相同"
            )
    return self
```

---

### 文件2：新增核心函数
**路径**：`/Users/yiqin/Documents/PythonProjects/GraspKit/src/graspkit/ml_module/ml_initializer.py`

新增函数 `check_reference_energy_agreement`，插入位置：`check_configuration_coupling` 之后，`check_energy_convergence` 之前。

**函数逻辑**：
```
输入:
  selected_energy_data: Polars DataFrame（仅含目标谱项的行）
  spectral_term: list[str]
  reference_energy_levels: list[float]  # cm⁻¹
  reference_energy_threshold: float     # cm⁻¹
  logger

1. 通过 configuration_raw 列匹配，提取每个 spectral_term 对应的 EnergyLevel (cm⁻¹)
   - 构建 {term: EnergyLevel} 字典
   - 若某 term 不在 selected_energy_data 中 → logger.error + return True（不阻断）

2. 以 reference_energy_levels 最小值为参考零点，计算参考相对能级差
   ref_base = min(reference_energy_levels)
   ref_rel[i] = reference_energy_levels[i] - ref_base

3. 以参考零点对应谱项的计算 EnergyLevel 为基准，计算计算值相对能级差
   base_term = spectral_term[argmin(reference_energy_levels)]
   calc_base = term_energy_dict[base_term]
   calc_rel[i] = term_energy_dict[spectral_term[i]] - calc_base

4. 计算各能级偏差
   discrepancy[i] = |calc_rel[i] - ref_rel[i]|

5. 用 tabulate 格式化输出对比表（谱项、参考值、计算值、偏差）到 logger.info

6. 判断:
   if reference_energy_threshold > 0:
       within_threshold = all(discrepancy[i] < reference_energy_threshold)
       return within_threshold  # False = 超出阈值（用于收敛门控）
   else:
       return True  # 纯监控模式

输出: bool（True=满足阈值或不启用阈值, False=超出阈值）
```

---

### 文件3：`evaluate_calculation_convergence` 函数
**路径**：同文件2

在函数签名中增加可选参数：
```python
def evaluate_calculation_convergence(
    config: MLCalConfig,
    logger: logging.Logger,
    cal_loop_csfs_count: int,
    current_energy_data: pl.DataFrame | None = None,  # 新增
) -> bool:
```

在第5步（读取阈值并判断收敛）后，增加第三判据：
```python
# === 6. 参考能级间距收敛判据（可选）===
ref_energy_converged = True  # 默认通过
if (
    config.cal_settings.reference_energy_levels
    and config.cal_settings.reference_energy_threshold > 0
    and current_energy_data is not None
):
    ref_energy_converged = check_reference_energy_agreement(
        current_energy_data,
        config.cal_settings.spectral_term,
        config.cal_settings.reference_energy_levels,
        config.cal_settings.reference_energy_threshold,
        logger,
    )
    logger.info(
        f"  参考能级间距收敛: {'满足' if ref_energy_converged else '未满足'}"
        f" (阈值: {config.cal_settings.reference_energy_threshold:.1f} cm⁻¹)"
    )
```

收敛判定改为三重 AND：
```python
if energy_converged and csfs_converged and ref_energy_converged:
    logger.info("能级、组态数量、参考能级间距均已收敛，停止计算")
    return False
```

---

### 文件4：`train.py` 调用
**路径**：`/Users/yiqin/Documents/PythonProjects/GraspKit-Tools/ml_CSFs_selection_scripts/ml_csf_choosing/train.py`

在 `check_configuration_coupling` 成功返回后、现有收敛检查之前，添加：
```python
# 每轮均进行参考能级间距监控（reference_energy_levels 非空时）
if config.cal_settings.reference_energy_levels:
    gk.check_reference_energy_agreement(
        selected_energy_data,
        config.cal_settings.spectral_term,
        config.cal_settings.reference_energy_levels,
        config.cal_settings.reference_energy_threshold,
        logger,
    )
```

在 `evaluate_calculation_convergence` 调用处传入 `current_energy_data`：
```python
should_continue = gk.evaluate_calculation_convergence(
    config,
    logger,
    train_data_counts.cal_csfs_count,
    current_energy_data=selected_energy_data,  # 新增参数
)
```

---

### 文件5：`config_with_comments.toml` 文档更新
**路径**：`/Users/yiqin/Documents/PythonProjects/GraspKit-Tools/ml_CSFs_selection_scripts/config_with_comments.toml`

在 `spectral_term` 配置项后添加注释和新字段说明。

---

## 导出（`__init__.py`）

需在两处 `__init__.py` 中注册新函数：

**文件A**：`GraspKit/src/graspkit/ml_module/__init__.py`
- 在 `from .ml_initializer import (...)` 块中添加 `check_reference_energy_agreement`
- 在 `__all__` 列表中添加 `"check_reference_energy_agreement"`

**文件B**：`GraspKit/src/graspkit/__init__.py`
- 在 `from .ml_module import (...)` 块中添加 `check_reference_energy_agreement`
- 在 `__all__` 列表中 `# ml_initializer` 注释下添加 `"check_reference_energy_agreement"`

---

## 补充讨论：两个深层设计问题

### 问题A：偏离参考值时触发回退 → 采用趋势回退策略

**策略**：仅当连续 3 轮能级间距偏差**单调递增**（持续远离参考值）时触发回退；
振荡或收敛则不触发，避免早期迭代频繁回退。

**实现位置**：在 `check_energy_convergence` 函数末尾（已用于其他回退判定）添加趋势检查：
1. 若 `reference_energy_levels` 非空且 `reference_energy_threshold > 0` 且 `cal_loop_num >= 3`：
2. 从最近 3 轮的 `{conf}_{loop_n}_correct_levels.csv` 中提取 `EnergyLevel`，计算各谱项对的间距偏差（与参考值的差值绝对值）
3. 若**所有谱项对**的偏差均满足 `dev[loop-2] < dev[loop-1] < dev[loop]`（单调递增）→ 返回 `False`（触发回退）
4. 否则返回 `True`（继续）

**数据来源**：直接读取已保存的 CSV 文件（无需新增存储）。

---

### 问题B：参考值引入训练过程 → 差动 CI 加权正样本

**物理依据**：
- **均动关联（symmetric correlation）**：CI² 对所有谱项贡献相近 → 主要改善绝对能量，不改变间距
- **差动关联（differential correlation）**：CI² 在不同谱项间差异大 → 主要影响能级间距

**正样本标签扩充规则**（在 `generate_train_csfs_descriptors` 中实现）：

对于每对谱项 (i, j)（i = 高能态，j = 低能态），若 `calc_gap_ij > ref_gap_ij`（计算间距偏大）：
- 需要更多贡献于**降低高能态**的 CSF，即对高能态贡献更大：`ci²[i,:] - ci²[j,:] >= diff_ci_cutoff`
- 这类 CSF 追加为正样本

若 `calc_gap_ij < ref_gap_ij`（计算间距偏小）：
- 需要更多贡献于**降低低能态**的 CSF：`ci²[j,:] - ci²[i,:] >= diff_ci_cutoff`
- 追加为正样本

**多谱项（N>2）处理**：遍历所有谱项对，对每对独立应用上述规则并取 OR 合并。

**新增配置参数**：在 `[cal_settings]` 中添加 `diff_ci_cutoff: float = 0.0`（0 = 不启用差动加权）。

**函数签名变更**：`generate_train_csfs_descriptors` 新增可选参数 `selected_energy_data: pl.DataFrame | None = None`，
在 `train.py` 调用处传入。函数内部从 `config.cal_settings.reference_energy_levels` 和 `selected_energy_data` 获取间距信息。

---

## 关键边界情况

| 情况 | 处理方式 |
|------|----------|
| `reference_energy_levels = []` | 完全跳过，不影响任何现有逻辑 |
| `reference_energy_threshold = 0.0` | 仅记录日志，`evaluate_calculation_convergence` 不加第三判据 |
| `spectral_term` 只有 1 个元素 | 所有相对差均为 0，无意义但不报错，直接返回 True |
| 某 spectral_term 在 selected_energy_data 中找不到 | 应由 `check_configuration_coupling` 已拦截，此处防御性处理 |
| `cal_loop_num < 3`（evaluate_calculation_convergence 不被调用） | 参考值比较仅通过 train.py 中的独立调用点执行（监控模式） |

---

## 验证方式

1. 单谱项配置（`reference_energy_levels = []`）运行不受影响
2. 多谱项配置不设阈值（`reference_energy_threshold = 0.0`）：日志每轮输出能级间距对比表
3. 多谱项配置设阈值：若偏差超出阈值，即使绝对能量已收敛，迭代继续
4. 验证 `len(reference_energy_levels) != len(spectral_term)` 时 Pydantic 报错
5. 验证输出日志格式正确（tabulate 表格）

---

## 修改文件汇总（共6个文件）

| 文件 | 改动内容 |
|------|----------|
| `GraspKit/src/graspkit/data_IO/ml_cal_config_module.py` | `CalSettings` 添加 3 个字段（`reference_energy_levels`, `reference_energy_threshold`, `diff_ci_cutoff`）+ 1 个 `model_validator` |
| `GraspKit/src/graspkit/ml_module/ml_initializer.py` | 新增 `check_reference_energy_agreement`；修改 `check_energy_convergence`（加趋势回退）；修改 `evaluate_calculation_convergence`（加第三收敛判据）；修改 `generate_train_csfs_descriptors`（加差动 CI 加权正样本） |
| `GraspKit/src/graspkit/ml_module/__init__.py` | 导出 `check_reference_energy_agreement` |
| `GraspKit/src/graspkit/__init__.py` | 导出 `check_reference_energy_agreement` |
| `GraspKit-Tools/ml_CSFs_selection_scripts/ml_csf_choosing/train.py` | 添加参考能级监控调用；向 `evaluate_calculation_convergence` 传 `current_energy_data`；向 `generate_train_csfs_descriptors` 传 `selected_energy_data` |
| `GraspKit-Tools/ml_CSFs_selection_scripts/config_with_comments.toml` | 添加 3 个新参数的注释文档 |
