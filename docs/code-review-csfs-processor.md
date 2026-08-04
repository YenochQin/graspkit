# Code Review: `src/graspkit/CSFs_processor/`

**日期**: 2026-08-04
**范围**: `CSFs_processor/__init__.py`、`CSFs_choosing.py`(1185 行)、`CSFs_compress_extract.py`(380 行)，共 1621 行
**方法**: 通读全部源码；交叉核对依赖模块 `grasp_data_extractor/rmix_data_processor.py`、`data_IO/loaders/mix_coef_loader.py`、`utils/data_modules.py`、`utils/tool_function.py`；用 CodeGraph 1.5.0 检查调用链与影响面；用 `ruff check --select ALL`、`mypy --strict`、`pytest` 做实证核验
**状态**: 仅审核，尚未应用任何修改

---

## Overview

本次审核针对 `CSFs_processor` 目录下的两个源文件展开，目的是为后续"固化"（稳定化）该模块提供依据。整体上模块处于持续硬化过程中（`docs/modify_logs/strict_mypy_cleanup_20260310.md` 记录过 `data_IO` 与 `CSFs_processor` 的历史类型问题收敛），`mypy --strict` 目前完全干净，多数公开函数文档完整。但仍存在若干真实的正确性缺陷、一批死代码/重复逻辑，以及一个规模问题：`CSFs_choosing.py` 1185 行，塞了三块不相关的职责。

### 验证记录

| 检查 | 命令 | 结果 |
|------|------|------|
| 类型检查 | `mypy --no-incremental src/graspkit/CSFs_processor/` | ✅ `Success: no issues found in 3 source files` |
| 现有测试基线 | `pytest tests/test_rmix_data_processor.py -q` | ✅ `22 passed` |
| 全量静态检查 | `ruff check --select ALL --statistics src/graspkit/CSFs_processor/` | 320 条命中（详见下方分类，多数为噪音规则） |
| CodeGraph 索引 | `codegraph status graspkit` | ✅ 47 files / 672 nodes / 1298 edges；精确符号查询可用 |
| Codex 二次复核 | `ruff check ../graspkit/src/graspkit/CSFs_processor`；`mypy --no-incremental ...`；`pytest ../graspkit/tests/test_rmix_data_processor.py -q` | ✅ Ruff 通过；mypy 通过；22 passed |

---

## High Priority Issues

### 1. `CSFs_sort_by_mix_coefficient` 对单 ASF（1D）输入必崩溃

`CSFs_choosing.py:460`

```python
coeff_lengths = {len(coeff) for coeff in mix_coefficients}
```

若 `mix_coefficients` 是 1D 数组（单 ASF），迭代会产出 `numpy.float64` 标量，`len()` 调用抛 `TypeError: len() of unsized object`。`rmix_data_processor.py` 的 `_normalize_asf_indices`/`_as_1d_or_2d_float_array` 明确将 1D（单 ASF）视为合法输入形状，docstring "Coefficient array whose length matches CSFs_block" 也暗示支持这种用法，实现与文档/兄弟模块的假设不一致。

**建议**：在 `ci_squared(np.asarray(mix_coefficients))` 之前，先用 `np.atleast_2d` 归一化，复用函数内已有的 2D 处理路径。

### 2. `radom_choose_csfs` 类型标注与实际用法矛盾，非法输入处理不完整

`CSFs_choosing.py:504-565`

- `508`: `selected_csfs_idxs: list[list[str]] = []` — 可变默认参数（ruff `B006`），且函数体内当作扁平 `list[int]` 使用（`set()`、与 int64 数组 `np.concatenate`），类型标注与实际语义矛盾。这是 `__all__` 导出的公开 API。
- `525-532`: `method` 只处理 `"ratio"`/`"quality"` 两个分支，无 `else`；非法值会导致 `total_needed` 未绑定，第 532 行抛出令人费解的 `UnboundLocalError` 而非清晰的 `ValueError`（`Literal` 类型只在静态检查时有效）。

**建议**：`selected_csfs_idxs: list[int] | None = None` + 函数内 `if selected_csfs_idxs is None: selected_csfs_idxs = []`（与同文件其他函数已用的模式一致）；给 `method` 补 `else: raise ValueError(...)`。

### 3. `subshell_charged_state` 正则错误，遗漏高角动量壳层

`CSFs_compress_extract.py:26-28`（对照 `39-66` 的 `if_subshell_full_charged`）

```python
r"([0-9]*)([s,p,d,f,g][\s,-])\( (\d+)\)"
```

字符类 `[s,p,d,f,g]` 混入了字面逗号（应为 `[spdfg]`），且只覆盖到 g 壳层；而 `if_subshell_full_charged` 明确定义了 h/i 壳层的满壳电子数（`"h-":10, "h ":12, "i-":12, "i ":14`）。只要配置中出现 h 或 i 壳层，`re.findall(...)[0]` 会因空匹配抛 `IndexError`。

**建议**：修正字符类为 `[spdfg]`（如需支持 h/i 应一并扩展），最好让两处壳层字母表来自同一个常量，避免再次分叉。

### 4. CSF 描述符批处理对坏数据静默丢弃

`CSFs_compress_extract.py:317-330`

```python
try:
    if len(csf_item) != 3:
        print(...); continue
    descriptor = parse_csf_2_descriptor(...)
    all_descriptors.append(descriptor)
except Exception as e:
    print(f"Error processing CSF ...: {e}")
    continue
```

单条 CSF 解析失败（行数不对、壳层查不到）时打印后 `continue`，既不计数也不上抛（ruff `BLE001` 标记了过宽的 `except Exception`）。对 ML 训练数据管线而言，这可能悄悄削减训练集而没有任何硬失败信号。

**建议**：收集失败项（索引 + 原因），处理完成后如有失败则抛出聚合异常或至少通过 `logger.error` 输出汇总，不要单纯 `continue`。

### 5. `extract_csfs_units` 合并多来源 CSFs 时未校验 header 一致性

`CSFs_choosing.py:837-841`

```python
_write_csfs_blocks_to_cfile(
    selected_blocks[0].header_lines,
    [block.csfs_df for block in selected_blocks],
    output_file,
)
```

只用第一个 unit 的 header 写出合并文件，未校验其余 unit 的 header（轨道基组）是否一致。若各来源基组不同，会静默生成一个结构不自洽的 `.c` 文件，这类损坏往往要到后续 GRASP 计算失败才会暴露。

**建议**：合并前校验所有 `selected_blocks[i].header_lines` 相等（或至少 peel-subshell 定义一致），不一致时抛出明确错误。

---

## Medium Priority Issues

### 6. 日志占位符未插值

`CSFs_choosing.py:254`：`logger.info("Block {block + 1}: 包含 {len(block_csfs)} 个 CSF")` 漏了 `f` 前缀，花括号内容原样打印，日志完全失去信息量。

### 7. 无意义的 `position_set` 判断（死逻辑）

`CSFs_choosing.py:313, 325-328`：`position_set` 由 `block_asfs_position` 构建，随后遍历同一个 `block_asfs_position` 判断 `if asf_idx in position_set`，恒真。注释"用 set 将 O(n) 查找降为 O(1)"暗示这里本该有过滤作用，实际没有，容易误导后来者。

**建议**：删除该判断，或如果确实需要过滤，修正为对完整 ASF 范围按 `position_set` 筛选。

### 8. `CSFs_compress_extract.py` 全文件用 `print` 而非 `logger`

第 266、320-321、328-330、338-340、371-374 行，共 11 处（ruff `T201` 计数一致）。该文件未创建 logger，与 `CSFs_choosing.py` 已建立的 logging 方式不一致。

### 9. `rcsfs` 是隐式可选依赖

`CSFs_choosing.py:615-632` 的 `convert_csfs` 在函数体内才 `from rcsfs import convert_csfs`。`rcsfs` 未出现在本仓库 `pyproject.toml` 的 `dependencies` 中（只在 `graspkit-tools` 里作为 path dependency 声明）。单独安装 `grasp-kit` 的用户传入 `.c` 文件时会直接收到裸的 `ModuleNotFoundError`，没有任何指引。

**建议**：在 `pyproject.toml` 增加可选依赖组，和/或对 import 包一层 `try/except ImportError` 给出可操作的报错信息。

### 10. 4 处 `zip()` 未加 `strict=`

`CSFs_choosing.py:112, 383, 684`、`CSFs_compress_extract.py:235`（ruff `B905`）。目前均有前置长度校验或结构性不变量兜底，暂不构成活跃 bug，但一旦相关不变量在未来被改动，会静默截断而非报错。

**建议**：统一加 `strict=True`，低成本的防御性加固。

### 11. 两个函数完全无引用（死代码）

`_unit_from_pipeline_config`（`CSFs_choosing.py:998`）、`_output_from_pipeline_config`（`CSFs_choosing.py:1057`）。全仓库搜索只在各自 `def` 处命中一次。从签名看（接受 `config: Any` 并用属性访问），这是被 `_pipeline_paths_from_raw_config` 取代前的旧实现（后者 docstring 自述"Resolve the legacy Tools pipeline paths **without importing Tools models**"）。同一职责两份实现并存是维护风险。

**建议**：确认无外部依赖后直接删除。

### 12. 无用的通配符导入

`CSFs_choosing.py:27`：`from ..utils.tool_function import *`。该模块导出的 4 个名字（`str_subshell_2_kappa`/`doubleJ_to_J`/`chunk_string`/`LS_shell_full_charged`）在文件里一个都没被引用（ruff `F403` 标记），可整行删除。

### 13. `single_asf_mix_square_above_threshold` 公开性边界不清

`CSFs_choosing.py:38`：未进入 `__init__.py`/`__all__`，却被 `tests/test_rmix_data_processor.py` 直接从 `CSFs_choosing` 子模块导入并测试。目前不清楚它是内部实现细节还是公开能力——按现状改动/删除它不会被当作破坏性变更对待，但确实有外部测试依赖它。

**建议**：要么加入 `__all__` 正式导出，要么改名加下划线前缀并让测试改走公开入口。

### 14. 重复的定宽解析逻辑 + 魔法数字

`CSF_subshell_split`（`CSFs_compress_extract.py:69-86`）手写按 9 字符分块，和同文件已在用的 `chunk_string(s, 9)` 是同一件事的两份实现；`parse_csf_2_descriptor` 里 `coupling_line_raw[4:-5]`（216-218 行）、5 行 header（748、792 行）、3 行 CSF 记录（319 行）等魔法数字都是 GRASP 定宽格式的隐含约定，没有命名常量或格式说明。

**建议**：`CSF_subshell_split` 直接复用 `chunk_string`；给 `4`/`5`/`9` 等定宽偏移量提取命名常量。

---

## Low Priority Issues

### 15. `CSFs_choosing.py` 规模超出项目约定上限

1185 行，超出项目 800 行上限的约定，且可清晰拆成三块：分析类函数（阈值筛选、coupling-J 收集，≈1-410 行）、排序/随机选择工具（≈415-566 行）、以及一整套与前两者零函数共享、基于 TOML 的提取 CLI/流水线（`CsfsSelectionUnit` 起到文件末尾，≈600 行）。

**建议**：把第三块拆成独立模块，例如 `CSFs_extraction_pipeline.py`。

### 16. 命名风格不统一

ruff `pep8-naming` 系列在这两个文件里共命中 37 处（`N802` 函数名 12、`N803` 参数名 14、`N806` 局部变量 9、`N999` 模块名 2），如 `CSFs_block_get_CSF`、`CSf_idx`、`J_str` 等。这些多是导出的公开 API，重命名属破坏性变更，建议列入长期计划而非立即处理。

### 17. `ruff` 配置形同虚设

`pyproject.toml:90-91` 只开了 `select = ["NPY201"]`。这意味着上面 High/Medium 里的第 2、7、10、12 条这些真实问题目前不会被 CI 拦截。

**建议**：扩展为实用规则集，如 `E, F, B, I, UP, N`。

### 18. 其余 ruff `--select ALL` 命中的噪音规则（仅记录，不建议启用）

`COM812`(36)/`TRY003`(32)/`RUF003`(32，多为中文全角标点误报)/`D413`(25)/`EM102`(25) 等共占 320 条命中里的大头。`COM812` 与 formatter 冲突，`RUF001-3` 对中英混排的科学计算代码库不适用，`TRY`/`EM`/`D` 系列风格约束与本仓库现状不符，不建议纳入 CI。

---

## Codex 二次复核新增问题

以下条目是在复核上述报告、CodeGraph 调用链以及当前测试后新增；编号延续原报告。它们未被现有 22 个相关测试覆盖。

### 19. `level_indices` 被同时当作“能级标识”和“系数矩阵行号”使用（High）

`CSFs_choosing.py:94-122, 365-405`；对照 `data_IO/loaders/mix_coef_loader.py:160-195` 与 `grasp_data_extractor/rmix_data_processor.py:324-366`。

`MixCoefLoader` 将文件中的 `ivec - 1` 原样保存为 `MixCoefficientBlock.level_indices`，但 `batch_asfs_mix_square_above_threshold` 默认把这些值直接用于：

```python
block.mix_coefficients[selected_positions]
```

`batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum` 也先用 `level_indices` 校验 `selected_positions`，随后把同一组值当作矩阵行号传入下层函数。相比之下，兄弟模块 `_normalize_asf_indices` 明确把 ASF 选择解释为 `[0, n_asf)` 的矩阵行号。当前测试辅助函数 `_mix_block` 固定使用 `np.arange(n_asf)` 作为 `level_indices`，因此掩盖了两种概念不相等时的越界或错选风险。

**建议**：明确区分 `level_id` 与 `asf_row_index`。若公开参数表示矩阵行号，应按 `mix_coefficients.shape[0]` 校验且默认使用 `np.arange(n_asf)`；若表示文件中的 level id，则先建立 `level_id -> row_index` 映射再取矩阵。增加非连续 `level_indices`（如 `[2, 5]`）的回归测试。

### 20. 索引文件会被无条件转成整数，越界项又会被静默丢弃（High）

`CSFs_choosing.py:648-660, 770-779`。

`_load_selection_idxs` 对任何可转换 dtype 直接执行 `astype(np.int64)`；因此浮点索引（例如 `1.9`）会静默截断为 `1`，布尔值也会变成 `0/1`。随后 `_valid_row_idxs` 仅保留合法范围内的值；只要还剩一个合法索引，`_select_unit_csfs` 就继续生成结果，不报告其余负数或超界值。这会让损坏或版本不匹配的索引映射产生一个“看似成功但集合已改变”的 `.c` 文件。

**建议**：加载后先验证数组是一维整数 dtype（拒绝 bool、float、字符串和对象）；对任一越界值整体失败，并在错误中报告最小值、最大值和 `row_count`。重复索引也应明确选择“拒绝”或“按首次出现去重”，而不是保留为隐式行为。

### 21. 两条描述符批处理路径返回不同 dtype（High）

`CSFs_compress_extract.py:230, 292-342, 345-377`。

`parse_csf_2_descriptor` 创建 `float32` 数组，因而 `batch_process_csfs_to_descriptors` 的 `np.stack` 返回 `float32`；parquet 路径却声明 `return_dtype=pl.List(pl.Float64)`，最终 `np.array(...)` 返回 `float64`。同一 CSF 数据仅因入口不同就产生不同的内存占用、序列化结果和潜在数值行为，违反稳定描述符应有的单一数据契约。

**建议**：为描述符定义一个模块级 dtype 常量并让两条路径强制使用同一 dtype；若以现有解析器为准，parquet 路径应使用 `pl.Float32` 且最终 `np.asarray(..., dtype=np.float32)`。增加同一 fixture 经两条路径处理后 `shape`、`dtype`、逐元素值完全一致的测试。

### 22. `J_to_doubleJ` 忽略分母，非法分数会得到错误但貌似合理的结果（Medium）

`CSFs_compress_extract.py:131-148`。

函数遇到分数时只返回分子：`numerator, _ = ...; return numerator`。这仅在输入已经保证为半整数形式 `n/2` 时成立；例如 `"3/4"` 会错误返回 `3`，而不是拒绝无法表示为整数 `2J` 的输入。当前调用链没有在进入该函数前验证分母。

**建议**：使用 `Fraction` 解析后计算 `2 * J`，仅当结果分母为 1 且值非负时返回整数，否则抛出包含原始输入的 `ValueError`。覆盖整数、半整数、不可约分数、零、负数和畸形字符串。

### 23. `coupling_level <= 0` 未校验并产生反直觉切片（Medium）

`CSFs_choosing.py:198-232`。

`coupling_level=0` 时 `tokens[-0:]` 等同于 `tokens[0:]`，结果是“保留全部层级”而非零层；负数则变成从正偏移位置开始切片。公开签名允许任意 `int`，docstring 也没有限定必须为正数。

**建议**：仅允许 `None` 或正整数，并在入口抛出明确 `ValueError`；为 `None/1/大于 token 数/0/负数` 建表测试，固化短记录的回退语义。

### 24. 并行提取可能重复转换同一源文件并造成线程放大（Medium）

`CSFs_choosing.py:810-841`。

多 unit 且 `workers > 1` 时，每个外层线程都会调用 `_select_unit_csfs`；如果两个 unit 指向同一个 `.c` 源文件（不同 idx/rmix 选择是合理场景），它们会同时向同一个 `<stem>.parquet` 和 header sidecar 发起转换，存在竞争写入。与此同时，每个外层任务又把完整的 `workers` 数传给 `rcsfs.convert_csfs`，最坏会形成约 `workers × workers` 的线程放大。

**建议**：提取前按解析后的源路径分组，每个唯一 `.c` 只转换一次；转换完成后再并行执行只读选择。并行预算只设在一层，或显式拆分 `unit_workers` 与 `convert_workers`。

### 25. `.c` 输出不是原子写入，失败可能留下截断的有效路径（Medium）

`CSFs_choosing.py:785-808`。

`_write_csfs_blocks_to_cfile` 直接以 `"w"` 打开最终路径；列结构检查发生在写入循环内，磁盘错误、某个后续 DataFrame 缺列或运行时异常都会在目标已经被截断后退出。函数也未阻止 `output_file` 与任一输入 `.c` 路径相同，误配置时会覆盖原始 CSF 数据。

**建议**：写入前完成所有 header、列、行类型和输入/输出路径冲突校验；在目标目录创建临时文件，完整写入并 flush/close 后用 `Path.replace` 原子替换。异常时删除临时文件并保留旧目标。

### 二次复核对原报告的两点修正

1. 原 Test Coverage 表把 `CSFs_sort_by_mix_coefficient` 标为“无测试”不准确。`tests/test_rmix_data_processor.py:343-383` 已有 2 个二维系数测试；真正缺失的是 High #1 指出的 **1D 单 ASF** 回归用例。
2. “`@dataclass(frozen=True)` 带来不可变值对象”需要限定为**浅层冻结**。`CsfsSelectionUnit.select_asfs`、`SelectedCsfsBlock.header_lines` 仍是可变 list，`SelectedCsfsBlock.csfs_df` 也不是不可变值；若该性质被作为并发安全保证，应改用 tuple/不可变表示或在边界做防御性复制。

---

## Strengths

- **`mypy --strict` 完全干净**：三个源文件类型标注覆盖全面，无一处报错。
- **文档完整**：绝大多数公开函数有完整的 Args/Returns/Raises docstring。
- **边界校验到位**：TOML 提取流水线（`_load_header_lines`、`_parse_select_asfs`、`_parse_cumulative_threshold`，以及各处 CSFs/mix-coefficient 长度一致性检查）体现了认真的加固工作。
- **安全细节**：`np.load(..., allow_pickle=False)` 正确避免了任意代码执行风险。
- **限制字段重绑定**：`CsfsSelectionUnit`、`SelectedCsfsBlock` 使用 `@dataclass(frozen=True)`；这是有益的浅层保护，但其 list/DataFrame 字段仍需按上方二次复核说明处理。
- **架构边界遵守良好**：`test_logical_packages.py` 验证的"核心层不依赖 `ml_module`"边界在本模块中被正确遵守。

---

## Test Coverage

### Covered

| 函数 | 测试文件 |
|------|----------|
| `single_asf_mix_square_above_threshold` | `tests/test_rmix_data_processor.py`（直接从 `CSFs_choosing` 子模块导入，非公开入口） |
| `batch_asfs_mix_square_above_threshold` | `tests/test_rmix_data_processor.py`（默认选择、显式 block/ASF 位置） |
| `batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum` | `tests/test_rmix_data_processor.py`（单 block、连续 `level_indices`） |
| `CSFs_sort_by_mix_coefficient` | `tests/test_rmix_data_processor.py`（2D 多 ASF；尚缺 1D） |

### Not Covered

| 区域 | 备注 |
|------|------|
| `parse_csf_2_descriptor` / `batch_process_csfs_to_descriptors` / `batch_process_csfs_parquet_to_descriptors` | 无任何测试，且含 High Priority #4 的静默失败路径 |
| `radom_choose_csfs` | 无测试，且含 High Priority #2 的两个缺陷 |
| `CSFs_sort_by_mix_coefficient` 的 1D 单 ASF 分支 | 已有 2D 测试，但未覆盖 High Priority #1 的崩溃路径 |
| `extract_from_config` 两套 TOML schema 分支 | 无测试 |
| `_rmix_selected_local_ci_idxs` 的 block-offset 累加 | 无测试，索引运算复杂，回归风险高 |
| `extract_csfs_units` 多 unit 合并路径 | 无测试，含 High Priority #5 |
| `subshell_charged_state` / `if_subshell_full_charged` / `CSF_subshell_split` | 无测试，且含 High Priority #3 |
| 非连续 `level_indices` 与 ASF 行号映射 | 无测试，含新增 High #19 |
| `.npy` 索引 dtype、越界值、重复值 | 无测试，含新增 High #20 |
| 同一 CSF 经内存/parquet 两条描述符路径的一致性 | 无测试，含新增 High #21 |
| `J_to_doubleJ` 非半整数分数；`coupling_level <= 0` | 无测试，含新增 Medium #22/#23 |
| 多 unit 共用同一 `.c` 源文件的并发路径 | 无测试，含新增 Medium #24 |

---

## 建议整改路线图（分阶段）

**阶段一 · 低风险修正缺陷**（不改变公开签名）
修复 #1、#3、#6、#22、#23；`radom_choose_csfs`（#2）改为 `None` 默认值 + 补 `else` 分支；删除 #7 的死判断；删除 #11 的两个死函数和 #12 的通配符导入。

**阶段二 · 健壮性**
优先修复数据契约 #19、#20、#21；`CSFs_compress_extract.py` 全面 `print → logger`（#8）；为 #4 定一个明确的失败策略（收集失败项、超阈值报错，而非静默 `continue`）；4 处 `zip()` 加 `strict=True`（#10）；#5 的合并前 header 校验；#9 的 `rcsfs` 可选依赖声明或友好报错；提取写入改为原子提交并防止覆盖输入（#25）。

**阶段三 · 结构**
拆分 `CSFs_choosing.py`（#15）；整合 #14 的重复定宽解析逻辑为命名常量/共享 helper；理清 #13 的公开边界；重构并行提取为“唯一源文件转换一次，再并行选择”（#24）。

**阶段四 · 测试与工具链（杠杆最大）**
为本模块补测试，覆盖上方 "Not Covered" 表中列出的路径；将 `ruff` 的 `select` 从 `["NPY201"]` 扩展为 `#17` 建议的规则集。

**长期 / 视情况处理**
命名风格统一（#16）——涉及公开 API 重命名，建议独立评估兼容性策略后再排期。
