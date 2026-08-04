# CSFs Processor 固化与跨仓库重构计划

**状态**：计划阶段，尚未修改业务代码
**日期**：2026-08-04
**依据**：[code-review-csfs-processor.md](code-review-csfs-processor.md) 及 CodeGraph 二次复核
**范围**：`graspkit` + `graspkit-tools`；`rCSFs` 作为既有描述符能力的权威依赖进行契约核对，不在本计划中新增 Python 描述符实现

## 1. 已确认的设计决策

1. 允许破坏性重构，不为旧函数保留兼容包装。
2. `graspkit-tools` 纳入同步改动范围；不能只修底层库而让 Tools 保留重复实现。
3. 描述符生成从 `graspkit` 移除，以 `rCSFs` 的 Rust/PyO3 实现为唯一生产实现。
4. TOML 配置解析、pipeline 路径推导、CLI 和多 unit 编排下沉到 `graspkit-tools`。
5. 索引和格式采用严格失败策略：不静默截断、过滤、补默认值或继续生成部分结果。
6. `.npy` 索引文件必须是一维整数数组；负数、越界和重复索引直接报错。
7. CSF 必须严格按三行记录解析；壳层格式和 J 格式非法直接报错。
8. 多 unit 合并时，所有 unit 的 5 行 header 必须完全一致；任何差异都直接退出并警告。
9. 输出路径与任一输入路径冲突时直接退出并警告。
10. 描述符/Parquet 行数与索引映射不一致时直接退出并警告。

## 2. 目标架构

```text
graspkit-tools
  解析 TOML / Pydantic 配置
  解析 pipeline 路径与 CLI
  去重输入源、安排并发预算
  调用 rCSFs 转换 .c -> Parquet / 生成描述符
  调用 graspkit 的纯库 API 完成严格索引选择
  调用稳定 writer 原子写出 .c

graspkit
  CSF 领域数据结构与格式校验
  CI 系数筛选、排序、coupling-J 分析
  严格索引选择
  多 block / header 一致性校验
  原子 .c writer（建议归入 data_IO）

rCSFs
  .c -> Parquet 高性能转换
  描述符生成与归一化
```

## 3. `graspkit` 改动计划

### 3.1 拆分模块职责

将当前 `src/graspkit/CSFs_processor/CSFs_choosing.py` 拆为清晰的库模块，建议结构如下：

```text
src/graspkit/CSFs_processor/
  selection.py        # CI 阈值、排序、随机/确定性选择
  coupling.py         # coupling-J 收集与 CI-square 汇总
  validation.py       # 索引、CSF 记录、header、shape/dtype 校验
  extraction.py       # 仅保留无配置依赖的 unit 选择原语（如仍有必要）
  __init__.py         # 只导出稳定、实际使用的库 API
```

若 `.c` writer 与现有 `data_IO` writer 语义重合，将 writer 迁移到 `src/graspkit/data_IO/`，避免 `CSFs_processor` 同时负责算法和持久化。

### 3.2 删除描述符重复实现

从 `CSFs_compress_extract.py` 和 `CSFs_processor/__init__.py` 删除：

- `parse_csf_2_descriptor`
- `batch_process_csfs_to_descriptors`
- `batch_process_csfs_parquet_to_descriptors`
- 仅服务于上述函数的重复定宽解析辅助逻辑

保留或迁移仍有明确消费者的基础解析能力；没有调用方的 `CSF_subshell_split`、`CSF_item_2_dict` 等旧辅助函数按 CodeGraph 结果删除。同步删除对应导出项和失效测试。

### 3.3 固化严格数据契约

- `_load_selection_idxs` 只接受一维整数 dtype；拒绝 bool、float、object、字符串和多维数组。
- 对所有索引做整体范围校验；任一负数或越界值都失败，不再使用 `_valid_row_idxs` 静默过滤。
- 明确重复索引策略；默认拒绝重复，除非调用方明确请求稳定去重。
- 明确 `level_id` 与 ASF 矩阵行号的概念；选择 API 只接收行号，或建立显式映射后再索引。
- 所有 CSF block 检查为严格三行记录，缺行、多行和行字段缺失都提供 block/row 上下文。
- 壳层字母表、满壳规则和 J 解析使用共享常量/解析器；`J_to_doubleJ` 必须验证 `2J` 为非负整数。
- `coupling_level` 只允许 `None` 或正整数。
- 描述符既然移除 Python 实现，`graspkit` 不再定义描述符 dtype 契约；Tools 对 `rCSFs` 输出 schema 做显式断言。

### 3.4 多 unit 合并和写入安全

- 多 unit 合并前逐行比较 5 行 header；任何差异立即失败。
- 校验所有 DataFrame 具有 `line1/line2/line3` 且行数与选择索引一致。
- 检查输出路径不得等于任何输入 `.c`、Parquet、header 或索引文件。
- 先写同目录临时文件，完整关闭后使用 `Path.replace` 原子替换目标；异常时清理临时文件。
- 在写文件前完成全部结构校验，避免目标文件已截断后才发现后续 block 错误。

### 3.5 删除重复/死 API

在 CodeGraph 和跨仓库搜索确认没有消费者后，删除：

- `_unit_from_pipeline_config`
- `_output_from_pipeline_config`
- 旧配置解析与 CLI 入口
- 重复的 coupling-J/选择包装函数
- 未使用通配符导入及其仅用于兼容的遗留导出

破坏性重构允许直接移除，不保留 deprecated wrapper。最终 `__all__` 只列出稳定 API。

## 4. `graspkit-tools` 改动计划

### 4.1 接管配置与 CLI

把以下逻辑从 `graspkit` 移入 Tools 现有配置/编排层：

- pipeline TOML 解析
- `[[csfs_units]]` schema 解析与兼容字段归一化
- 相对路径解析
- loop/config 输出路径推导
- `build_arg_parser` / `run_from_cli`
- 用户级错误信息、日志和退出码

Tools 最终只向 `graspkit` 传入已经解析、类型明确的参数，不再让库读取 Tools TOML。

### 4.2 统一索引和选择流程

- 由 Tools 负责发现、验证和去重输入源。
- 同一个 `.c` 源文件只转换一次 Parquet，之后只读复用。
- 外层 unit 并发与 `rcsfs` 内部转换并发只保留一层预算，避免 `workers × workers` 放大。
- 所有 `.npy` 索引通过统一严格校验器进入流程。
- header 不一致、shape 不一致、路径冲突和非法格式均转为明确的失败状态并退出。

### 4.3 采用 `rCSFs` 描述符实现

- 删除 Tools 对 `graspkit` Python 描述符 API 的调用。
- 直接调用 `rcsfs` 的 Parquet 转换、描述符生成和归一化入口。
- 对输出列名、行数、descriptor size、dtype/schema 做显式断言。
- 将 Rust 测试 fixture 与 Tools 集成测试 fixture 对齐，确保跨语言契约稳定。

### 4.4 消除 Tools 重复实现

重点迁移/删除 `sampling_csfs.py` 中与底层库重复的：

- `.npy` 索引加载与整数转换
- 越界过滤
- DataFrame 按索引选择
- `.c` 写入

Tools 保留业务策略（采样比例、ML 选择优先级、loop 状态），调用 `graspkit` 的严格领域 API 完成数据操作。

## 5. 测试计划

### 5.1 `graspkit` 单元测试

新增/迁移测试覆盖：

- 一维整数 `.npy` 索引通过；多维、float、bool、object、字符串拒绝。
- 负数、越界、重复索引拒绝。
- 非连续 `level_indices` 不会被误当作矩阵行号。
- CSF 缺行、多行、缺列、非法壳层、非法 J 拒绝。
- `J_to_doubleJ` 的整数、半整数、非法分数和负数。
- `coupling_level` 的合法边界和非正值。
- 多 unit header 完全一致时合并成功，任一行不同则失败。
- 输入/输出路径冲突失败。
- writer 中途失败不会破坏已有目标文件。
- 所有选择/汇总函数的结果顺序、重复策略和空输入行为。

### 5.2 `graspkit-tools` 集成测试

- 两种配置 schema 归一化到同一内部模型。
- CLI 到严格库 API 的完整调用路径。
- 多 unit 共享 `.c` 时只转换一次。
- 多 unit header 不一致时退出并输出可定位警告。
- ML 索引和预采样索引的 dtype/范围错误均 fail-fast。
- `rCSFs` 输出 schema、行数和描述符数量一致。
- 失败场景不会留下截断 `.c` 或错误的索引文件。

### 5.3 验证命令

```text
cd graspkit-tools
uv run pytest ../graspkit/tests
uv run pytest
uv run ruff check ../graspkit/src/graspkit
uv run ruff check ml_CSFs_selection_scripts pyscript
uv run mypy ../graspkit/src
uv run mypy ml_CSFs_selection_scripts pyscript

cd ../rCSFs
uv run cargo test
```

## 6. 实施顺序与提交边界

1. 先在 `graspkit` 建立严格校验、header 比较、原子 writer 和新纯库 API，并为其补测试。
2. 在 `graspkit-tools` 迁移配置/CLI 和调用方，切换到新 API；加入跨仓库集成测试。
3. 删除 `graspkit` 描述符实现、TOML/CLI 实现和已确认无调用方的重复 API。
4. 删除 Tools 中被替代的索引过滤、DataFrame 选择和 writer 重复逻辑。
5. 用 `rCSFs` 测试确认描述符契约；必要时只调整 rCSFs 的公开 schema 或文档，不重新引入 Python 实现。
6. 各子仓库分别提交：先提交 `graspkit` API/测试，再提交 `graspkit-tools` 迁移/测试；最后回到 workspace 更新两个 submodule gitlink。

## 7. 验收标准

- `graspkit` 不再包含 Python 描述符生成、Tools TOML 解析或 CLI 工作流。
- `graspkit-tools` 不再重复实现索引静默过滤、CSF writer 和描述符生成。
- 所有索引、CSF、header、J、shape 和路径错误均 fail-fast，错误可定位。
- 多 unit header 不完全一致时不会生成输出文件。
- 同一输入源在并发流程中只转换一次。
- 输出采用原子提交，失败不会破坏既有结果。
- `rCSFs` 是唯一描述符生产实现，Tools 对其输出有明确契约测试。
- 两仓库测试、类型检查和 lint 通过；无未确认的旧 API 引用。

## 8. 待最终确认项

按当前破坏性重构方向，默认删除 CodeGraph 确认无调用方的旧函数和导出项；若其中某个函数仍需保留，应在实施前明确列入“保留 API”清单。
