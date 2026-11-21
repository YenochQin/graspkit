# CSF 计算描述符 V3 在 GraspKit 中的耦合分组计划

2026-11-21 · GraspKit 接入已实施 · 源码基线 `a10dfc9`。GraspKit 消费 rCSFs 返回的 Polars DataFrame，将 CSF 行与 mixing 文件对齐，执行确定性的分组、覆盖分析和选择。本仓库的交付重点是验证这个公开数据接口。Parquet 编解码由 rCSFs 的 `docs/CSF_DESCRIPTOR_V3_PLAN.md` 定义。[CSF 选择原语](csf-selection-primitives.md)

## 计算数据与分组接口

磁盘上的 V3 对 M 个有序 peel 子壳层存 `4M+2` 个 Int8。每个四元组是 `(qᵢ,vᵢ,2Jᵢ+1,2Kᵢ+1)`，末两列是 `total_two_j_plus_one`、`parity`。例如 J=½ 的奇宇称态以 `[2,-1]` 结尾。GraspKit 的分组入口消费以下派生列。

| 列 | GraspKit 使用的含义 |
| --- | --- |
| `idx` | 原始数据集中的 CSF 行号。使用 mixing 时，块内行号必须严格递增。输出选择结果时保留行号。 |
| `block_id` | 显式块身份；在块内建立局部索引，并按原次序匹配 mixing 的块 |
| `coupling_signature` | 非空的 `List(Int32)`。rCSFs 按已占据 peel 的规范打印约定生成分组值，尾项使用物理总 2J。纯闭核签名为 `[0]`。 |
| `parity` | V3 的 `+1` 和 `-1` 分别对应 mixing 数据的 `1` 和 `2`。提供此列时逐块校验。 |

例如总 J=½ 的磁盘编码是 2，签名尾项是 1。签名内部是用于分组的二倍角动量值。按规范打印边界取得中间值能够保留既有分组含义，包括合法的中间 0；逐轨道累计计算码直接由 rCSFs 的 V3 原始列提供。只取占据轨道再把每个累计码减一，可能得到不同的分组。例如 `1s¹ 2s² 3s¹` 取总 J=0 时，中间满壳继承 K=½，但在规范打印签名中该位置回落到 0。

## 实施与验收

1. rCSFs 负责解析原始版本、列顺序和计算码。GraspKit 使用派生列执行耦合分组和 CI² 筛选。[耦合分组实现](../src/graspkit/CSFs_processor/coupling.py)
2. 指定 `coupling_level` 时先去掉末项总 2J，再取所需的中间部分。全签名模式包含总 2J。加权和筛选前核对每块的 CSF 数、总 J、宇称和块内行序，覆盖相邻同 J 同宇称的独立块。
3. 在库内用公开接口的最小 DataFrame 测试。对于 peel 为 `1s,2s,3s,4s` 的 A₀、A₁（`1s² 2s¹ 3s¹ 4s¹`，3s 处分别 K=0、1），期望签名为 `[0,1,0,1]` 和 `[0,1,2,1]`。B=`1s² 2s² 3s¹ 4s⁰` 的签名为 `[0,0,1]`。三条记录均与总 J=½ 的 mixing 块对齐；深度 1 能分辨 A₀ 与 A₁。再检查纯闭核、非连续全局行号及对错误总 2J 的拒绝。[现有耦合测试](../tests/test_csfs_processor_coupling.py) · [mixing 分块测试](../tests/test_mix_coefficient_data_blocks.py)
4. 拉取请求、`main` 分支推送和发布构建均执行真实 V3 Parquet 用例。测试覆盖分组、原始行号、CI 权重选择，以及对宇称不符和块内行重排的拒绝。[V3 测试工作流](../.github/workflows/coupling-v3.yml)

在共享 Python 环境装有支持 V3 的 rCSFs 时，从 `graspkit/` 运行真实文件用例。测试将本库的 GRASP 文本夹具转换成 Parquet，核对版本为 3，再由 `read_csfs` 读取。V3 读取端拒绝 V2 文件。

```fish
GRASPKIT_TEST_RCSFS_V3=1 PYTHONPATH=src ../graspkit-tools/.venv/bin/python -m pytest 'tests/test_csfs_processor_coupling.py::test_v3_coupling_groups_and_selects_csfs_by_block[parquet]' --basetemp=../temp/tests/graspkit-coupling-v3
```
