# graspkit Package Splitting Implementation Spec

## 文档信息

- 状态：Draft
- 目标版本：`3.2.x` 起始，跨两个小版本完成迁移
- 负责人：Yenoch(Yi) Qin
- 适用范围：当前单仓库 `graspkit`

## 1. 目标

本实施规范定义 `graspkit` 从“单一大包 + 平面导出”演进为“分层包边界 + 可选安装能力”的落地路径。

本次改造的直接目标不是立即拆成多仓库，而是：

1. 降低 `import graspkit` 的导入副作用和启动成本
2. 建立稳定的模块边界：`config`、`core`、`ml`
3. 为后续独立分发 `graspkit-core` / `graspkit-ml` 做准备
4. 保持现有调用侧在过渡期内可运行

## 2. 非目标

以下事项不属于本轮实施范围：

1. 本轮不拆多仓库
2. 本轮不重写 ML 算法逻辑
3. 本轮不大规模调整文件内部实现，只调整导入边界、导出边界和分发边界
4. 本轮不承诺一次性移除所有旧 API

## 3. 当前问题陈述

当前代码已在源码目录层面形成领域分层：

- `src/graspkit/data_IO`
- `src/graspkit/grasp_data_extractor`
- `src/graspkit/CSFs_processor`
- `src/graspkit/ml_module`
- `src/graspkit/utils`
- `src/graspkit_config`

但包边界仍然不清晰，主要问题如下：

1. [src/graspkit/__init__.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/__init__.py) 进行了大规模平面重导出
2. [src/graspkit/utils/__init__.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/utils/__init__.py) 默认导入了绘图接口
3. [src/graspkit/utils/plot_functions.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/utils/plot_functions.py) 和 [src/graspkit/utils/fig_settings.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/utils/fig_settings.py) 直接依赖 `matplotlib`
4. `ml_module` 依赖 `data_IO` 和 `utils`，但调用侧常通过根包访问，导致轻量场景也会触发重模块导入
5. 目前 `pyproject.toml` 仍把 `matplotlib`、`scikit-learn` 等能力放在统一安装集合中

## 4. 现状约束

### 4.1 已确认的依赖方向

当前更合理的目标依赖关系应为：

- `config` 不依赖 `core`
- `core` 不依赖 `ml`
- `ml` 可以依赖 `core` 和 `config`

注意：实现语义应为 `ml -> core -> config?` 并不准确。这里的设计约束是：

- `config` 只承载轻量配置模型
- `core` 可以使用 `config` 中的配置类型
- `ml` 可以使用 `core` 和 `config`

因此，推荐最终方向为：

- `config`
- `core -> config`
- `ml -> core + config`

### 4.2 已验证可复用的模式

当前配置模型已直接位于 [src/graspkit_config/ml_config_models.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit_config/ml_config_models.py)，`graspkit.data_IO` 只保留对 `MLCalConfig` / `CalPath` 的正式导出，不再经过兼容转发层。

这证明下列模式可用：

1. 抽离轻量模块到独立包
2. 保留兼容转发层
3. 通过逐步迁移替代一次性破坏式变更

## 5. 目标架构

### 5.1 逻辑包划分

目标逻辑结构如下：

#### `graspkit-config`

职责：

- Pydantic 配置模型
- 轻量路径模型
- 与训练、绘图、文件解析无关的纯配置定义

当前主要文件：

- [src/graspkit_config/ml_config_models.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit_config/ml_config_models.py)

#### `graspkit-core`

职责：

- 文件解析
- 计算结果读取
- CSF 数据提取与转换
- 非绘图型基础工具

建议包含：

- `src/graspkit/data_IO`
- `src/graspkit/grasp_data_extractor`
- `src/graspkit/CSFs_processor`
- `src/graspkit/utils/data_modules.py`
- `src/graspkit/utils/tool_function.py`
- `src/graspkit/utils/environment_config.py`
- `src/graspkit/utils/quadrupole_deformation.py`

#### `graspkit-ml`

职责：

- CSF 训练与预测
- 分类与回归模型
- 训练初始化
- 结果分析

建议包含：

- `src/graspkit/ml_module`

依赖：

- `graspkit-core`
- `graspkit-config`
- `torch`
- `scikit-learn`

#### 可选后续层：`graspkit-plot`

职责：

- 绘图工具
- `matplotlib` 和相关可视化配置

建议包含：

- `src/graspkit/utils/plot_functions.py`
- `src/graspkit/utils/fig_settings.py`

本轮不要求独立发布 `graspkit-plot`，但要求先从默认入口中解耦。

## 6. 实施策略

采用“四阶段渐进迁移”：

1. 阶段 0：收紧顶层导出，降低默认导入副作用
2. 阶段 1：调用侧改为显式子模块导入
3. 阶段 2：仓库内形成稳定逻辑拆包
4. 阶段 3：分发层拆分

仅当阶段 3 稳定后，再评估是否进入多仓库维护。

## 7. 交付标准

本 spec 的总体完成标准如下：

1. `import graspkit` 不再自动导入 `ml_module`
2. `import graspkit` 不再自动导入绘图模块
3. 轻量调用可通过显式子模块导入运行，而不触发 `torch` 或 `matplotlib`
4. 旧导入路径在兼容期内仍可工作，并给出明确 deprecation warning
5. 打包层能表达 `core` 与 `ml` 的分发边界

## 8. 阶段实施明细

### 阶段 0：顶层 API 收口

#### 目标

在不改变主功能行为的前提下，先降低根包导入副作用。

#### 修改范围

- [src/graspkit/__init__.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/__init__.py)
- [src/graspkit/utils/__init__.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/utils/__init__.py)

#### 必做任务

1. 缩减 `graspkit.__all__`
2. 根包只保留以下类别的稳定轻量导出：
   - `__version__`
   - `__author__`
   - 少量明确声明为稳定的轻量类型或函数
3. 从根包默认导出中移除 `ml_module` 的符号
4. 从 `graspkit.utils` 默认导出中移除绘图函数
5. 如需兼容，使用惰性导入或 `__getattr__` 转发，而不是 import-time 全量加载

#### 建议保留的根包最小集合

建议仅保留：

- `__author__`
- `__version__`

可选保留：

- `MLCalConfig`
- `CalPath`
- `load_config`

如果保留可选项，必须保证不会触发 `ml_module` 和绘图模块加载。

#### 验收标准

1. `python -X importtime -c 'import graspkit'` 不再显示 `graspkit.ml_module` 自动导入
2. `python -X importtime -c 'import graspkit'` 不再显示 `graspkit.utils.plot_functions` 自动导入
3. 现有轻量使用场景可通过显式子模块路径运行

#### 回退策略

如果根包缩减导致明显兼容中断，则先恢复兼容转发，但必须保留懒加载实现，不允许回退到 import-time 全量加载。

### 阶段 1：调用侧迁移到显式子模块导入

#### 目标

消除调用侧对 `import graspkit as gk` 的平面 API 依赖。

#### 优先迁移对象

1. `graspkit-tools/ml_CSFs_selection_scripts`
2. `pyscript/`
3. 文档中的示例代码

#### 导入规则

禁止新增：

```python
import graspkit as gk
from graspkit import *
```

要求新增代码改为显式导入，例如：

```python
from graspkit.data_IO import MLCalConfig, load_config, update_config
from graspkit.ml_module import train_model, evaluate_model
from graspkit.utils.data_modules import MLDataCounts
```

#### 必做任务

1. 搜索并列出所有根包平面导入调用点
2. 为每个调用点映射显式子模块路径
3. 更新示例、教程和安装文档
4. 在旧导入路径触发 warning，提示迁移目标模块

#### 验收标准

1. 仓库内新增代码不再使用根包平面导入
2. 文档示例默认使用显式子模块导入
3. 至少一轮真实调用侧迁移完成并验证可运行

### 阶段 2：仓库内逻辑拆包

#### 目标

在单仓库中形成稳定的逻辑边界，为独立分发做准备。

#### 结构策略

允许以下两种实现路径，任选其一：

1. 直接新增包目录：
   - `src/graspkit_core/...`
   - `src/graspkit_ml/...`
   - `src/graspkit_config/...`
2. 保留当前目录结构，但在打包和导入层定义清晰边界

本项目建议优先采用路径 2，原因如下：

1. 当前目录已经稳定
2. 本轮核心目标是导入边界和分发边界，而不是大规模搬迁文件
3. 可减少一次性重命名带来的兼容风险

#### 必做任务

1. 为每个现有模块标注归属层：`config`、`core`、`ml`、`plot`
2. 清理 `utils` 的混合边界
3. 将绘图能力定义为可选层，不再视为 `core` 默认组成
4. 确保 `core` 代码不反向依赖 `ml`
5. 保留兼容导入转发层

#### 推荐的模块归属清单

`config`：

- [src/graspkit_config/ml_config_models.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit_config/ml_config_models.py)

`core`：

- [src/graspkit/data_IO/__init__.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/data_IO/__init__.py)
- [src/graspkit/data_IO/file_locator.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/data_IO/file_locator.py)
- [src/graspkit/data_IO/processing_data_loader.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/data_IO/processing_data_loader.py)
- [src/graspkit/data_IO/produced_data_writor.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/data_IO/produced_data_writor.py)
- `src/graspkit/data_IO/loaders/*`
- `src/graspkit/grasp_data_extractor/*`
- `src/graspkit/CSFs_processor/*`
- [src/graspkit/utils/data_modules.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/utils/data_modules.py)
- [src/graspkit/utils/tool_function.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/utils/tool_function.py)
- [src/graspkit/utils/environment_config.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/utils/environment_config.py)
- [src/graspkit/utils/quadrupole_deformation.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/utils/quadrupole_deformation.py)

`ml`：

- `src/graspkit/ml_module/*`

`plot`：

- [src/graspkit/utils/plot_functions.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/utils/plot_functions.py)
- [src/graspkit/utils/fig_settings.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/utils/fig_settings.py)

#### 验收标准

1. 每个模块都有明确归属
2. `core` 不依赖 `ml`
3. `plot` 不再通过默认入口隐式加载
4. 兼容转发层有测试覆盖

### 阶段 3：分发层拆分

#### 目标

把逻辑边界升级为可独立安装的分发包边界。

#### 修改范围

- [pyproject.toml](/Users/yiqin/Documents/PythonProjects/graspkit/pyproject.toml)
- 打包配置
- 发布脚本
- 安装文档

#### 建议发布形态

1. `graspkit-core`
2. `graspkit-ml`
3. `grasp-kit` 作为聚合包或兼容元包

#### 必做任务

1. 定义 `core` 的最小依赖集合
2. 将 ML 依赖从默认依赖集合中剥离到扩展包或 extra
3. 明确 `matplotlib` 是否留在 `core`，若不留则移入可选层
4. 更新安装说明：
   - 仅数据处理安装方式
   - 含 ML 安装方式
   - GPU 安装方式

#### 验收标准

1. 用户可只安装核心数据处理能力
2. 需要训练时再安装 ML 扩展
3. 聚合包能兼容现有安装习惯

## 9. 兼容策略

### 9.1 兼容窗口

兼容窗口建议覆盖两个小版本：

1. 第一个版本：默认兼容 + warning
2. 第二个版本：保留兼容层，但 warning 升级为更明确迁移提示
3. 第三个版本：再评估是否删除旧平面导出

### 9.2 warning 规则

旧路径访问以下内容时，允许兼容但必须告警：

1. 从 `graspkit` 根包访问 `ml_module` 相关符号
2. 从 `graspkit.utils` 访问绘图相关符号

warning 至少需要包含：

1. 旧路径
2. 新路径
3. 计划移除的版本窗口

### 9.3 文档策略

所有新文档、示例、README 片段、安装说明必须使用新路径，不再展示根包平面导入作为推荐用法。

## 10. 验证方案

### 10.1 导入行为验证

必须新增或补充以下验证：

1. `import graspkit` 不加载 `graspkit.ml_module`
2. `import graspkit` 不加载 `graspkit.utils.plot_functions`
3. `import graspkit.data_IO` 不因根包副作用加载 `ml_module`
4. `import graspkit_config.ml_config_models` 保持轻量

### 10.2 兼容性验证

必须验证：

1. 旧导入路径仍能运行
2. warning 正常发出
3. 新显式导入路径正常运行

### 10.3 依赖边界验证

建议增加静态检查脚本，至少覆盖：

1. `core` 中禁止导入 `graspkit.ml_module`
2. `plot` 不得成为 `core` 默认导出的一部分

### 10.4 性能基线

需要记录至少三组导入基线：

1. `import graspkit`
2. `import graspkit.data_IO`
3. `import graspkit_config.ml_config_models`

记录项：

1. importtime 总耗时
2. 是否触发 `torch`
3. 是否触发 `matplotlib`

## 11. 测试任务清单

需要补充的测试至少包括：

1. 根包导入测试
2. 兼容 warning 测试
3. 显式子模块导入测试
4. `graspkit.utils` 不自动导入 plotting 的测试
5. `graspkit.data_IO` 对 `MLCalConfig` / `CalPath` 的正式导出测试

建议测试文件：

- `tests/test_import_boundaries.py`
- `tests/test_deprecated_exports.py`
- `tests/test_config_compatibility.py`

## 12. 风险与应对

### 风险 1：旧代码高度依赖平面 API

影响：

- 改动面大
- 容易漏改

应对：

1. 保留兼容转发
2. 为 warning 提供精确替换路径
3. 优先迁移内部调用侧和文档

### 风险 2：`utils` 边界不纯

影响：

- `core` 被绘图依赖污染
- 导入成本继续偏高

应对：

1. 拆分 `utils` 中的轻量能力与绘图能力
2. 禁止将绘图函数重新纳入默认导出

### 风险 3：发布层拆分与安装说明不同步

影响：

- 用户安装体验混乱
- issue 成本上升

应对：

1. 分发改造与文档同步发布
2. 保留聚合包或清晰的迁移说明

## 13. 里程碑

### Milestone A

完成条件：

1. 阶段 0 完成
2. 导入副作用明显下降
3. 新增边界测试通过

### Milestone B

完成条件：

1. 阶段 1 完成
2. 主要内部调用侧完成迁移
3. 文档示例全部切换到显式导入

### Milestone C

完成条件：

1. 阶段 2 完成
2. 模块归属稳定
3. 兼容层策略经过至少一个版本验证

### Milestone D

完成条件：

1. 阶段 3 完成
2. 分发层支持按需安装
3. 聚合安装路径可用

## 14. 最小落地版本

如果只做一轮最小、稳妥、收益最高的改造，本轮必须至少完成：

1. 收缩 [src/graspkit/__init__.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/__init__.py) 默认导出
2. 收缩 [src/graspkit/utils/__init__.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/utils/__init__.py) 默认导出
3. 保持 [src/graspkit_config/ml_config_models.py](/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit_config/ml_config_models.py) 继续作为轻量配置层
4. 增加导入边界测试和 warning 测试
5. 将主要调用侧从 `import graspkit as gk` 迁移到显式子模块导入

## 15. 结论

本项目应采用“单仓库、分层包边界、渐进迁移”的实施路径，而不是立即拆成多仓库。

本 spec 的执行顺序必须是：

1. 先收口顶层导出
2. 再迁移调用侧
3. 再稳定逻辑拆包
4. 最后才做分发拆分

只有按这个顺序推进，`graspkit` 才能同时得到以下收益：

1. 更低的默认导入成本
2. 更清晰的模块边界
3. 更细粒度的安装方式
4. 更低的迁移风险
