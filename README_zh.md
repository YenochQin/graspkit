# graspkit

[English](README.md) | 简体中文

graspkit 是一个面向 GRASP 原子结构计算结果的 Python 工具包，主要用于把原始输出文件转换为更容易处理的 Python 对象、Polars DataFrame 和机器学习输入。当前代码库主要覆盖三类工作流：

- 读取和整理 GRASP 文本文件与二进制文件
- 将 CSF 转换为描述符矩阵，供筛选与建模使用
- 基于配置文件执行迭代式的 CSF 机器学习筛选流程

## 包含哪些内容

`src/graspkit/` 按职责拆分为几个核心模块，这些能力大多也会从 `graspkit.__init__` 直接导出：

- `data_IO/`：`.level`、`.lsj.lbl`、`.c`、`.ct`/`.t`、混合系数二进制文件的加载器，以及描述符/配置读写和 `MLCalConfig`
- `grasp_data_extractor/`：能级表格式化、迭代能级比较、LSJ 组成合并、跃迁后处理
- `CSFs_processor/`：CSF 解析、描述符生成、基于 CI 权重或简单采样的选择工具
- `ml_module/`：`ANNClassifier`、`ANNRegressor`、训练/评估辅助函数、迭代筛选工具
- `utils/`：通用数据结构、环境检测、绘图辅助、量子数和字符串工具

## 逻辑包入口

拆包计划的阶段 3 保持单仓库结构，但已经将 `graspkit` 明确收敛为 core 包入口：

- `graspkit_config`：轻量的 Pydantic 配置模型
- `graspkit`：数据加载、CSF 处理、结果提取和轻量工具
- `graspkit_ml`：机器学习训练、推理与迭代筛选辅助逻辑
- `graspkit_plot`：可选的绘图和 matplotlib 样式工具

新代码建议优先使用这些逻辑包入口：

```python
from graspkit_config import MLCalConfig
from graspkit.data_IO import load_config
from graspkit.grasp_data_extractor import format_energy_configurations
from graspkit_ml import train_model, evaluate_model
from graspkit_plot import configure_matplotlib_for_publication
```

旧的根包平面 ML 和 plotting 兼容导出不再属于推荐 API。新增代码应将 `graspkit` 视为 core 层，并从 `graspkit_ml` 导入机器学习能力。

## 安装

项目当前要求 Python `>=3.14`。

### 使用 UV

```bash
uv venv
source .venv/bin/activate

# 运行时依赖
uv sync --extra cpu

# 开发环境
uv sync --extra cpu --extra dev

# CUDA 环境
uv sync --extra gpu
uv sync --extra gpu --extra dev
```

### 使用 Pixi

```bash
pixi install
pixi shell

# 可选环境
pixi shell -e cpu
pixi shell -e gpu
pixi shell -e dev-gpu
```

### 验证导入

```bash
python -c "import graspkit; print(graspkit.__version__)"
```

更完整的安装说明见 [INSTALL.md](INSTALL.md)。

## 开发

常用本地开发命令：

```bash
ruff check .
ruff check . --fix
mypy src/
pytest tests/
python test/test.py
python -m build
```

## 当前代码库范围说明

- `graspkit` 是 core 包，`graspkit_ml` 和 `graspkit_plot` 是其上层扩展入口。
- 表格数据主要使用 Polars，描述符和模型数据主要使用 NumPy。
- PyTorch 通过 `cpu` 或 `gpu` extra 进行可选安装。
- SLURM 和调试环境检测位于 `graspkit.utils.environment_config`。

## 仓库内参考文件

- [INSTALL.md](INSTALL.md)
- [pyproject.toml](pyproject.toml)
- [src/graspkit](src/graspkit)
