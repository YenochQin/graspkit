# graspkit

[English](README.md) | 简体中文

`graspkit` 是用于读取、二次处理和绘制 GRASP 原子结构计算结果的稳定基础库。

## 包结构

- `graspkit.data_IO`：GRASP 文本/二进制文件加载器和稳定的 CSF 写出接口。
- `graspkit.grasp_data_extractor`：ASF、能级、组分、跃迁和混合系数二次处理。
- `graspkit.CSFs_processor`：CSF 解析、描述符和确定性筛选函数。
- `graspkit.utils`：通用 GRASP 数据模型、环境检测和量子数工具。
- `graspkit_plot`：绘图和出版级样式工具。

## 安装

使用 Python 3.14 或更新版本，从 [PyPI 安装 graspkit-atomic](https://pypi.org/project/graspkit-atomic/)。
分发包名为 `graspkit-atomic`，Python 导入名为 `graspkit` 和 `graspkit_plot`。

### 使用 uv 管理项目

可以用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 管理你自己的分析项目环境和依赖。新建项目时：

```bash
uv init --python 3.14 atomic-analysis
cd atomic-analysis
uv add graspkit-atomic
uv run python -c "import graspkit; print(graspkit.__version__)"
```

已有 uv 项目使用 Python 3.14 或更新版本时，直接在其目录运行 `uv add graspkit-atomic`。
该命令会将依赖写入 `pyproject.toml`，更新 `uv.lock`，并安装到项目的 `.venv` 中。
运行自己的分析脚本时使用 `uv run python analysis.py`，无需手动激活环境。
需要固定版本时，使用 `uv add "graspkit-atomic==3.4.5"`。

也可以手动将依赖加入已有的 `pyproject.toml`：

```toml
[project]
name = "atomic-analysis"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
    "graspkit-atomic==3.4.5",
]
```

编辑后，在该项目目录运行 `uv sync`，完成依赖解析和环境同步。
将 `pyproject.toml` 和 `uv.lock` 随分析代码一起保存到版本控制中，便于记录依赖版本。

### 使用 pip

安装到已有的 Python 环境：

```bash
python -m pip install graspkit-atomic
```

两种方式提供相同的 Python API，均包含绘图功能。如果项目同时使用 rCSFs，
可另行执行 `uv add rcsfs`。本地 wheel 安装见 [INSTALL.md](INSTALL.md)。

## 导入示例

```python
from graspkit.data_IO import EnergyFileLoader, MixCoefLoader
from graspkit.grasp_data_extractor import format_energy_configurations
from graspkit.CSFs_processor import select_csf_indices_by_ci_squared_cutoff
from graspkit_plot import configure_matplotlib_for_publication
```

## Windows 终端编码

在使用旧代码页的 Windows 终端中，`MixCoefLoader` 等加载器通过 Rich
输出 `cm⁻¹` 等科学单位时，可能触发 `UnicodeEncodeError`。请使用 UTF-8
模式启动 Python：

```powershell
python -X utf8 your_script.py
```

也可以使用以下命令快速检查混合系数文件的加载：

```powershell
python -X utf8 -c "from graspkit.data_IO import MixCoefLoader; MixCoefLoader(r'path\to\file.m').load()"
```

`-X utf8` 只会调整 Python 的文本与终端输出编码，不会改变 GRASP 文件的解析方式。

## 发布准备

构建、验证和手动发布工作流见[发布指南](docs/releasing.md)。

## 许可证

GNU GPL 第 3 版（`GPL-3.0-only`），见 [LICENSE](LICENSE)。
