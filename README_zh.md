# graspkit

[English](README.md) | 简体中文

`graspkit` 是用于读取、二次处理和绘制 GRASP 原子结构计算结果的稳定基础库。机器学习模型、训练配置和迭代筛选流程已归入同级的 `graspkit-tools` 仓库。

## 包结构

- `graspkit.data_IO`：GRASP 文本/二进制文件加载器和稳定的 CSF 写出接口。
- `graspkit.grasp_data_extractor`：ASF、能级、组分、跃迁和混合系数二次处理。
- `graspkit.CSFs_processor`：CSF 解析、描述符和确定性筛选函数。
- `graspkit.utils`：通用 GRASP 数据模型、环境检测和量子数工具。
- `graspkit_plot`：绘图和出版级样式工具。

本仓库不再包含 ML 模型、ML 配置或迭代训练状态；这些代码位于：

```text
graspkit-tools/ml_CSFs_selection_scripts/ml_csf_choosing/
```

## 安装

使用 Python 3.14 或更新版本，从 [PyPI 安装 graspkit-atomic](https://pypi.org/project/graspkit-atomic/)。
分发包名为 `graspkit-atomic`，Python 导入名为 `graspkit` 和 `graspkit_plot`。

### 使用 pip

> 新 PyPI 名称尚待首次发布，以下安装命令在 `graspkit-atomic` 发布后生效。

```bash
python -m pip install graspkit-atomic
```

需要固定为当前发布版本时：

```bash
python -m pip install "graspkit-atomic==3.4.1"
```

### 在 pyproject.toml 中声明依赖

将依赖加入你自己项目已有的 `[project].dependencies`。最小示例如下：

```toml
[project]
name = "atomic-analysis"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
    "graspkit-atomic==3.4.1",
]
```

在你的使用项目目录中安装项目及其依赖：

```bash
python -m pip install .
```

两种安装方式使用相同的 Python API。如果项目同时使用 rCSFs，可在上述依赖列表中
另加 `"rcsfs==1.3.1"`。绘图 API 已包含在本包中；ML 训练和工作流编排由
`graspkit-tools` 提供。本地 wheel 安装和贡献者开发环境见 [INSTALL.md](INSTALL.md)。

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

## 开发检查

在完整工作区中，先激活共享环境：

```bash
source ../graspkit-tools/.venv/bin/activate
pytest tests/
ruff check .
basedpyright src/
python -m build
```

## 发布准备

构建、验证和手动发布工作流见[发布指南](docs/releasing.md)。

## 许可证

GNU GPL 第 3 版（`GPL-3.0-only`），见 [LICENSE](LICENSE)。
