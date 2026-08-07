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

需要 Python 3.14 或更新版本。

```bash
uv sync --extra dev
```

在完整 workspace 中开发时，推荐使用 `graspkit-tools` 所属的 uv 环境；该环境会以 editable 方式安装本仓库。

## 导入示例

```python
from graspkit.data_IO import EnergyFileLoader, MixCoefLoader
from graspkit.grasp_data_extractor import format_energy_configurations
from graspkit.CSFs_processor import batch_asfs_mix_square_above_threshold
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

```bash
pytest tests/
ruff check .
mypy src/
python -m build
```
