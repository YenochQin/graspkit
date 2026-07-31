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

## 开发检查

```bash
pytest tests/
ruff check .
mypy src/
python -m build
```
