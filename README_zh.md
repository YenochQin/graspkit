# GraspKit

**版本**: 2.9.1 | **Python**: 3.13

[English](README.md) | 简体中文

一个用于收集和处理 GRASP（广义相对论原子结构包）计算结果的 Python 工具包。该工具增强了 GRASP 内置的数据处理能力，提供更灵活的基于 Python 的数据处理、机器学习优化和自动化工作流管理。

## 概述

GraspKit 为使用 GRASP2018 计算的原子物理研究人员提供了全面的工具包。它实现了原子结构数据的自动提取、处理和分析，专注于基于机器学习的组态波函数（CSF）选择优化，用于大规模 MCDHF 计算。

## 主要特性

- **机器学习驱动的 CSF 选择流程** - 使用神经网络智能选择量子力学计算的 CSF，在保持精度的同时降低计算成本
- **GRASP 集成** - 通过 shell 脚本实现 GRASP2018 计算的自动化工作流管理
- **多格式数据处理** - 支持二进制、Parquet、HDF5 和文本格式的原子物理数据分析综合工具
- **HPC 就绪** - 原生 SLURM 环境检测，自动隐藏进度条并支持环境感知日志记录
- **GPU 加速** - 支持 PyTorch 模型的 CUDA 加速，自动设备选择
- **C++ 集成** - 通过 C++ 扩展实现快速描述符生成

## 安装

GraspKit 支持多种安装方法，选择最适合您工作流程的一种。

### 方法 1: UV（推荐）

UV 是一个超快速的 Python 包管理器，依赖解析速度比 pip 快 10-100 倍。

```bash
# 安装 UV（如果尚未安装）
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库
git clone https://github.com/YenochQin/graspkit-tools.git
cd graspkit-tools

# 创建并激活虚拟环境
uv venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 安装依赖 - 选择 CPU 或 GPU 版本
# CPU 版本（推荐，兼容性更好）
uv sync --extra cpu

# GPU 版本（需要 NVIDIA CUDA）
uv sync --extra gpu

# 开发环境（包含测试工具）
uv sync --extra dev --extra cpu    # CPU
uv sync --extra dev --extra gpu    # GPU
```

### 方法 2: Pixi（备选）

Pixi 是一个使用 conda-forge 依赖的跨平台包管理器。

```bash
# 安装 Pixi（如果尚未安装）
# Windows (PowerShell)
powershell -c "irm https://pixi.sh/install.ps1 | iex"
# macOS/Linux
curl -fsSL https://pixi.sh/install.sh | bash

# 克隆仓库
git clone https://github.com/YenochQin/graspkit-tools.git
cd graspkit-tools

# 安装依赖并激活环境
pixi install
pixi shell

# Pixi 默认配置 dev-gpu 环境
# 切换环境：
pixi shell -e cpu     # 仅 CPU
pixi shell -e gpu     # 启用 GPU
pixi shell -e dev-gpu # 开发环境 + GPU
```

**Pixi 特性**：
- 基于 conda-forge 的包管理
- 跨平台支持（Linux、macOS、Windows）
- 基于锁文件的可重现性（pixi.lock）
- 自动环境切换
- 与现有 conda 工作流集成

### 系统要求

- **Python**: 3.13（仅支持此版本）
- **操作系统**: Linux、Windows 10+、macOS 10.15+
- **内存**: 4GB RAM（推荐 8GB+ 用于大型数据集）
- **GPU（可选）**: 支持 CUDA 的 NVIDIA GPU，用于加速机器学习训练

### 验证安装

```bash
# 验证安装
python -c "import graspkit; print('Package OK')"

# 检查版本
python -c "import graspkit; print(graspkit.__version__)"
```

详细的安装说明和故障排除，请参阅 [INSTALL.md](INSTALL.md)。

## 包架构

```
graspkit/
+-- CSFs_processor/          # CSF 选择和处理算法
|   +-- CSFs_choosing.py     # 随机/阈值 CSF 选择
|   +-- CSFs_compress_extract.py  # 从 CSF 生成描述符
|
+-- data_IO/                 # 数据输入/输出处理
|   +-- GraspFileLoad        # GRASP 输出文件的主加载器
|   +-- EnergyFile2csv       # 能量文件转换
|   +-- data_writer.py       # 保存处理后的数据（pickle/parquet/HDF5）
|   +-- data_loader.py       # 加载处理后的数据
|
+-- grasp_data_extractor/    # 物理量提取
|   +-- ASF_data_collection.py      # 能级、ASF 组成
|   +-- transition_data_collection.py  # 跃迁率数据
|   +-- transition_data_analyzer.py    # 跃迁分析
|
+-- ml_module/               # 机器学习流程
|   +-- neural_network.py    # ANN 和 TensorNet 架构
|   +-- ml_initializer.py    # 训练数据设置和验证
|   +-- ml_trainer.py        # 模型训练和评估
|   +-- ml_results_analyzer.py  # 结果分析和 CSF 选择
|
+-- utils/                   # 工具函数
    +-- data_modules.py      # 数据结构（MixCoefficientData、CSFs）
    +-- environment_config.py  # HPC/SLURM 检测
    +-- progress_manager.py  # 环境感知进度条
    +-- plot_functions.py    # 可视化工具
```

## 快速开始

```python
import graspkit as gk

# 1. 加载 GRASP 计算结果
loader = gk.GraspFileLoad("path/to/grasp/output")
energy_data = gk.mcdhf_energy_data_collection(loader)

# 2. 使用 ML 驱动的选择处理 CSF
# 从 CSF 配置生成描述符
csf_descriptors = gk.batch_process_csfs_to_descriptors(csfs_data)

# 3. 训练 ML 模型用于 CSF 预测
model = gk.ANNClassifier(
    input_dim=len(csf_descriptors[0]),
    hidden_layers=[256, 128, 64],
    num_classes=2
)
gk.train_model(model, training_data, labels)

# 4. 为下一次迭代选择 CSF
selected_csfs = gk.select_csfs_for_coverage(
    model,
    candidate_csfs,
    coverage_threshold=0.95
)

# 5. 检查收敛性
converged = gk.evaluate_calculation_convergence(
    current_energy,
    previous_energy
)
```

## 使用示例

`tests/` 目录包含示例脚本：

```bash
# ML 分类器示例
python tests/ANN.py

# 波函数可视化
python tests/rwfn_plotter.py

# 可视化示例
python tests/Nightingale_rose.py

# 运行测试覆盖
python tests/test_coverage_simple.py
python tests/test_coverage_function.py
```

## 开发

### 代码检查和格式化

```bash
# 运行 Ruff 检查（NumPy 2.0 兼容）
ruff check .

# 自动修复检查问题
ruff check . --fix

# 类型检查（可选）
mypy src/
```

### 构建包

```bash
# 清理构建
python build_package.py --clean

# 开发构建（包含所有依赖）
python build_package.py --dev --clean
```

### 测试

```bash
# 运行测试
pytest tests/

# 运行特定测试文件
python tests/test_coverage_simple.py
```

更详细的开发说明，请参阅 [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)。

## 主要工作流程

1. **数据加载** - 加载 GRASP 输出文件（能级、混合系数、CSF）
2. **描述符生成** - 通过 `parse_csf_2_descriptor()` 将 CSF 转换为数值描述符
3. **ML 训练** - 使用 `train_model()` 训练神经网络以预测重要的 CSF
4. **CSF 选择** - 使用 ML 预测为下一次迭代选择 CSF
5. **收敛检查** - 通过 `evaluate_calculation_convergence()` 监控能量和 CSF 计数
6. **迭代** - 重复直到满足收敛标准

## 环境特定行为

GraspKit 自动检测 HPC/SLURM 环境：
- 在 SLURM 作业中隐藏进度条
- 调试模式启用详细日志记录
- GPU 支持自动 CUDA 设备选择

```python
# 检查当前环境
from graspkit.utils.environment_config import is_slurm_job, is_debug_mode
print(f"SLURM 作业: {is_slurm_job()}")
print(f"调试模式: {is_debug_mode()}")
```

## 项目链接

- **主页**: https://github.com/YenochQin/graspkit
- **文档**: [INSTALL.md](INSTALL.md)、[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)
- **许可证**: 参见 LICENSE 文件

## 引用

如果您在研究中使用了 GraspKit，请引用：

```bibtex
@software{graspkit2024,
  author = {Qin, Yi (秦亦)},
  title = {GraspKit: ML-driven CSF Selection for GRASP Calculations},
  version = {2.9.1},
  year = {2024},
  url = {https://github.com/YenochQin/graspkit}
}
```

## 许可证

本项目根据 LICENSE 文件中指定的条款进行许可。
