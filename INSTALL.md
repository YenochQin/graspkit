# 安装指南

本项目提供了多种安装方式，推荐使用 Pixi 进行现代化的环境管理。

#### 方法一：使用UV (推荐)
UV 是超快速的Python包和项目管理器，提供极快的依赖解析和安装。

```bash
# 安装 UV (如果尚未安装)
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目并进入目录
git clone https://github.com/YenochQin/graspkit-tools.git
cd graspkit-tools

# 创建虚拟环境
uv venv

# 激活环境
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 安装依赖
uv pip install -e .

# 运行特定命令
uv run python your_script.py
```

**UV 环境特性**：
- 超快的依赖解析和安装（比pip快10-100倍）
- 自动管理Python版本 (>=3.12)
- 支持CPU和GPU两种环境配置
- 跨平台支持 (Linux, Windows, macOS)
- 现代化的锁文件机制 (uv.lock)
- 隔离的开发环境
- 与pip完全兼容

### 环境管理

```bash
# 查看可用环境
pixi info

# 切换到 GPU 环境
pixi shell --feature gpu

# 添加开发依赖
pixi add --feature dev pytest black flake8

# 更新依赖
pixi update
```

### Pixi 优势

- **自动版本管理**：无需手动管理 Python 版本
- **跨平台兼容**：支持 Linux、Windows、macOS
- **环境隔离**：不同项目使用不同依赖版本
- **快速安装**：并行下载和安装依赖
- **GPU 支持**：自动处理 CUDA 依赖

---

## 📦 方法二：传统 pip 安装

如果您更喜欢使用传统的 pip 安装方式，我们仍然提供了相应的配置文件。

### 环境选择

#### 🖥️ CPU环境安装
适用于：
- 没有GPU的机器
- 不需要GPU加速的场景
- 快速测试和开发
- 资源受限的环境

```bash
# 创建虚拟环境
python -m venv grasp_env

# 激活环境 (Windows)
grasp_env\Scripts\activate

# 激活环境 (Linux/Mac)
source grasp_env/bin/activate

# 安装依赖
pip install -r requirements-cpu.txt

# 开发模式安装
pip install -e .
```

#### 🚀 GPU环境安装
适用于：
- 有NVIDIA GPU的机器
- 需要深度学习加速的场景
- 大规模模型训练
- 高性能计算需求

```bash
# 创建虚拟环境
python -m venv grasp_env

# 激活环境
grasp_env\Scripts\activate  # Windows
source grasp_env/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements-gpu.txt

# 开发模式安装
pip install -e .
```

**前提条件**：
- 安装了NVIDIA GPU驱动
- 安装了合适版本的CUDA (推荐11.8+)
- 确认GPU可用：`nvidia-smi`

---

## 🔍 验证安装

### 快速验证

```bash
# 使用 Pixi
pixi run python -c "
import torch
import numpy as np
import pandas as pd
import sklearn
import graspkit
print(f'✅ graspkit 版本: {graspkit.__version__}')
print(f'✅ PyTorch版本: {torch.__version__}')
print(f'✅ NumPy版本: {np.__version__}')
print(f'✅ GPU可用: {torch.cuda.is_available()}')
"

# 使用传统环境
python -c "
import torch
import numpy as np
import pandas as pd
import sklearn
import graspkit
print(f'✅ graspkit 版本: {graspkit.__version__}')
print(f'✅ PyTorch版本: {torch.__version__}')
print(f'✅ NumPy版本: {np.__version__}')
print(f'✅ GPU可用: {torch.cuda.is_available()}')
"
```

### 功能测试

```python
# 测试核心功能
from graspkit.data_IO import GraspFileLoad
from graspkit.ml_module import NeuralNetwork
from graspkit.utils import calculate_energy

print("✅ 核心模块导入成功")

# 测试数据处理功能
try:
    # 这里可以添加具体的功能测试
    print("✅ 功能测试通过")
except Exception as e:
    print(f"❌ 功能测试失败: {e}")
```

---

## 🛠️ 开发环境设置

### 使用 Pixi 开发

```bash
# 添加开发依赖
pixi add --feature dev pytest black ruff mypy

# 运行测试
pixi run pytest

# 代码格式化
pixi run black .

# 代码检查
pixi run ruff check .
```

### 传统开发环境

```bash
# 安装开发依赖
pip install pytest black ruff mypy

# 运行测试
pytest

# 代码格式化
black .

# 代码检查
ruff check .
```

---

## 📋 系统要求

### 最低要求
- **操作系统**: Linux, Windows 10+, macOS 10.15+
- **Python**: 3.12+ (Pixi 自动管理)
- **内存**: 4GB RAM (推荐 8GB+)
- **存储**: 2GB 可用空间

### GPU 环境要求
- **GPU**: NVIDIA GPU (支持 CUDA)
- **CUDA**: 11.8+ (推荐 12.0+)
- **GPU 内存**: 4GB+ (推荐 8GB+)

---

## 🔧 故障排除

### 常见问题

1. **Pixi 安装失败**
   ```bash
   # 检查网络连接
   curl -I https://pixi.sh/install.sh

   # 使用代理 (如果需要)
   https_proxy=your_proxy curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. **GPU 环境无法使用 CUDA**
   ```bash
   # 检查 CUDA 安装
   nvidia-smi

   # 检查 PyTorch CUDA 支持
   pixi run python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **依赖冲突**
   ```bash
   # 清理并重新安装
   pixi clean
   pixi install

   # 或者使用传统方式
   pip install --upgrade pip
   pip install -r requirements-cpu.txt --force-reinstall
   ```

4. **导入错误**
   ```bash
   # 检查安装路径
   pixi run python -c "import graspkit; print(graspkit.__file__)"

   # 确保开发模式安装
   pixi run pip install -e .
   ```

### 获取帮助

如果遇到安装问题，请提供以下信息：
- 操作系统版本
- Python 版本
- 使用的安装方法 (Pixi/pip)
- 错误信息完整输出
- `pixi info` 或 `pip list` 的输出

---

## 📚 相关文档

- [项目主页](https://github.com/YenochQin/graspkit-tools)
- [Pixi 官方文档](https://pixi.sh)
- [API 文档](docs/)
- [使用示例](examples/)