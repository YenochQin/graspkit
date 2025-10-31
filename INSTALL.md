# 安装指南

本项目提供了多种安装方式，推荐使用 UV 进行现代化的环境管理。

## 🚀 方法一：使用UV (推荐)
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

# 注意：镜像源已在 pyproject.toml 中配置为清华镜像源
# 如果需要覆盖配置，可以设置环境变量：
# export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"


# 安装依赖 - 必须选择CPU或GPU版本
# CPU版本 (推荐，兼容性好)
uv sync --extra cpu

# GPU版本 (如果有NVIDIA GPU和CUDA)
uv sync --extra gpu

# 开发环境 CPU版本
uv sync --extra dev --extra cpu

# 开发环境 GPU版本
uv sync --extra dev --extra gpu

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

### UV 环境管理

```bash
# 查看UV版本
uv --version

# 创建特定Python版本环境
uv venv --python 3.12

# 注意：镜像源已在 pyproject.toml 的 [tool.uv] 部分配置
# UV 会自动使用配置的清华镜像源加速下载

# 如果需要临时覆盖配置，可以设置环境变量：
# export UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"

# 同步依赖 (使用uv.lock文件) - 注意：需要指定CPU或GPU
# 基础安装无法工作，必须选择以下之一：
uv sync --extra cpu    # CPU版本
uv sync --extra gpu    # GPU版本

# 开发环境安装
uv sync --extra dev --extra cpu    # 开发环境CPU版本
uv sync --extra dev --extra gpu    # 开发环境GPU版本

# 安装所有额外依赖 (包含GPU版PyTorch)
uv sync --all-extras

# 添加新的依赖到pyproject.toml
uv add pytest
uv add torch --extra cpu

# 更新依赖
uv sync --upgrade

# 查看当前环境变量
echo $UV_INDEX_URL
echo $UV_EXTRA_INDEX_URL
```

### UV 优势

- **超快速度**：依赖解析和安装比pip快10-100倍
- **自动版本管理**：无需手动管理 Python 版本 (>=3.12)
- **跨平台兼容**：支持 Linux、Windows、macOS
- **环境隔离**：不同项目使用不同依赖版本
- **现代化锁文件**：使用 uv.lock 确保可重现构建
- **GPU 支持**：支持CPU和GPU两种环境配置
- **完全兼容**：与pip完全兼容

### 📄 pyproject.toml 配置

项目已在 `pyproject.toml` 文件中配置了镜像源：

```toml
[tool.uv]
# Configure package index for faster downloads in China
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
extra-index-url = ["https://pypi.org/simple"]
```

**配置优势**：
- 🎯 **项目级配置**：随项目一起版本控制，团队共享
- 🚀 **自动应用**：无需手动设置环境变量
- 🔄 **灵活覆盖**：仍可通过环境变量临时覆盖
- 📦 **官方支持**：UV 原生支持 pyproject.toml 配置

**临时覆盖配置**：
```bash
# 使用其他镜像源
UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/" uv sync

# 使用官方源
UV_INDEX_URL="https://pypi.org/simple" uv sync
```

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
# 使用 UV 环境
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

# 或者使用 uv run
uv run python -c "
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

### 使用 UV 开发

```bash
# 安装开发版本 (包含所有开发工具) - 必须指定CPU或GPU
# CPU版本开发环境 (推荐)
uv sync --extra dev --extra cpu

# GPU版本开发环境
uv sync --extra dev --extra gpu

# 或者逐个添加开发依赖
uv add pytest --dev
uv add black --dev
uv add ruff --dev
uv add mypy --dev

# 运行测试
uv run pytest

# 代码格式化
uv run black .

# 代码检查
uv run ruff check .

# 自动修复代码格式问题
uv run ruff check . --fix

# 类型检查
uv run mypy src/
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

# 自动修复代码格式问题
ruff check . --fix
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

1. **UV 安装失败**
   ```bash
   # 检查网络连接
   curl -I https://astral.sh/uv/install.sh

   # 使用代理 (如果需要)
   https_proxy=your_proxy curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **依赖下载速度慢**
   ```bash
   # 项目已配置清华镜像源，如果仍需要切换：
   UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/" uv sync

   # 或者使用华为云镜像
   UV_INDEX_URL="https://repo.huaweicloud.com/repository/pypi/simple/" uv sync

   # 测试镜像源速度
   curl -I https://pypi.tuna.tsinghua.edu.cn/simple/
   curl -I https://mirrors.aliyun.com/pypi/simple/
   ```

3. **NVIDIA CUDA 包下载超时**
   ```bash
   # 方法1: 增加网络超时时间
   UV_HTTP_TIMEOUT=120 uv sync --extra dev

   # 方法2: 强制使用CPU版本的PyTorch (推荐)
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   uv sync --extra dev

   # 方法3: 跳过有问题的包，手动安装
   uv sync --extra dev --no-build-isolation

   # 方法4: 使用官方PyTorch源
   UV_INDEX_URL="https://download.pytorch.org/whl/cpu" uv sync --extra dev
   ```

4. **镜像源连接失败**
   ```bash
   # 测试当前配置的镜像源
   curl -I https://pypi.tuna.tsinghua.edu.cn/simple/

   # 临时切换到其他镜像源
   UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/" uv sync

   # 使用官方源作为备选
   UV_INDEX_URL="https://pypi.org/simple" uv sync

   # 检查 pyproject.toml 配置
   cat pyproject.toml | grep -A 5 "\[tool.uv\]"
   ```

5. **GPU 环境无法使用 CUDA**
   ```bash
   # 检查 CUDA 安装
   nvidia-smi

   # 检查 PyTorch CUDA 支持
   python -c "import torch; print(torch.cuda.is_available())"

   # 或者使用 uv run
   uv run python -c "import torch; print(torch.cuda.is_available())"
   ```

6. **依赖冲突**
   ```bash
   # 使用UV清理并重新同步
   uv sync --refresh

   # 或者强制重新安装
   uv sync --reinstall

   # 或者使用传统方式
   pip install --upgrade pip
   pip install -e . --force-reinstall
   ```

7. **导入错误**
   ```bash
   # 检查安装路径
   uv run python -c "import graspkit; print(graspkit.__file__)"

   # 确保包已正确同步
   uv sync

   # 检查虚拟环境
   uv venv --seed
   uv sync
   ```

8. **虚拟环境激活失败**
   ```bash
   # Windows: 确保使用PowerShell或CMD
   .venv\Scripts\activate

   # Linux/macOS: 确保使用bash或zsh
   source .venv/bin/activate
   ```

### 获取帮助

如果遇到安装问题，请提供以下信息：
- 操作系统版本
- Python 版本
- 使用的安装方法 (UV/pip)
- 错误信息完整输出
- `uv --version` 或 `pip list` 的输出

---

## 📚 相关文档

- [项目主页](https://github.com/YenochQin/graspkit-tools)
- [UV 官方文档](https://docs.astral.sh/uv/)
- [构建说明](BUILD_INSTRUCTIONS.md)
- [API 文档](docs/)
- [使用示例](examples/)
