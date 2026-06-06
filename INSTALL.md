# 安装指南

## 前提

- Python `>=3.14`
- 建议在独立虚拟环境中安装
- 如果要使用 GPU 版 PyTorch，需要 NVIDIA CUDA 环境

项目依赖通过 extras 管理：

- `cpu`：安装 CPU 版 PyTorch 运行依赖
- `gpu`：安装 CUDA 版 PyTorch 运行依赖
- `dev`：安装测试、类型检查、构建等开发依赖

## 方式一：使用 UV

推荐使用 `uv`，因为当前仓库已经为 extras 和 PyTorch CUDA 源做了配置。

### 安装 UV

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 创建环境并安装

```bash
git clone https://github.com/YenochQin/graspkit.git
cd graspkit

uv venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

### CPU 运行环境

```bash
uv sync --extra cpu
```

### GPU 运行环境

```bash
uv sync --extra gpu
```

### 开发环境

```bash
# CPU 开发环境
uv sync --extra cpu --extra dev

# GPU 开发环境
uv sync --extra gpu --extra dev
```

### 常用命令

```bash
uv run python -c "import graspkit; print(graspkit.__version__)"
uv run pytest tests/
uv run ruff check .
uv run mypy src/
```

## 方式二：使用 pip

如果你不使用 `uv`，也可以直接基于 extras 安装。当前仓库没有 `requirements-cpu.txt` 或 `requirements-gpu.txt`，请不要再使用旧文档中的那些命令。

### 创建虚拟环境

```bash
python3.14 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

### 安装 CPU 环境

```bash
pip install -e ".[cpu]"
```

### 安装 GPU 环境

GPU 版依赖在 `uv` 下会自动走项目配置的 PyTorch CUDA 源；如果你用 `pip`，通常需要先按 PyTorch 官方方式安装匹配 CUDA 的 wheel，再安装项目本体。例如：

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

如果你要装开发依赖：

```bash
pip install -e ".[cpu,dev]"
```

说明：

- 在 `zsh` 里，extras 最好加引号，避免 `[]` 被 shell 展开
- 若你已经手动安装 GPU 版 `torch`/`torchvision`，再执行 `pip install -e .` 即可安装项目本体

## 构建与开发依赖

开发环境推荐直接安装 `dev` extra：

```bash
uv sync --extra cpu --extra dev
```

安装后可使用这些命令：

```bash
ruff check .
ruff check . --fix
mypy src/
pytest tests/
python -m build
python build_package.py --clean
```

仓库当前可见的测试入口主要是：

- `pytest tests/`
- `python test/test.ipynb` 不适合作为命令行测试入口

## 验证安装

### 最小验证

```bash
python -c "import graspkit; print(graspkit.__version__)"
```

### 依赖验证

```bash
python - <<'PY'
import graspkit
import numpy
import polars
import torch

print("graspkit:", graspkit.__version__)
print("numpy:", numpy.__version__)
print("polars:", polars.__version__)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
PY
```

### 简单功能验证

```bash
python - <<'PY'
from graspkit import ANNClassifier, EnergyFileLoader, MLCalConfig

print("core imports ok")
print("ANNClassifier:", ANNClassifier.__name__)
print("EnergyFileLoader:", EnergyFileLoader.__name__)
print("MLCalConfig:", MLCalConfig.__name__)
PY
```

## 版本与环境说明

当前仓库元数据中的关键点：

- 包名：`grasp-kit`
- 当前版本：`3.2.dev1`
- Python 要求：`>=3.14`
- PyTorch 通过 optional dependencies 安装
- `gpu` extra 使用 `https://download.pytorch.org/whl/cu128`

如果你的 Python 版本低于 3.14，安装失败是预期行为。

## 常见问题

### 1. `uv sync` 失败，提示 Python 版本不满足

检查解释器版本：

```bash
python --version
uv python list
```

然后使用 3.14 环境重新创建虚拟环境。

### 2. GPU 安装失败

先确认 CUDA 和驱动可用：

```bash
nvidia-smi
```

如果只是想先跑功能或测试，优先切到 CPU 环境：

```bash
uv sync --extra cpu --extra dev
```

### 3. pip 安装 GPU 版 PyTorch 失败

这是因为项目里的 CUDA wheel 源配置主要服务于 `uv`。使用 `pip` 时，请直接按 PyTorch 官方方式指定 `--index-url` 安装对应 CUDA 版本的 wheel。

### 4. 想只安装项目本体，不要 PyTorch

可以直接：

```bash
pip install -e .
```

但要注意：

- `graspkit` 中的部分 ML 功能依赖 `torch`
- 如果你只使用数据加载、表格处理、部分工具函数，这种安装方式通常够用

## 推荐安装组合

大多数情况直接选下面其中之一：

- 日常开发：`uv sync --extra cpu --extra dev`
- 有 CUDA 的训练环境：`uv sync --extra gpu --extra dev`
- 只做运行验证：`uv sync --extra cpu`
