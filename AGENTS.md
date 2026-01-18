# AGENTS.md

Guide for agentic coding assistants working in this repository.

## Essential Commands

### Installation & Setup
```bash
# Recommended: UV package manager
uv venv && source .venv/bin/activate  # macOS/Linux
uv sync --extra cpu --extra dev        # CPU version with dev tools
uv sync --extra gpu --extra dev         # GPU version (NVIDIA CUDA)

# Alternative: Pixi
pixi install && pixi shell

# Verify
python -c "import graspkit; print('Package OK')"
```

### Build & Quality
```bash
# Build package
python build_package.py --clean         # Clean build
python -m build                        # Manual build

# Linting (NumPy 2.0 compatible)
ruff check .                          # Check for issues
ruff check . --fix                    # Auto-fix issues

# Type checking (optional)
mypy src/

# Test single file
python test/test.py                   # Run specific test
python tests/ANN.py                   # Run example script
```

## Code Style Guidelines

### File Structure
- **Encoding**: UTF-8 with `# -*- encoding: utf-8 -*-` at file start
- **Module header**: Docstring with `@Id`, `@date`, `@author` fields
- **Python version**: 3.13 only (exclusive)

### Import Organization
```python
# -*- encoding: utf-8 -*-
"""
@Id :module_name.py
@date :YYYY/MM/DD
@author :Your Name
"""

# 标准库导入
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Literal

# 第三方库导入
import numpy as np
import pandas as pd
import torch

# 本地模块导入 (relative imports within graspkit)
from ..utils.data_modules import CSFs
from ..data_IO.grasp_data_loader import GraspFileLoad
```

### Type Hints (Python 3.13 Unified Style)
- Use modern syntax `|` for unions: `x: str | int | None` (NOT `Optional` or `Union`)
- Use builtin types: `list[str]`, `dict[str, int]`, `tuple[int, str]` (NOT `List`, `Dict`, `Tuple` from typing)
- Use `Literal` for string enums: `mode: Literal["standard", "tensornet"]`
- Use `NDArray` for NumPy arrays: `data: NDArray[np.float64]` (import from `numpy.typing`)
- Always include return types: `def process() -> pd.DataFrame:`
- Cast ambiguous returns: `from typing import cast; result = cast(Foo, func())`
- Only import from typing: `Literal`, `cast`, `TypedDict`, `Protocol` (all else use builtin)

### Naming Conventions
- **Functions/Variables**: `snake_case` - `get_csf_data`, `energy_list`
- **Classes**: `PascalCase` - `GraspFileLoad`, `ANNClassifier`
- **Constants**: `UPPER_SNAKE_CASE` - `DEFAULT_LEARNING_RATE`
- **Private methods**: `_leading_underscore` - `_detect_slurm_environment()`
- **Class methods**: Use `@classmethod` for alternative constructors
- **Static methods**: Use `@staticmethod` for utility functions

### Data Structures
```python
from dataclasses import dataclass

@dataclass(frozen=True)  # For immutable data
class MixCoefficientData:
    block_num: int
    block_idx_list: list
    block_energy_list: list

@dataclass  # For mutable data
class CSFs:
    subshell_info_raw: list[str]
    block_num: int

    @classmethod
    def from_dict(cls, data: dict) -> "CSFs":
        """Factory method for dict initialization"""
        return cls(...)
```

### Error Handling
- Type validation in public methods:
  ```python
  def get_csfs_data(self) -> CSFs:
      result = self.load_csfs_file(path)
      if isinstance(result, CSFs):
          return result
      else:
          raise TypeError(f"Expected CSFs object, got {type(result)}")
  ```

- Value checking with descriptive messages:
  ```python
  if ncfblk != len(evecs[0]):
      raise ValueError("ncfblk should equal len(evecs[0])")
  ```

- Never suppress errors with empty catch blocks

### Path Handling
- Always use `pathlib.Path` for file operations
- Validate paths exist before use:
  ```python
  path = Path(filepath)
  if not path.exists():
      raise FileNotFoundError(f"Path not found: {path}")
  ```

### Logging & Progress
- Use Python `logging` module for structured logging
- Environment-aware progress bars (auto-hide in SLURM):
  ```python
  from graspkit.utils.progress_manager import wrap_iterator

  for item in wrap_iterator(items, desc="Processing"):
      ...
  ```

- Use `log_stage_start()` / `log_stage_end()` for workflow phases

### ML/PyTorch Patterns
- Device selection: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Random seeds: `torch.manual_seed(seed)` for reproducibility
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Type safety: Use `torch.float32` / `torch.long` for tensor dtypes

### Environment-Aware Code
- Check SLURM environment: `is_slurm_environment()`
- Check debug mode: `is_debug_mode()`
- Production mode detected automatically (SLURM + non-interactive)
- Use helper functions from `utils/environment_config.py`

### Docstrings
- Module-level docstrings with @Id/@date/@author
- Function docstrings with Args/Returns sections:
  ```python
  def train_model(X: np.ndarray, y: np.ndarray) -> dict:
      """
      Train ML model

      Args:
          X: Training features
          y: Training labels

      Returns:
          Dictionary with training metrics
      """
  ```

## Module Interdependencies
- **data_IO** → Foundation layer (file I/O, loading, saving)
- **CSFs_processor** → Quantum mechanics algorithms
- **ml_module** → Depends on CSFs_processor (processed data)
- **grasp_data_extractor** → GRASP-specific formats
- **utils** → Shared across all modules (environment, progress, tools)

## Testing
- Tests in `test/` directory
- Run specific test: `python test/test.py`
- Example scripts in root: `test/test.py` (existing)
- Use pytest for structured testing (configured in dev dependencies)

## Package Configuration
- Build system: Hatchling (`build-backend = "hatchling.build"`)
- Linter: Ruff with NumPy 2.0 compatibility (`select = ["NPY201"]`)
- Type checker: mypy (optional, not enforced)
- Dependencies: CPU/GPU split via `--extra` flags
- Lock files: `uv.lock` (UV), `pixi.lock` (Pixi)

## Important Constraints
- Python version: 3.13 ONLY (exclusive range: `<3.14`)
- NumPy 2.0+ compatibility required (use `np.typing.NDArray`)
- Never commit without explicit user request
- No type suppression (`as any`, `@ts-ignore`, `@ts-expect-error`)
- Frontend visual changes → delegate to `frontend-ui-ux-engineer` agent
- Complex architecture decisions → consult `oracle` agent

## Code Quality Before Submitting
- Run `ruff check .` and fix all issues
- Run `mypy src/` and address type errors (optional but recommended)
- Build package: `python build_package.py --clean`
- Test changes: `python test/test.py` or relevant example script
- Verify imports are properly ordered and grouped
