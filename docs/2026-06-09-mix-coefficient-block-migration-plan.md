# MixCoefficientBlock Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a block-centric `MixCoefficientBlock` model for rmix data while preserving current `MixCoefficientData` compatibility fields during migration.

**Architecture:** Introduce `MixCoefficientBlock` in `utils/data_modules.py` and add a `blocks` field to `MixCoefficientData`. Make legacy parallel-list fields optional derived properties so existing code keeps working, then migrate high-value consumers to `blocks` incrementally. Keep the loader reading process block-local because `MixCoefLoader.load()` already reads all metadata and coefficient matrix for one block inside one loop iteration.

**Tech Stack:** Python 3.14+, dataclasses, NumPy, `numpy.typing.NDArray`, pytest, existing `MixCoefLoader`.

---

## Current Context

Relevant files:

- `src/graspkit/utils/data_modules.py`
  - Defines `MixCoefficientData` as a frozen dataclass with many parallel lists.
  - Current fields include `block_idx_list`, `block_CSFs_nums`, `block_energy_count_list`, `block_levels_idx_list`, `block_energy_list`, `block_level_energy_list`, and `mix_coefficient_list`.
- `src/graspkit/data_IO/loaders/mix_coef_loader.py`
  - `MixCoefLoader.load()` already loops over rmix blocks.
  - Each loop has complete block data: block index, CSF count, level count, J location, parity, level indices, base energy, level energies, and coefficient matrix.
- `src/graspkit/grasp_data_extractor/rmix_data_processor.py`
  - Currently consumes `mix_data.mix_coefficient_list` and `mix_data.block_idx_list`.
  - This should become an early consumer of `mix_data.blocks`.
- `src/graspkit/CSFs_processor/CSFs_choosing.py`
  - Several legacy functions consume the old parallel lists.
  - Keep these working first; migrate only after the block model is proven.
- `src/graspkit/ml_module/ml_initializer.py`
  - Uses `rmix_file_data.block_CSFs_nums[0]` and `rmix_file_data.mix_coefficient_list[0]`.
  - Leave compatibility intact in the first pass.
- `src/graspkit/data_IO/loaders/mix_coef_loader.py::print_mix_coef_levels_rich`
  - Reconstructs block-level level display from old lists.
  - Should migrate to `data.blocks` after `blocks` is available.

Important observation:

`level_J_value_list` is currently ambiguous. `MixCoefLoader.load()` builds it from `temp_J`, which is level-level data after iterating selected level positions, but `print_mix_coef_levels_rich()` indexes it by `jblock` as if it were block-level data. The block model should include `j_value` per block and use that in display logic.

## Files

- Modify: `src/graspkit/utils/data_modules.py`
  - Add `MixCoefficientBlock`.
  - Add `blocks` to `MixCoefficientData`.
  - Provide compatibility properties for the old parallel-list names.
- Modify: `src/graspkit/data_IO/loaders/mix_coef_loader.py`
  - Construct `MixCoefficientBlock` during the existing block loop.
  - Create `MixCoefficientData(blocks=..., level_list=...)`.
  - Migrate `print_mix_coef_levels_rich()` to iterate `data.blocks`.
- Modify: `src/graspkit/grasp_data_extractor/rmix_data_processor.py`
  - Use `mix_data.blocks` in `load_rmix_ci_squared()`.
- Modify: `tests/test_rmix_data_processor.py`
  - Update dummy `MixCoefficientData` construction to include `blocks`, or use a shared helper.
  - Add tests for `load_rmix_ci_squared()` consuming blocks.
- Create: `tests/test_mix_coefficient_data_blocks.py`
  - Cover `MixCoefficientBlock`, compatibility properties, and rich-print data reconstruction behavior without binary fixtures.

Do not migrate all legacy consumers in the first implementation. The purpose of this plan is to add the block model safely and migrate only the loader display and new rmix processor path.

---

## Task 1: Add `MixCoefficientBlock` and Compatibility Properties

**Files:**

- Modify: `src/graspkit/utils/data_modules.py`
- Create: `tests/test_mix_coefficient_data_blocks.py`

- [ ] **Step 1: Write failing tests for block model and compatibility fields**

Create `tests/test_mix_coefficient_data_blocks.py`:

```python
import numpy as np

from graspkit.utils.data_modules import MixCoefficientBlock, MixCoefficientData


def test_mix_coefficient_data_derives_legacy_lists_from_blocks() -> None:
    block0 = MixCoefficientBlock(
        block_index=0,
        csf_count=3,
        level_count=2,
        j_value_location=1,
        j_value="0",
        parity=1,
        level_indices=np.array([0, 1], dtype=np.int64),
        base_energy=0.5,
        level_energies=np.array([0.0, 0.1], dtype=np.float64),
        mix_coefficients=np.array(
            [
                [0.5, 0.1, 0.2],
                [0.0, 0.4, 0.1],
            ],
            dtype=np.float64,
        ),
    )
    block1 = MixCoefficientBlock(
        block_index=1,
        csf_count=2,
        level_count=1,
        j_value_location=2,
        j_value="1/2",
        parity=2,
        level_indices=np.array([0], dtype=np.int64),
        base_energy=0.8,
        level_energies=np.array([0.05], dtype=np.float64),
        mix_coefficients=np.array([[0.3, 0.4]], dtype=np.float64),
    )

    data = MixCoefficientData(
        blocks=[block0, block1],
        level_list=[0.5, 0.6, 0.85],
    )

    assert data.block_num == 2
    assert data.block_idx_list == [0, 1]
    assert data.block_CSFs_nums == [3, 2]
    assert data.block_energy_count_list == [2, 1]
    assert data.level_J_value_list == ["0", "1/2"]
    assert data.parity_list == [1, 2]
    assert data.block_energy_list == [0.5, 0.8]
    np.testing.assert_array_equal(data.block_levels_idx_list[0], np.array([0, 1]))
    np.testing.assert_allclose(data.block_level_energy_list[1], np.array([0.05]))
    np.testing.assert_allclose(
        data.mix_coefficient_list[0],
        np.array([[0.5, 0.1, 0.2], [0.0, 0.4, 0.1]]),
    )
```

- [ ] **Step 2: Run the new test and verify it fails**

Run from `graspkit/`:

```bash
/home/workstation3/AppFiles/GraspKit-Workspace/graspkit-tools/.venv/bin/python -m pytest tests/test_mix_coefficient_data_blocks.py -v
```

Expected: FAIL with `ImportError` for `MixCoefficientBlock` or `TypeError` because `MixCoefficientData` does not accept `blocks`.

- [ ] **Step 3: Implement `MixCoefficientBlock` and compatibility properties**

Modify `src/graspkit/utils/data_modules.py` near `MixCoefficientData`:

```python
@dataclass(frozen=True)
class MixCoefficientBlock:
    """Parsed mixing coefficients and metadata for one rmix block."""

    block_index: int
    csf_count: int
    level_count: int
    j_value_location: int
    j_value: str
    parity: int
    level_indices: NDArray[np.int64]
    base_energy: float
    level_energies: NDArray[np.float64]
    mix_coefficients: NDArray[np.float64]
```

Replace the `MixCoefficientData` field list with:

```python
@dataclass(frozen=True)
class MixCoefficientData:
    """Container for parsed ASF mixing coefficients grouped by block."""

    blocks: list[MixCoefficientBlock]
    level_list: list[float]

    @property
    def block_num(self) -> int:
        return len(self.blocks)

    @property
    def block_idx_list(self) -> list[int]:
        return [block.block_index for block in self.blocks]

    @property
    def block_CSFs_nums(self) -> list[int]:
        return [block.csf_count for block in self.blocks]

    @property
    def block_energy_count_list(self) -> list[int]:
        return [block.level_count for block in self.blocks]

    @property
    def level_J_value_list(self) -> list[str]:
        return [block.j_value for block in self.blocks]

    @property
    def parity_list(self) -> list[int]:
        return [block.parity for block in self.blocks]

    @property
    def block_levels_idx_list(self) -> list[NDArray[np.int64]]:
        return [block.level_indices for block in self.blocks]

    @property
    def block_energy_list(self) -> list[float]:
        return [block.base_energy for block in self.blocks]

    @property
    def block_level_energy_list(self) -> list[NDArray[np.float64]]:
        return [block.level_energies for block in self.blocks]

    @property
    def mix_coefficient_list(self) -> list[NDArray[np.float64]]:
        return [block.mix_coefficients for block in self.blocks]
```

- [ ] **Step 4: Run the new test and verify it passes**

Run:

```bash
/home/workstation3/AppFiles/GraspKit-Workspace/graspkit-tools/.venv/bin/python -m pytest tests/test_mix_coefficient_data_blocks.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/graspkit/utils/data_modules.py tests/test_mix_coefficient_data_blocks.py
git commit -m "utils: add mix coefficient block model"
```

Expected: commit succeeds inside the `graspkit` submodule.

---

## Task 2: Update Test Fixtures to Use `blocks`

**Files:**

- Modify: `tests/test_rmix_data_processor.py`

- [ ] **Step 1: Update imports**

Modify the import:

```python
from graspkit.utils.data_modules import MixCoefficientBlock, MixCoefficientData
```

- [ ] **Step 2: Add a test helper for dummy `MixCoefficientData`**

Add near the top of `tests/test_rmix_data_processor.py`:

```python
def _mix_block(
    block_index: int,
    coefficients: np.ndarray,
    *,
    j_value: str = "0",
    parity: int = 1,
) -> MixCoefficientBlock:
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    return MixCoefficientBlock(
        block_index=block_index,
        csf_count=coefficient_array.shape[1],
        level_count=coefficient_array.shape[0],
        j_value_location=block_index + 1,
        j_value=j_value,
        parity=parity,
        level_indices=np.arange(coefficient_array.shape[0], dtype=np.int64),
        base_energy=float(block_index),
        level_energies=np.arange(coefficient_array.shape[0], dtype=np.float64) * 0.1,
        mix_coefficients=coefficient_array,
    )
```

- [ ] **Step 3: Replace direct `MixCoefficientData(...)` fixtures**

Replace the loader dummy in `test_load_rmix_ci_squared_returns_selected_asf_square_data()` with:

```python
return MixCoefficientData(
    blocks=[
        _mix_block(
            0,
            np.array([[0.5, 0.1, 0.2], [0.0, 0.4, 0.1]]),
            j_value="0",
            parity=1,
        ),
        _mix_block(
            1,
            np.array([[0.3, 0.4], [0.5, 0.1]]),
            j_value="1",
            parity=1,
        ),
    ],
    level_list=[0.0, 0.1],
)
```

Replace the fixture in `test_legacy_batch_threshold_returns_unique_indices_per_block()` with:

```python
mix_data = MixCoefficientData(
    blocks=[
        _mix_block(
            0,
            np.array(
                [
                    [0.5, 0.1, 0.0],
                    [0.0, 0.3, 0.1],
                ]
            ),
        )
    ],
    level_list=[0.0, 0.1],
)
```

- [ ] **Step 4: Run rmix processor tests**

Run:

```bash
/home/workstation3/AppFiles/GraspKit-Workspace/graspkit-tools/.venv/bin/python -m pytest tests/test_rmix_data_processor.py tests/test_mix_coefficient_data_blocks.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add tests/test_rmix_data_processor.py
git commit -m "tests: use mix coefficient block fixtures"
```

Expected: commit succeeds.

---

## Task 3: Construct Blocks in `MixCoefLoader.load()`

**Files:**

- Modify: `src/graspkit/data_IO/loaders/mix_coef_loader.py`
- Modify: `src/graspkit/utils/data_modules.py` if import updates are needed

- [ ] **Step 1: Write a loader-construction unit test with monkeypatched reads**

Append to `tests/test_mix_coefficient_data_blocks.py`:

```python
from pathlib import Path
from typing import Any

import pytest

from graspkit.data_IO.loaders.mix_coef_loader import MixCoefLoader


def test_mix_coef_loader_builds_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    records = iter(
        [
            np.array(list(b"G92MIX"), dtype="S1"),
            np.array([2, 3, 4, 5, 6, 1], dtype=np.int32),
            np.array([1, 3, 2, 1, 1], dtype=np.int32),
            np.array([1, 2], dtype=np.int32),
            np.array([0.5, 0.0, 0.1], dtype=np.float64),
            np.array([0.5, 0.1, 0.2, 0.0, 0.4, 0.1], dtype=np.float64),
        ]
    )

    def fake_read_fortran_record(self: MixCoefLoader, file: Any, dtype: str, count: int) -> np.ndarray:
        return next(records)

    def fake_read_mixed_scalars(self: MixCoefLoader, file: Any, field_specs: list[str]) -> np.ndarray:
        return next(records)

    printed: list[MixCoefficientData] = []
    monkeypatch.setattr(MixCoefLoader, "read_fortran_record", fake_read_fortran_record)
    monkeypatch.setattr(MixCoefLoader, "read_mixed_scalars", fake_read_mixed_scalars)
    monkeypatch.setattr(
        "graspkit.data_IO.loaders.mix_coef_loader.print_mix_coef_levels_rich",
        lambda data: printed.append(data),
    )

    mix_path = tmp_path / "example.m"
    mix_path.write_bytes(b"dummy")

    result = MixCoefLoader(mix_path).load()

    assert len(result.blocks) == 1
    block = result.blocks[0]
    assert block.block_index == 0
    assert block.csf_count == 3
    assert block.level_count == 2
    assert block.j_value_location == 1
    assert block.j_value == "0"
    assert block.parity == 1
    assert block.base_energy == 0.5
    np.testing.assert_array_equal(block.level_indices, np.array([0, 1]))
    np.testing.assert_allclose(block.level_energies, np.array([0.0, 0.1]))
    np.testing.assert_allclose(
        block.mix_coefficients,
        np.array([[0.5, 0.1, 0.2], [0.0, 0.4, 0.1]]),
    )
    assert printed == [result]
```

- [ ] **Step 2: Run the loader-construction test and verify it fails**

Run:

```bash
/home/workstation3/AppFiles/GraspKit-Workspace/graspkit-tools/.venv/bin/python -m pytest tests/test_mix_coefficient_data_blocks.py::test_mix_coef_loader_builds_blocks -v
```

Expected: FAIL until `MixCoefLoader.load()` constructs `MixCoefficientBlock`.

- [ ] **Step 3: Import `MixCoefficientBlock` in loader**

Change:

```python
from ...utils.data_modules import MixCoefficientData
```

to:

```python
from ...utils.data_modules import MixCoefficientBlock, MixCoefficientData
```

- [ ] **Step 4: Replace parallel-list construction inside `load()`**

Inside `MixCoefLoader.load()`, replace the list initializations:

```python
idx_block_list: list[int] = []
ncfblk_list: list[int] = []
block_energy_count_list: list[int] = []
j_value_location_list: list[int] = []
parity_list: list[int] = []
ivec_list: list[NDArray[np.int32]] = []
block_energy_list: list[float] = []
block_level_energy_list: list[NDArray[np.float64]] = []
mix_coefficient_list: list[NDArray[np.float64]] = []
```

with:

```python
blocks: list[MixCoefficientBlock] = []
```

Inside the block loop, remove all `.append(...)` calls to old parallel lists and append one block after `evecs` validation:

```python
block_index = nb - 1
j_value = _J_VALUE_LIST[iatjp - 1]
blocks.append(
    MixCoefficientBlock(
        block_index=block_index,
        csf_count=ncfblk,
        level_count=nevblk,
        j_value_location=iatjp,
        j_value=j_value,
        parity=iaspa,
        level_indices=ivec_array.astype(np.int64, copy=False),
        base_energy=eav,
        level_energies=evals,
        mix_coefficients=evecs,
    )
)
```

- [ ] **Step 5: Rebuild level display data from blocks**

Replace:

```python
for jblock in range(nblock):
    for pos in ivec_list[jblock].tolist():
        temp_pos.append(pos)
        temp_J.append(_J_VALUE_LIST[j_value_location_list[jblock] - 1])
        temp_parity_idx.append(parity_list[jblock])
        temp_energy.append(
            float(block_energy_list[jblock] + block_level_energy_list[jblock][pos])
        )
```

with:

```python
for block in blocks:
    for pos in block.level_indices.tolist():
        temp_pos.append(pos)
        temp_J.append(block.j_value)
        temp_parity_idx.append(block.parity)
        temp_energy.append(float(block.base_energy + block.level_energies[pos]))
```

- [ ] **Step 6: Create `MixCoefficientData` from blocks**

Replace the current constructor call with:

```python
data = MixCoefficientData(
    blocks=blocks,
    level_list=level_energy_list,
)
```

- [ ] **Step 7: Run loader and model tests**

Run:

```bash
/home/workstation3/AppFiles/GraspKit-Workspace/graspkit-tools/.venv/bin/python -m pytest tests/test_mix_coefficient_data_blocks.py tests/test_rmix_data_processor.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

Run:

```bash
git add src/graspkit/data_IO/loaders/mix_coef_loader.py tests/test_mix_coefficient_data_blocks.py
git commit -m "data_IO: build mix coefficient blocks in loader"
```

Expected: commit succeeds.

---

## Task 4: Migrate Rich Printer and Rmix Processor to Blocks

**Files:**

- Modify: `src/graspkit/data_IO/loaders/mix_coef_loader.py`
- Modify: `src/graspkit/grasp_data_extractor/rmix_data_processor.py`
- Modify: `tests/test_mix_coefficient_data_blocks.py`
- Modify: `tests/test_rmix_data_processor.py`

- [ ] **Step 1: Add test for rich printer using block J values**

Append to `tests/test_mix_coefficient_data_blocks.py`:

```python
def test_print_mix_coef_levels_uses_block_j_values(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[tuple[str, ...]] = []

    class DummyTable:
        def add_column(self, *args: object, **kwargs: object) -> None:
            return None

        def add_row(self, *values: str) -> None:
            rows.append(values)

    class DummyConsole:
        def print(self, *args: object, **kwargs: object) -> None:
            return None

    monkeypatch.setattr("graspkit.data_IO.loaders.mix_coef_loader.Table", DummyTable)
    monkeypatch.setattr("graspkit.data_IO.loaders.mix_coef_loader.Console", DummyConsole)

    data = MixCoefficientData(
        blocks=[
            MixCoefficientBlock(
                block_index=0,
                csf_count=1,
                level_count=1,
                j_value_location=1,
                j_value="0",
                parity=1,
                level_indices=np.array([0], dtype=np.int64),
                base_energy=0.0,
                level_energies=np.array([0.0], dtype=np.float64),
                mix_coefficients=np.array([[1.0]], dtype=np.float64),
            ),
            MixCoefficientBlock(
                block_index=1,
                csf_count=1,
                level_count=1,
                j_value_location=2,
                j_value="1/2",
                parity=2,
                level_indices=np.array([0], dtype=np.int64),
                base_energy=0.1,
                level_energies=np.array([0.0], dtype=np.float64),
                mix_coefficients=np.array([[1.0]], dtype=np.float64),
            ),
        ],
        level_list=[0.0, 0.1],
    )

    from graspkit.data_IO.loaders.mix_coef_loader import print_mix_coef_levels_rich

    print_mix_coef_levels_rich(data)

    assert rows[0][2] == "0"
    assert rows[1][2] == "1/2"
```

- [ ] **Step 2: Run the rich-printer test**

Run:

```bash
/home/workstation3/AppFiles/GraspKit-Workspace/graspkit-tools/.venv/bin/python -m pytest tests/test_mix_coefficient_data_blocks.py::test_print_mix_coef_levels_uses_block_j_values -v
```

Expected: FAIL if current printer still indexes `level_J_value_list` by block with ambiguous data.

- [ ] **Step 3: Migrate `print_mix_coef_levels_rich()` to blocks**

In `src/graspkit/data_IO/loaders/mix_coef_loader.py`, replace:

```python
for jblock in range(data.block_num):
    for pos in data.block_levels_idx_list[jblock].tolist():
        temp_pos.append(pos)
        temp_J.append(data.level_J_value_list[jblock])
        temp_parity_idx.append(data.parity_list[jblock])
        temp_energy.append(
            float(data.block_energy_list[jblock] + data.block_level_energy_list[jblock][pos])
        )
```

with:

```python
for block in data.blocks:
    for pos in block.level_indices.tolist():
        temp_pos.append(pos)
        temp_J.append(block.j_value)
        temp_parity_idx.append(block.parity)
        temp_energy.append(float(block.base_energy + block.level_energies[pos]))
```

- [ ] **Step 4: Migrate `load_rmix_ci_squared()` to blocks**

In `src/graspkit/grasp_data_extractor/rmix_data_processor.py`, replace:

```python
for block_index, block_coefficients in enumerate(mix_data.mix_coefficient_list):
    coefficient_array = _as_1d_or_2d_float_array(block_coefficients, "coefficients")
```

with:

```python
for block_index, block in enumerate(mix_data.blocks):
    coefficient_array = _as_1d_or_2d_float_array(block.mix_coefficients, "coefficients")
```

Replace:

```python
block_indices=list(mix_data.block_idx_list),
```

with:

```python
block_indices=[block.block_index for block in mix_data.blocks],
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
/home/workstation3/AppFiles/GraspKit-Workspace/graspkit-tools/.venv/bin/python -m pytest tests/test_mix_coefficient_data_blocks.py tests/test_rmix_data_processor.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add src/graspkit/data_IO/loaders/mix_coef_loader.py src/graspkit/grasp_data_extractor/rmix_data_processor.py tests/test_mix_coefficient_data_blocks.py
git commit -m "grasp_data_extractor: consume mix coefficient blocks"
```

Expected: commit succeeds.

---

## Task 5: Final Compatibility Verification

**Files:**

- No source edits expected.

- [ ] **Step 1: Run focused tests**

Run:

```bash
/home/workstation3/AppFiles/GraspKit-Workspace/graspkit-tools/.venv/bin/python -m pytest tests/test_mix_coefficient_data_blocks.py tests/test_rmix_data_processor.py -v
```

Expected: PASS.

- [ ] **Step 2: Run import and processor tests**

Run:

```bash
/home/workstation3/AppFiles/GraspKit-Workspace/graspkit-tools/.venv/bin/python -m pytest tests/test_asfs_data_processor.py tests/test_logical_packages.py tests/test_import_boundaries.py -v
```

Expected: PASS.

- [ ] **Step 3: Run tests for known rmix consumers**

Run:

```bash
/home/workstation3/AppFiles/GraspKit-Workspace/graspkit-tools/.venv/bin/python -m pytest tests/test_iterative_training_mode.py tests/test_streaming_descriptors.py -v
```

Expected: PASS. These tests cover downstream ML paths that rely on `MixCoefficientData` compatibility fields or CI-square data.

- [ ] **Step 4: Run ruff on touched files**

Run:

```bash
/home/workstation3/AppFiles/GraspKit-Workspace/graspkit-tools/.venv/bin/python -m ruff check src/graspkit/utils/data_modules.py src/graspkit/data_IO/loaders/mix_coef_loader.py src/graspkit/grasp_data_extractor/rmix_data_processor.py tests/test_mix_coefficient_data_blocks.py tests/test_rmix_data_processor.py
```

Expected: PASS.

- [ ] **Step 5: Inspect submodule status**

Run:

```bash
git status --short --branch
```

Expected: clean working tree inside `graspkit/` after commits.

- [ ] **Step 6: Parent workspace gitlink**

Run from `/home/workstation3/AppFiles/GraspKit-Workspace`:

```bash
git status --short --branch
git diff --submodule
```

Expected: parent workspace shows `M graspkit` with only a submodule gitlink update. Commit the parent gitlink after the submodule commits are complete:

```bash
git add graspkit
git commit -m "workspace: update graspkit submodule"
```

---

## Deferred Follow-Ups

- Remove old parallel-list compatibility properties after all consumers migrate to `MixCoefficientData.blocks`.
- Rename old field users in `CSFs_processor/CSFs_choosing.py` and `ml_module/ml_initializer.py` to block-based access.
- Re-evaluate whether `level_J_value_list` should exist after block migration; block-level `j_value` should be the preferred source for display and block metadata.
- Add binary rmix fixture tests if stable binary fixtures become available.

## Self-Review

- Spec coverage: The plan adds `MixCoefficientBlock`, keeps compatibility fields, migrates loader construction, migrates rich printer and new rmix processor use, and verifies downstream compatibility.
- Placeholder scan: The plan contains no `TBD` placeholders and every implementation step has concrete code or exact commands.
- Type consistency: `MixCoefficientBlock`, `MixCoefficientData.blocks`, `level_indices`, `level_energies`, and `mix_coefficients` names are consistent across tasks.
