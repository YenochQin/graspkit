# GraspKit Bilous CNN Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Bilous-2023-style 1D CNN classifier module to `GraspKit/src/graspkit/ml_module` and make it usable by the existing GraspKit CSF-selection training pipeline.

**Architecture:** Implement a focused `cnn.py` containing a PyTorch `BilousCNN` module that treats CSF descriptors as `(orbital_sequence, 3_channels)`. Integrate it into the existing `ANNClassifier` wrapper so training, prediction, multi-label BCE loss, checkpoint save/load, and `ml_trainer.py` can keep using the current pipeline interface.

**Tech Stack:** Python 3.14+, PyTorch, NumPy, pytest, GraspKit `ml_module`.

---

## Context And Design Constraints

Bilous 2023 uses a Keras CNN in `pavlobilous/neural_grasp/Re187_init_on_random.py`:

```python
model.add(tf.keras.layers.InputLayer(input_shape=(num_params//3, 3)))
model.add(tf.keras.layers.Conv1D(96, 3, activation="relu"))
model.add(tf.keras.layers.Conv1D(16, 1, activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(150, activation="relu"))
model.add(tf.keras.layers.Dense(120, activation="relu"))
model.add(tf.keras.layers.Dense(90, activation="relu"))
model.add(tf.keras.layers.Dense(2, activation="softmax"))
```

GraspKit currently uses PyTorch and `ANNClassifier` as the common wrapper. `ANNClassifier` already supports `model_architecture="standard"` and `"tensornet"`, uses `BCEWithLogitsLoss` for CSF importance labels, and predicts per-level probabilities via sigmoid. Therefore the GraspKit-compatible CNN should output logits with shape `(batch_size, output_size)`, not a Keras-style 2-class softmax.

The CNN must preserve these GraspKit behaviors:

- Input descriptors remain flat NumPy arrays with shape `(n_samples, 3 * n_orbitals)`.
- Internally the CNN reshapes to `(batch_size, n_orbitals, 3)`.
- Multi-label output supports one or more target levels.
- Existing `fit`, `predict_proba`, `predict_proba_batch`, `save_model`, and `load_model` continue working.
- Existing TensorNet and standard ANN behavior does not change.

## File Structure

- Create: `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/src/graspkit/ml_module/cnn.py`
  - Owns the Bilous-style CNN `nn.Module`.
  - Keeps CNN architecture separate from the larger `neural_network.py` wrapper.

- Modify: `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/src/graspkit/ml_module/neural_network.py`
  - Import `BilousCNN`.
  - Extend `model_architecture` type to include `"cnn"`.
  - Validate CNN channel reshaping.
  - Build `BilousCNN` from `_build_model`.
  - Preserve checkpoint compatibility.

- Modify: `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/src/graspkit/ml_module/ml_trainer.py`
  - Add CNN as an architecture choice, initially as an opt-in or conservative automatic late-stage alternative.
  - Keep current early-iteration safeguards.

- Modify: `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/src/graspkit/ml_module/__init__.py`
  - Export `BilousCNN`.

- Create: `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/tests/test_bilous_cnn_model.py`
  - Unit tests for shape, validation, training compatibility, probability prediction, batch prediction, and checkpoint load.

## Task 1: Add The Bilous CNN Module

**Files:**
- Create: `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/src/graspkit/ml_module/cnn.py`
- Test: `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/tests/test_bilous_cnn_model.py`

- [ ] **Step 1: Write failing shape tests**

Create `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/tests/test_bilous_cnn_model.py` with:

```python
import pytest
import torch

from graspkit.ml_module.cnn import BilousCNN


def test_bilous_cnn_forward_accepts_flat_descriptor() -> None:
    model = BilousCNN(input_size=18, output_size=3)
    x = torch.randn(5, 18)

    logits = model(x)

    assert logits.shape == (5, 3)


def test_bilous_cnn_forward_accepts_structured_descriptor() -> None:
    model = BilousCNN(input_size=18, output_size=2)
    x = torch.randn(5, 6, 3)

    logits = model(x)

    assert logits.shape == (5, 2)


def test_bilous_cnn_rejects_non_three_channel_input_size() -> None:
    with pytest.raises(ValueError, match="input_size"):
        BilousCNN(input_size=17, output_size=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run from `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit`:

```bash
uv run pytest tests/test_bilous_cnn_model.py -v
```

Expected: FAIL because `graspkit.ml_module.cnn` does not exist.

- [ ] **Step 3: Create `cnn.py` with the PyTorch model**

Create `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/src/graspkit/ml_module/cnn.py`:

```python
"""Bilous-style convolutional neural network for CSF descriptors.

The model follows the CNN block used in Bilous et al. 2023 for GRASP2018
CSF-level selection, adapted from Keras softmax classification to GraspKit's
PyTorch multi-label BCE-with-logits training pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BilousCNN(nn.Module):
    """1D CNN for three-channel CSF descriptors.

    Input descriptors are expected to encode each orbital with three channels:
    occupation, intermediate coupling, and cumulative coupling. The public
    GraspKit pipeline passes flat arrays with length ``3 * n_orbitals``; this
    module reshapes them internally to the PyTorch Conv1d layout.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        channels: int = 3,
        conv1_filters: int = 96,
        conv2_filters: int = 16,
        dense_sizes: tuple[int, int, int] = (150, 120, 90),
    ) -> None:
        """Initialize the Bilous-style CNN.

        Args:
            input_size: Flat descriptor length, normally ``3 * n_orbitals``.
            output_size: Number of independent target levels.
            channels: Descriptor channels per orbital. Bilous and rCSFs use 3.
            conv1_filters: Number of filters for the kernel-size-3 convolution.
            conv2_filters: Number of filters for the kernel-size-1 convolution.
            dense_sizes: Dense head sizes corresponding to Bilous Fig. 1.

        Raises:
            ValueError: If ``input_size`` is not divisible by ``channels`` or if
                fewer than three orbitals are available for the kernel-size-3
                convolution.
        """
        super().__init__()
        if input_size % channels != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by channels ({channels})"
            )

        seq_length = input_size // channels
        if seq_length < 3:
            raise ValueError(
                "BilousCNN requires at least 3 orbital positions for Conv1d kernel_size=3"
            )

        self.input_size = input_size
        self.output_size = output_size
        self.channels = channels
        self.seq_length = seq_length
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.dense_sizes = dense_sizes

        self.conv = nn.Sequential(
            nn.Conv1d(channels, conv1_filters, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(conv1_filters, conv2_filters, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        flattened_size = conv2_filters * (seq_length - 2)
        dense1, dense2, dense3 = dense_sizes
        self.head = nn.Sequential(
            nn.Linear(flattened_size, dense1),
            nn.ReLU(),
            nn.Linear(dense1, dense2),
            nn.ReLU(),
            nn.Linear(dense2, dense3),
            nn.ReLU(),
            nn.Linear(dense3, output_size),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize layers with stable defaults for sparse CSF labels."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d | nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for flat or structured CSF descriptors."""
        if x.dim() == 2:
            if x.shape[1] != self.input_size:
                raise ValueError(
                    f"Expected flat input with {self.input_size} features, got {x.shape[1]}"
                )
            x = x.reshape(x.shape[0], self.seq_length, self.channels)
        elif x.dim() == 3:
            expected_shape = (self.seq_length, self.channels)
            actual_shape = (x.shape[1], x.shape[2])
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Expected structured input shape (*, {expected_shape[0]}, {expected_shape[1]}), "
                    f"got (*, {actual_shape[0]}, {actual_shape[1]})"
                )
        else:
            raise ValueError(f"Expected 2D or 3D input tensor, got {x.dim()}D")

        # Conv1d expects (batch, channels, sequence).
        x = x.transpose(1, 2)
        return self.head(self.conv(x))
```

- [ ] **Step 4: Run shape tests**

Run:

```bash
uv run pytest tests/test_bilous_cnn_model.py -v
```

Expected: PASS for the three shape/validation tests.

- [ ] **Step 5: Commit**

Run from `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit`:

```bash
git add src/graspkit/ml_module/cnn.py tests/test_bilous_cnn_model.py
git commit -m "ml: add Bilous-style CNN module"
```

## Task 2: Integrate CNN Into `ANNClassifier`

**Files:**
- Modify: `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/src/graspkit/ml_module/neural_network.py`
- Modify: `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/src/graspkit/ml_module/__init__.py`
- Test: `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/tests/test_bilous_cnn_model.py`

- [ ] **Step 1: Add failing wrapper tests**

Append to `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/tests/test_bilous_cnn_model.py`:

```python
import numpy as np

from graspkit.ml_module import ANNClassifier
from graspkit.ml_module.cnn import BilousCNN


def test_ann_classifier_builds_cnn_architecture() -> None:
    classifier = ANNClassifier(
        input_size=18,
        output_size=3,
        model_architecture="cnn",
        random_seed=7,
    )

    assert isinstance(classifier.model, BilousCNN)
    assert classifier.multi_label is True


def test_ann_classifier_cnn_predict_proba_shape() -> None:
    classifier = ANNClassifier(
        input_size=18,
        output_size=3,
        model_architecture="cnn",
        random_seed=7,
    )
    x = np.random.default_rng(3).normal(size=(4, 18)).astype(np.float32)

    proba = classifier.predict_proba(x)

    assert proba.shape == (4, 3)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_bilous_cnn_model.py::test_ann_classifier_builds_cnn_architecture tests/test_bilous_cnn_model.py::test_ann_classifier_cnn_predict_proba_shape -v
```

Expected: FAIL because `ANNClassifier` does not accept `"cnn"` yet.

- [ ] **Step 3: Modify `neural_network.py` imports and type hints**

In `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/src/graspkit/ml_module/neural_network.py`, add this local import near the other local module imports:

```python
from .cnn import BilousCNN
```

Change the `model_architecture` type hint in `ANNClassifier.__init__` from:

```python
model_architecture: Literal["standard", "tensornet"] = "standard",
```

to:

```python
model_architecture: Literal["standard", "tensornet", "cnn"] = "standard",
```

Update the docstring text for `model_architecture` to:

```python
model_architecture: 模型架构类型 ("standard" 标准全连接, "tensornet" 张量网络,
    或 "cnn" Bilous-style 1D CNN)
```

- [ ] **Step 4: Extend input validation for CNN**

In `ANNClassifier.__init__`, replace the TensorNet-only validation block:

```python
if model_architecture == "tensornet":
    if input_size % tensor_channels != 0:
        raise ValueError(
            f"对于 TensorNet 架构，input_size ({input_size}) "
            f"必须能被 tensor_channels ({tensor_channels}) 整除"
        )
```

with:

```python
if model_architecture in {"tensornet", "cnn"}:
    if input_size % tensor_channels != 0:
        raise ValueError(
            f"对于 {model_architecture} 架构，input_size ({input_size}) "
            f"必须能被 tensor_channels ({tensor_channels}) 整除"
        )
```

- [ ] **Step 5: Extend `_build_model`**

In `ANNClassifier._build_model`, insert the CNN branch between the TensorNet branch and the standard branch:

```python
if self.model_architecture == "tensornet":
    seq_length = self.input_size // self.tensor_channels
    model = TensorNet(
        input_shape=(seq_length, self.tensor_channels),
        hidden_dim=self.hidden_size,
        num_classes=self.output_size,
    ).to(self.device)
elif self.model_architecture == "cnn":
    model = BilousCNN(
        input_size=self.input_size,
        output_size=self.output_size,
        channels=self.tensor_channels,
    ).to(self.device)
else:
    model = nn.Sequential(
        nn.Linear(self.input_size, self.hidden_size),
        nn.BatchNorm1d(self.hidden_size),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(self.hidden_size, self.hidden_size // 2),
        nn.BatchNorm1d(self.hidden_size // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(self.hidden_size // 2, self.output_size),
    ).to(self.device)
    self._initialize_weights(model)
```

- [ ] **Step 6: Export `BilousCNN`**

In `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/src/graspkit/ml_module/__init__.py`, add:

```python
from .cnn import BilousCNN
```

and include `"BilousCNN"` in `__all__`.

- [ ] **Step 7: Run wrapper tests**

Run:

```bash
uv run pytest tests/test_bilous_cnn_model.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add src/graspkit/ml_module/neural_network.py src/graspkit/ml_module/__init__.py tests/test_bilous_cnn_model.py
git commit -m "ml: integrate CNN architecture into classifier"
```

## Task 3: Verify Training, Batch Prediction, And Checkpoint Compatibility

**Files:**
- Test: `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/tests/test_bilous_cnn_model.py`

- [ ] **Step 1: Add training and checkpoint tests**

Append to `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/tests/test_bilous_cnn_model.py`:

```python
def test_ann_classifier_cnn_fit_and_batch_predict() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(24, 18)).astype(np.float32)
    y = np.zeros((24, 2), dtype=np.float32)
    y[:8, 0] = 1.0
    y[8:14, 1] = 1.0

    classifier = ANNClassifier(
        input_size=18,
        output_size=2,
        model_architecture="cnn",
        random_seed=11,
    )
    history = classifier.fit(x, y, batch_size=8, max_epochs=2)
    proba = classifier.predict_proba_batch(x, batch_size=7)

    assert len(history["train_loss"]) == 2
    assert proba.shape == (24, 2)
    assert np.all(np.isfinite(proba))


def test_ann_classifier_cnn_save_and_load(tmp_path) -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=(10, 18)).astype(np.float32)

    classifier = ANNClassifier(
        input_size=18,
        output_size=2,
        model_architecture="cnn",
        random_seed=13,
    )
    before = classifier.predict_proba(x)
    model_path = tmp_path / "cnn_model.pt"

    classifier.save_model(str(model_path))
    restored = ANNClassifier.load_model(str(model_path))
    after = restored.predict_proba(x)

    assert restored.model_architecture == "cnn"
    assert isinstance(restored.model, BilousCNN)
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)
```

- [ ] **Step 2: Run checkpoint tests**

Run:

```bash
uv run pytest tests/test_bilous_cnn_model.py::test_ann_classifier_cnn_fit_and_batch_predict tests/test_bilous_cnn_model.py::test_ann_classifier_cnn_save_and_load -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

Run:

```bash
git add tests/test_bilous_cnn_model.py
git commit -m "test: cover CNN classifier training and checkpoints"
```

## Task 4: Add CNN Selection Policy To `ml_trainer.py`

**Files:**
- Modify: `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/src/graspkit/ml_module/ml_trainer.py`
- Test: `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/tests/test_bilous_cnn_model.py`

- [ ] **Step 1: Add architecture-selection tests**

Append to `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/tests/test_bilous_cnn_model.py`:

```python
import logging

from graspkit.ml_module.ml_trainer import _select_model_architecture


def test_select_model_architecture_keeps_standard_for_early_iterations() -> None:
    architecture = _select_model_architecture(
        cal_loop_num=2,
        current_loop_sample_count=100_000,
        accumulated_sample_count=150_000,
        positive_sample_count=10_000,
        logger=logging.getLogger("test"),
    )

    assert architecture == "standard"


def test_select_model_architecture_uses_cnn_for_large_late_csf_level_data() -> None:
    architecture = _select_model_architecture(
        cal_loop_num=4,
        current_loop_sample_count=100_000,
        accumulated_sample_count=150_000,
        positive_sample_count=10_000,
        logger=logging.getLogger("test"),
    )

    assert architecture == "cnn"


def test_select_model_architecture_keeps_standard_for_late_small_positive_data() -> None:
    architecture = _select_model_architecture(
        cal_loop_num=4,
        current_loop_sample_count=100_000,
        accumulated_sample_count=150_000,
        positive_sample_count=500,
        logger=logging.getLogger("test"),
    )

    assert architecture == "standard"
```

- [ ] **Step 2: Run architecture-selection tests to verify failure**

Run:

```bash
uv run pytest tests/test_bilous_cnn_model.py::test_select_model_architecture_uses_cnn_for_large_late_csf_level_data -v
```

Expected: FAIL because the existing function returns `"tensornet"` in this case.

- [ ] **Step 3: Update `_select_model_architecture`**

In `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/src/graspkit/ml_module/ml_trainer.py`, change the return type comment/docstring from architecture understood by `ANNClassifier` to include CNN.

Replace the final TensorNet branch:

```python
logger.info(
    "使用 tensornet 架构进行训练（本轮CSF数=%s, 累积样本数=%s）",
    current_loop_sample_count,
    accumulated_sample_count,
)
return "tensornet"
```

with:

```python
logger.info(
    "使用 cnn 架构进行训练（Bilous-style CSF序列卷积；本轮CSF数=%s, 累积样本数=%s）",
    current_loop_sample_count,
    accumulated_sample_count,
)
return "cnn"
```

Keep the existing early-loop and low-positive-count standard fallback unchanged. This makes CNN a conservative late-stage architecture, matching Bilous's motivation while avoiding unstable early random data.

- [ ] **Step 4: Run architecture-selection tests**

Run:

```bash
uv run pytest tests/test_bilous_cnn_model.py::test_select_model_architecture_keeps_standard_for_early_iterations tests/test_bilous_cnn_model.py::test_select_model_architecture_uses_cnn_for_large_late_csf_level_data tests/test_bilous_cnn_model.py::test_select_model_architecture_keeps_standard_for_late_small_positive_data -v
```

Expected: PASS.

- [ ] **Step 5: Run all CNN tests**

Run:

```bash
uv run pytest tests/test_bilous_cnn_model.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/graspkit/ml_module/ml_trainer.py tests/test_bilous_cnn_model.py
git commit -m "ml: use CNN for late-stage CSF selection"
```

## Task 5: Add An Explicit Config Hook If Current Config Supports One

**Files:**
- Inspect: `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit/src/graspkit/data_IO/`
- Modify only if `MLCalConfig` already has a suitable model setting field.

- [ ] **Step 1: Search for existing model-architecture config fields**

Run:

```bash
rg -n "model_architecture|architecture|model_type|model_params|use_regression_model" src/graspkit/data_IO src/graspkit/ml_module
```

Expected: Determine whether there is already a config field that can override automatic architecture selection.

- [ ] **Step 2: If a suitable field exists, add override logic**

If a field such as `config.model_params.model_architecture` exists, update `_select_model_architecture` call site in `train_model` before the automatic selection:

```python
configured_architecture = getattr(config.model_params, "model_architecture", None)
if configured_architecture in {"standard", "tensornet", "cnn"}:
    desired_architecture = configured_architecture
    logger.info("使用配置指定的模型架构: %s", desired_architecture)
else:
    desired_architecture = _select_model_architecture(
        cal_loop_num=config.cal_settings.cal_loop_num,
        current_loop_sample_count=current_loop_sample_count,
        accumulated_sample_count=len(X_train),
        positive_sample_count=positive_sample_count,
        logger=logger,
    )
```

If no such field exists, do not add a new Pydantic/config field in this task. Keep the change scoped to automatic policy and add a follow-up note for a separate config-schema change.

- [ ] **Step 3: Add or skip tests based on Step 2**

If an existing config hook was used, add a focused test for the helper or wrapper behavior using the current config object pattern. If no hook exists, record this in the implementation notes and do not create a brittle synthetic config test.

- [ ] **Step 4: Commit if changed**

Run:

```bash
git add src/graspkit/ml_module/ml_trainer.py tests/test_bilous_cnn_model.py
git commit -m "ml: allow configured CNN architecture"
```

Skip this commit if no config hook exists.

## Task 6: Verification And Physics-Facing Smoke Test

**Files:**
- No new files required.

- [ ] **Step 1: Run focused tests**

Run from `/Users/yiqin/Documents/PythonProjects/GraspKit-Workspace/GraspKit`:

```bash
uv run pytest tests/test_bilous_cnn_model.py -v
```

Expected: PASS.

- [ ] **Step 2: Run existing GraspKit tests**

Run:

```bash
uv run pytest tests/ -v
```

Expected: PASS. If unrelated legacy tests fail, record exact failures and confirm whether they predate the CNN change.

- [ ] **Step 3: Run lint on touched files**

Run:

```bash
uv run ruff check src/graspkit/ml_module/cnn.py src/graspkit/ml_module/neural_network.py src/graspkit/ml_module/ml_trainer.py tests/test_bilous_cnn_model.py
```

Expected: PASS.

- [ ] **Step 4: Run a small end-to-end ML smoke calculation**

Use an existing small GraspKit-Tools ML fixture or a reduced local calculation directory. The goal is not physical production accuracy; it is verifying that `train.py` can:

```text
load descriptors
train ANNClassifier(model_architecture="cnn")
evaluate labeled samples
predict unselected CSFs
save the .pt checkpoint
load the checkpoint in the next loop
```

Expected outputs:

```text
models/<conf>_<loop>.pt exists
logs mention "架构=cnn" or "使用 cnn 架构"
candidate_hybrid_scores.csv still writes when hybrid mode is enabled
final_sampled_idxs.npy is produced when normal train.py conditions are satisfied
```

- [ ] **Step 5: Record physics-facing ablation plan**

For Ni I, run or schedule these four comparable ML-selection branches with the same fixed VV backbone orbitals and the same CSF pool:

```text
1. standard ANN single-J selection
2. BilousCNN single-J selection
3. standard ANN + cross-J occupation/coupling-channel union
4. BilousCNN + cross-J occupation/coupling-channel union
```

Compare:

```text
J=4,3,2 relative energies
fine-structure splitting
jj2lsj labels
dominant CSF weights
missed-important rate for high c_i^2 CSFs
cross-J coverage of occupation/coupling-channel keys
```

- [ ] **Step 6: Final commit if verification changed docs or notes**

If verification notes are added to repository docs, commit them:

```bash
git add <verification-note-path>
git commit -m "docs: record CNN selection verification"
```

## Self-Review

- Spec coverage: The plan creates `cnn.py`, matches Bilous 2023 CNN layers, adapts output to GraspKit multi-label logits, integrates with `ANNClassifier`, adds exports, adds architecture selection, covers save/load and batch prediction, and includes physics-facing validation.
- Placeholder scan: No implementation step relies on undefined "TBD" work. The optional config hook is deliberately conditional because the current config schema was not fully inspected in this planning pass.
- Type consistency: `BilousCNN(input_size, output_size, channels=3)` is used consistently by tests and `ANNClassifier._build_model`. `model_architecture="cnn"` is consistently used across tests, trainer, save/load metadata, and wrapper construction.

