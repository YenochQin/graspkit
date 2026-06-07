from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graspkit.ml_module.streaming_descriptors import (
    DescriptorShape,
    iter_indexed_descriptor_batches,
    validate_cnn_descriptor_shape,
)


def test_validate_cnn_descriptor_shape_accepts_three_channel_descriptor() -> None:
    shape = validate_cnn_descriptor_shape(n_features=168, channels=3)

    assert shape == DescriptorShape(n_features=168, channels=3, sequence_length=56)


def test_validate_cnn_descriptor_shape_rejects_non_divisible_width() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        validate_cnn_descriptor_shape(n_features=167, channels=3)


def test_validate_cnn_descriptor_shape_rejects_too_short_sequence() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        validate_cnn_descriptor_shape(n_features=6, channels=3)


def test_iter_indexed_descriptor_batches_returns_requested_rows_in_input_order() -> None:
    frame = pl.DataFrame(
        {
            "col_0": np.arange(10, dtype=np.float32),
            "col_1": np.arange(10, dtype=np.float32) + 100,
            "col_2": np.arange(10, dtype=np.float32) + 200,
        }
    ).lazy()

    batches = list(
        iter_indexed_descriptor_batches(
            frame,
            indices=np.array([7, 2, 7, 0], dtype=np.int64),
            batch_size=2,
        )
    )

    assert [batch_indices.tolist() for batch_indices, _ in batches] == [[7, 2], [7, 0]]
    np.testing.assert_array_equal(
        batches[0][1],
        np.array([[7, 107, 207], [2, 102, 202]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        batches[1][1],
        np.array([[7, 107, 207], [0, 100, 200]], dtype=np.float32),
    )


def test_iter_indexed_descriptor_batches_preserves_duplicate_rows_in_same_batch() -> None:
    frame = pl.DataFrame(
        {
            "col_0": np.arange(10, dtype=np.float32),
            "col_1": np.arange(10, dtype=np.float32) + 100,
            "col_2": np.arange(10, dtype=np.float32) + 200,
        }
    ).lazy()

    batches = list(
        iter_indexed_descriptor_batches(
            frame,
            indices=np.array([7, 7, 2], dtype=np.int64),
            batch_size=3,
        )
    )

    assert [batch_indices.tolist() for batch_indices, _ in batches] == [[7, 7, 2]]
    np.testing.assert_array_equal(
        batches[0][1],
        np.array(
            [[7, 107, 207], [7, 107, 207], [2, 102, 202]],
            dtype=np.float32,
        ),
    )


def test_iter_indexed_descriptor_batches_rejects_missing_indices() -> None:
    frame = pl.DataFrame(
        {
            "col_0": np.arange(3, dtype=np.float32),
            "col_1": np.arange(3, dtype=np.float32) + 100,
        }
    ).lazy()

    with pytest.raises(IndexError, match="4"):
        list(
            iter_indexed_descriptor_batches(
                frame,
                indices=np.array([2, 4], dtype=np.int64),
                batch_size=2,
            )
        )


def test_iter_indexed_descriptor_batches_rejects_non_positive_batch_size() -> None:
    frame = pl.DataFrame({"col_0": np.arange(3, dtype=np.float32)}).lazy()

    with pytest.raises(ValueError, match="batch_size"):
        list(
            iter_indexed_descriptor_batches(
                frame,
                indices=np.array([0, 1], dtype=np.int64),
                batch_size=0,
            )
        )
