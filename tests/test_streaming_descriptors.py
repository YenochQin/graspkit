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


class ForbidFullCollectLazyFrame:
    def __init__(self, lazy_frame: pl.LazyFrame) -> None:
        self._lazy_frame = lazy_frame
        self.with_row_index_calls = 0

    def collect(self) -> pl.DataFrame:
        raise AssertionError("full collect forbidden")

    def collect_schema(self) -> pl.Schema:
        return self._lazy_frame.collect_schema()

    def with_row_index(self, *args: object, **kwargs: object) -> pl.LazyFrame:
        self.with_row_index_calls += 1
        return self._lazy_frame.with_row_index(*args, **kwargs)


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


def test_build_labeled_training_array_from_lazy_descriptors_materializes_only_accumulated_rows() -> None:
    from graspkit.ml_module.ml_initializer import (
        build_labeled_training_array_from_lazy_descriptors,
    )

    frame = pl.DataFrame(
        {
            "col_0": np.arange(6, dtype=np.float32),
            "col_1": np.arange(6, dtype=np.float32) + 10,
            "col_2": np.arange(6, dtype=np.float32) + 20,
        }
    ).lazy()
    accumulated_idxs = np.array([4, 1, 5], dtype=np.int64)
    accumulated_ci_squared = np.array(
        [
            [0.2, 0.01, 0.7],
            [0.0, 0.3, 0.01],
        ],
        dtype=np.float64,
    )

    result = build_labeled_training_array_from_lazy_descriptors(
        frame,
        accumulated_idxs=accumulated_idxs,
        accumulated_ci_squared=accumulated_ci_squared,
        cutoff_value=0.1,
        batch_size=2,
    )

    expected = np.array(
        [
            [4, 14, 24, 1, 0],
            [1, 11, 21, 0, 1],
            [5, 15, 25, 1, 0],
        ],
        dtype=np.float32,
    )
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)


def test_build_labeled_training_array_from_lazy_descriptors_returns_empty_2d_array() -> None:
    from graspkit.ml_module.ml_initializer import (
        build_labeled_training_array_from_lazy_descriptors,
    )

    frame = pl.DataFrame(
        {
            "col_0": np.arange(6, dtype=np.float32),
            "col_1": np.arange(6, dtype=np.float32) + 10,
            "col_2": np.arange(6, dtype=np.float32) + 20,
        }
    ).lazy()

    result = build_labeled_training_array_from_lazy_descriptors(
        frame,
        accumulated_idxs=np.array([], dtype=np.int64),
        accumulated_ci_squared=np.empty((2, 0), dtype=np.float64),
        cutoff_value=0.1,
    )

    assert result.shape == (0, 5)
    assert result.dtype == np.float32


def test_build_labeled_training_array_from_lazy_descriptors_does_not_full_collect_raw_descriptors() -> None:
    from graspkit.ml_module.ml_initializer import (
        build_labeled_training_array_from_lazy_descriptors,
    )

    frame = ForbidFullCollectLazyFrame(
        pl.DataFrame(
            {
                "col_0": np.arange(6, dtype=np.float32),
                "col_1": np.arange(6, dtype=np.float32) + 10,
                "col_2": np.arange(6, dtype=np.float32) + 20,
            }
        ).lazy()
    )

    result = build_labeled_training_array_from_lazy_descriptors(
        frame,  # type: ignore[arg-type]
        accumulated_idxs=np.array([4, 1, 5], dtype=np.int64),
        accumulated_ci_squared=np.array(
            [
                [0.2, 0.01, 0.7],
                [0.0, 0.3, 0.01],
            ],
            dtype=np.float64,
        ),
        cutoff_value=0.1,
        batch_size=2,
    )

    assert result.shape == (3, 5)


def test_validate_csf_desc_coverage_streaming_adds_first_rows_covering_missing_orbitals() -> None:
    from graspkit.ml_module.streaming_descriptors import (
        validate_csf_desc_coverage_streaming,
    )

    frame = pl.DataFrame(
        {
            "col_0": [1, 0, 0, 0],
            "col_1": [0, 0, 0, 0],
            "col_2": [0, 0, 0, 0],
            "col_3": [0, 1, 0, 0],
            "col_4": [0, 0, 0, 0],
            "col_5": [0, 0, 0, 0],
            "col_6": [0, 0, 2, 0],
            "col_7": [0, 0, 0, 0],
            "col_8": [0, 0, 0, 0],
        }
    ).lazy()

    final = validate_csf_desc_coverage_streaming(
        final_sampled_idxs=np.array([0], dtype=np.int64),
        raw_csfs_descriptors=frame,
        total_csfs_count=4,
        batch_size=2,
    )

    np.testing.assert_array_equal(final, np.array([0, 1, 2], dtype=np.int64))


def test_validate_csf_desc_coverage_streaming_returns_original_selection_when_selected_covered() -> None:
    from graspkit.ml_module.streaming_descriptors import (
        validate_csf_desc_coverage_streaming,
    )

    frame = ForbidFullCollectLazyFrame(
        pl.DataFrame(
            {
                "col_0": [1, 0, 0, 0, 0],
                "col_1": [0, 0, 0, 0, 0],
                "col_2": [0, 0, 0, 0, 0],
                "col_3": [0, 1, 0, 0, 0],
                "col_4": [0, 0, 0, 0, 0],
                "col_5": [0, 0, 0, 0, 0],
            }
        ).lazy()
    )

    final = validate_csf_desc_coverage_streaming(
        final_sampled_idxs=np.array([1, 0], dtype=np.int64),
        raw_csfs_descriptors=frame,  # type: ignore[arg-type]
        total_csfs_count=5,
        batch_size=2,
    )

    np.testing.assert_array_equal(final, np.array([1, 0], dtype=np.int64))
    assert frame.with_row_index_calls == 1


def test_validate_csf_desc_coverage_streaming_rejects_empty_selection() -> None:
    from graspkit.ml_module.streaming_descriptors import (
        validate_csf_desc_coverage_streaming,
    )

    frame = pl.DataFrame(
        {
            "col_0": [1],
            "col_1": [0],
            "col_2": [0],
        }
    ).lazy()

    with pytest.raises(RuntimeError, match="final_sampled_idxs"):
        validate_csf_desc_coverage_streaming(
            final_sampled_idxs=np.array([], dtype=np.int64),
            raw_csfs_descriptors=frame,
            total_csfs_count=1,
        )


def test_validate_csf_desc_coverage_streaming_scans_candidates_across_bounded_chunks() -> None:
    from graspkit.ml_module.streaming_descriptors import (
        validate_csf_desc_coverage_streaming,
    )

    frame = ForbidFullCollectLazyFrame(
        pl.DataFrame(
            {
                "col_0": [1, 0, 0, 0, 0, 0, 0, 0],
                "col_1": [0, 0, 0, 0, 0, 0, 0, 0],
                "col_2": [0, 0, 0, 0, 0, 0, 0, 0],
                "col_3": [0, 0, 0, 0, 0, 1, 1, 0],
                "col_4": [0, 0, 0, 0, 0, 0, 0, 0],
                "col_5": [0, 0, 0, 0, 0, 0, 0, 0],
            }
        ).lazy()
    )

    final = validate_csf_desc_coverage_streaming(
        final_sampled_idxs=np.array([0], dtype=np.int64),
        raw_csfs_descriptors=frame,  # type: ignore[arg-type]
        total_csfs_count=8,
        batch_size=2,
    )

    np.testing.assert_array_equal(final, np.array([0, 5], dtype=np.int64))
