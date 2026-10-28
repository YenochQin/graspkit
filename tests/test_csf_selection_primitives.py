"""Stable CI/CSF primitives used by both iterative and final selection."""

from itertools import islice
import subprocess
import sys

import numpy as np
import pytest

from graspkit.CSFs_processor import (
    complete_orbital_coverage,
    count_orbital_coverage,
    iter_unselected_csf_indices,
    project_local_idxs_to_full_pool_idxs,
    select_ci_cumulative_local_idxs,
    select_csf_indices_by_cumulative_weight,
    validate_selection_idxs,
)
from graspkit.grasp_data_extractor.rmix_data_processor import (
    filter_sorted_ci_scores_by_cumulative,
)


def test_unit_prefix_includes_crossing_item_and_maps_global_indices() -> None:
    result = select_csf_indices_by_cumulative_weight(
        np.array([0.6, 0.3, 0.1]),
        csf_indices=np.array([20, 10, 30]),
        cumulative_threshold=0.8,
    )
    assert result.indices.tolist() == [10, 20]
    assert result.per_asf_indices[0].tolist() == [20, 10]
    assert result.retained_weight == pytest.approx([0.9])
    assert result.discarded_weight == pytest.approx([0.1])


def test_square_matrix_axes_are_not_guessed() -> None:
    result = select_csf_indices_by_cumulative_weight(
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        csf_indices=np.array([10, 20]),
        cumulative_threshold=0.8,
    )
    assert [x.tolist() for x in result.per_asf_indices] == [[10], [20]]
    assert result.indices.tolist() == [10, 20]
    assert result.retained_weight == pytest.approx([1, 1])


def test_ties_and_output_order_are_explicit() -> None:
    tied = select_csf_indices_by_cumulative_weight(
        np.array([0.5, 0.5]), csf_indices=np.array([3, 9]), cumulative_threshold=0.5
    )
    assert tied.indices.tolist() == [9]
    ordered = select_csf_indices_by_cumulative_weight(
        np.array([0.1, 0.7, 0.2]), cumulative_threshold=0.95, index_order="importance"
    )
    assert ordered.indices.tolist() == [1, 2, 0]


def test_target_one_keeps_nonzero_rows_without_overflowing_prefix() -> None:
    result = select_csf_indices_by_cumulative_weight(
        np.array([0.3, 0.7000000001, 0]), cumulative_threshold=1
    )
    assert result.indices.tolist() == [0, 1]
    assert result.discarded_weight == pytest.approx([0], abs=1e-12)
    assert filter_sorted_ci_scores_by_cumulative(
        np.array([0.7, 0.3, 0]), 1
    ).tolist() == [0, 1]
    relative = select_csf_indices_by_cumulative_weight(
        np.array([1e308, 1e-300]), cumulative_threshold=1, normalization="relative"
    )
    assert relative.indices.tolist() == [0, 1]


@pytest.mark.parametrize(
    "weights", [[0, 0], [0.1, 0.1], [float("nan"), 1], [-0.1, 1.1], [float("inf"), 0]]
)
def test_unit_mode_rejects_invalid_or_incomplete_weights(weights: list[float]) -> None:
    with pytest.raises(ValueError):
        select_csf_indices_by_cumulative_weight(
            np.array(weights), cumulative_threshold=0.999999
        )


def test_relative_mode_preserves_postprocessing_semantics() -> None:
    result = select_csf_indices_by_cumulative_weight(
        np.array([[8.0, 2.0, 0.0], [0.0, 0.0, 0.0]]),
        cumulative_threshold=0.8,
        normalization="relative",
    )
    assert result.indices.tolist() == [0]
    assert result.totals.tolist() == [10, 0]
    assert result.per_asf_indices[1].size == 0
    assert result.retained_weight == pytest.approx([0.8, 0])
    assert result.discarded_weight == pytest.approx([0.2, 0])


def test_index_shape_and_duplicate_checks_use_shared_validator() -> None:
    with pytest.raises(ValueError, match="重复"):
        select_csf_indices_by_cumulative_weight(
            np.array([0.5, 0.5]), csf_indices=np.array([1, 1]), cumulative_threshold=0.9
        )
    with pytest.raises(ValueError, match="shape"):
        select_csf_indices_by_cumulative_weight(
            np.ones((3, 2)), csf_indices=np.arange(3), cumulative_threshold=0.9
        )
    with pytest.raises(ValueError):
        validate_selection_idxs(np.array([2**64 - 1], dtype=np.uint64))
    for malformed in (
        np.array(1.0),
        np.empty((0, 2)),
        np.ones((1, 1, 1)),
        np.array([1 + 1j]),
    ):
        with pytest.raises(ValueError, match="shape"):
            select_csf_indices_by_cumulative_weight(malformed, cumulative_threshold=0.9)


def test_block_offsets_and_actual_asf_rows() -> None:
    blocks = [np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([[0.1, 0.9]])]
    result = select_ci_cumulative_local_idxs(
        blocks, asf_row_indices=[[1], [0]], cumulative_threshold=0.8
    )
    assert result.tolist() == [1, 3]
    assert (
        select_ci_cumulative_local_idxs(
            blocks, asf_row_indices=[[]], cumulative_threshold=0.8
        ).size
        == 0
    )
    with pytest.raises(ValueError):
        select_ci_cumulative_local_idxs(
            blocks, asf_row_indices=[[2]], cumulative_threshold=0.8
        )
    with pytest.raises(ValueError):
        select_ci_cumulative_local_idxs(
            blocks, asf_row_indices=[[0.5]], cumulative_threshold=0.8
        )  # type: ignore[list-item]


def test_projection_preserves_local_request_order_and_checks_both_index_spaces() -> (
    None
):
    source = np.array([10, 20, 30, 40])
    assert project_local_idxs_to_full_pool_idxs(source, np.array([2, 0])).tolist() == [
        30,
        10,
    ]
    for local in (np.array([-1]), np.array([4]), np.array([0.5]), np.array([1, 1])):
        with pytest.raises(ValueError):
            project_local_idxs_to_full_pool_idxs(source, local)
    with pytest.raises(ValueError):
        project_local_idxs_to_full_pool_idxs(source, np.array([0]), pool_row_count=20)


def test_complement_scans_large_pools_in_bounded_chunks() -> None:
    chunks = list(islice(iter_unselected_csf_indices(10**9, np.array([1, 5, 8]), 4), 3))
    assert [x.tolist() for x in chunks] == [[0, 2, 3], [4, 6, 7], [9, 10, 11]]
    assert list(iter_unselected_csf_indices(2, np.array([0, 1]), 2)) == []


def test_coverage_reuses_already_selected_orbitals_and_adds_in_source_order() -> None:
    masks = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=bool)
    calls: list[list[int]] = []

    def reader(indices: np.ndarray) -> np.ndarray:
        assert len(indices) <= 2
        calls.append(indices.tolist())
        return masks[indices]

    result = complete_orbital_coverage(
        np.array([1, 0]),
        read_occupations=reader,
        total_count=4,
        orbital_count=3,
        batch_size=2,
    )
    assert result.selected_indices.tolist() == [0, 1, 2]
    assert result.added_indices.tolist() == [2]
    assert result.missing_before.tolist() == [2]
    assert result.missing_after.size == 0
    assert result.occupation_counts.tolist() == [1, 2, 1]
    assert calls == [[1, 0], [2, 3]]
    complete = complete_orbital_coverage(
        np.array([2, 0]),
        read_occupations=reader,
        total_count=4,
        orbital_count=3,
        batch_size=2,
    )
    assert complete.selected_indices.tolist() == [2, 0]  # unchanged without additions
    assert complete.added_indices.size == 0


def test_empty_selection_and_uncoverable_orbitals_are_reported() -> None:
    masks = np.array([[1, 0], [0, 0]], dtype=bool)
    result = complete_orbital_coverage(
        np.empty(0, dtype=np.int64),
        read_occupations=lambda indices: masks[indices],
        total_count=2,
        orbital_count=2,
        batch_size=1,
    )
    assert result.added_indices.tolist() == [0]
    assert result.missing_after.tolist() == [1]
    assert result.occupation_counts.tolist() == [1, 0]


def test_coverage_rejects_malformed_reader_output() -> None:
    for result in (np.ones((2, 2), dtype=bool), np.ones((1, 2), dtype=float)):
        with pytest.raises(ValueError, match="boolean"):
            count_orbital_coverage(
                np.array([0]),
                read_occupations=lambda indices: result,
                total_count=2,
                orbital_count=2,
            )


def test_kit_selection_does_not_import_ml_packages() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import graspkit.CSFs_processor; assert 'torch' not in sys.modules; assert 'sklearn' not in sys.modules; assert not any(n.startswith('ml_CSFs_selection_scripts') for n in sys.modules)",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
