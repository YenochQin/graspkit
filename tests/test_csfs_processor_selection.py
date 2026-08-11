import numpy as np
import pytest

from graspkit.CSFs_processor.selection import (
    CSFs_sort_by_mix_coefficient,
    radom_choose_csfs,
    random_choose_csfs,
    rmix_cumulative_selected_row_idxs,
    sort_csfs_by_mix_coefficient,
)


def test_sort_csfs_by_mix_coefficient_names_coefficient_cutoff_explicitly() -> None:
    result = sort_csfs_by_mix_coefficient(
        [["csf0"], ["csf1"], ["csf2"]],
        np.array([0.1, 0.5, -0.2]),
        ci_coefficient_cutoff=0.3,
    )

    assert result == [["csf1"]]
from graspkit.grasp_data_extractor.rmix_data_processor import RmixCiSquaredData


def test_csfs_sort_by_mix_coefficient_accepts_1d_single_asf_input() -> None:
    csfs_block = [["csf0"], ["csf1"], ["csf2"]]
    coefficients = np.array([0.1, 0.5, -0.2])

    result = CSFs_sort_by_mix_coefficient(csfs_block, coefficients)

    assert result == [["csf1"], ["csf2"], ["csf0"]]


def test_csfs_sort_by_mix_coefficient_1d_applies_threshold() -> None:
    csfs_block = [["csf0"], ["csf1"], ["csf2"]]
    coefficients = np.array([0.1, 0.5, -0.2])

    result = CSFs_sort_by_mix_coefficient(csfs_block, coefficients, threshold=0.3)

    assert result == [["csf1"]]


def test_csfs_sort_by_mix_coefficient_rejects_empty_block() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        CSFs_sort_by_mix_coefficient([], np.array([0.1]))


def test_csfs_sort_by_mix_coefficient_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="长度必须"):
        CSFs_sort_by_mix_coefficient([["csf0"], ["csf1"]], np.array([0.1, 0.2, 0.3]))


def test_radom_choose_csfs_defaults_to_no_preselected_indices() -> None:
    block_csfs = [["csf0"], ["csf1"], ["csf2"], ["csf3"]]

    chosen_csfs, chosen_idxs, unselected_idxs = radom_choose_csfs(
        block_csfs,
        method="quality",
        ratio_or_quality=2,
    )

    assert len(chosen_csfs) == 2
    assert chosen_idxs.shape == (2,)
    assert set(chosen_idxs.tolist()) | set(unselected_idxs.tolist()) == {0, 1, 2, 3}
    assert set(chosen_idxs.tolist()).isdisjoint(set(unselected_idxs.tolist()))


def test_radom_choose_csfs_ratio_method_computes_target_from_block_size() -> None:
    block_csfs = [["csf0"], ["csf1"], ["csf2"], ["csf3"]]

    chosen_csfs, chosen_idxs, _unselected = radom_choose_csfs(
        block_csfs,
        method="ratio",
        ratio_or_quality=0.5,
    )

    assert len(chosen_csfs) == 2
    assert chosen_idxs.shape == (2,)


def test_radom_choose_csfs_tops_up_around_preselected_indices() -> None:
    block_csfs = [["csf0"], ["csf1"], ["csf2"], ["csf3"]]

    chosen_csfs, chosen_idxs, unselected_idxs = radom_choose_csfs(
        block_csfs,
        method="quality",
        ratio_or_quality=3,
        selected_csfs_idxs=[0],
    )

    assert 0 in chosen_idxs.tolist()
    assert len(chosen_csfs) == 3
    assert set(chosen_idxs.tolist()) | set(unselected_idxs.tolist()) == {0, 1, 2, 3}


def test_radom_choose_csfs_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="method"):
        radom_choose_csfs(
            [["csf0"]],
            method="bogus",  # type: ignore[arg-type]
            ratio_or_quality=1,
        )


def test_random_choose_csfs_accepts_reproducible_generator() -> None:
    block_csfs = [["csf0"], ["csf1"], ["csf2"], ["csf3"]]

    first = random_choose_csfs(
        block_csfs,
        method="quality",
        ratio_or_quality=2,
        rng=np.random.default_rng(42),
    )
    second = random_choose_csfs(
        block_csfs,
        method="quality",
        ratio_or_quality=2,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(first[1], second[1])


def test_radom_choose_csfs_warns_as_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="random_choose_csfs"):
        radom_choose_csfs(
            [["csf0"]],
            method="quality",
            ratio_or_quality=1,
        )


def test_rmix_cumulative_selected_row_idxs_offsets_across_blocks() -> None:
    data = RmixCiSquaredData(
        block_indices=[0, 1],
        selected_asfs=[[0], [0]],
        ci_squared_list=[
            np.array([[0.9, 0.06, 0.04]], dtype=np.float64),
            np.array([[0.9, 0.06, 0.04]], dtype=np.float64),
        ],
    )

    result = rmix_cumulative_selected_row_idxs(data, cumulative_threshold=0.9)

    # Each block's first CSF alone reaches the 0.9 cumulative threshold;
    # block 1's local idx 0 is offset by block 0's csf_count (3) -> global 3.
    np.testing.assert_array_equal(result, np.array([0, 3], dtype=np.int64))


def test_rmix_cumulative_selected_row_idxs_preserves_ci_importance_order() -> None:
    data = RmixCiSquaredData(
        block_indices=[0],
        selected_asfs=[[0]],
        ci_squared_list=[
            np.array([[0.1, 0.7, 0.2]], dtype=np.float64),
        ],
    )

    result = rmix_cumulative_selected_row_idxs(data, cumulative_threshold=0.95)

    # Ranked by descending CI-square: idx1 (0.7), idx2 (0.2), idx0 (0.1).
    np.testing.assert_array_equal(result, np.array([1, 2, 0], dtype=np.int64))


def test_rmix_cumulative_selected_row_idxs_rejects_all_zero_contribution() -> None:
    data = RmixCiSquaredData(
        block_indices=[0],
        selected_asfs=[[0]],
        ci_squared_list=[np.array([[0.0, 0.0]], dtype=np.float64)],
    )

    with pytest.raises(ValueError, match="没有选出任何 CSF"):
        rmix_cumulative_selected_row_idxs(data, cumulative_threshold=0.9)
