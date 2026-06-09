from pathlib import Path

import numpy as np
import pytest

from graspkit.grasp_data_extractor import rmix_data_processor
from graspkit.grasp_data_extractor.rmix_data_processor import (
    RmixCiSquaredData,
    RmixCsfIndexSelection,
    aggregate_ci_squared,
    ci_squared,
    filter_ci_scores_by_threshold,
    filter_sorted_ci_scores_by_cumulative,
    load_rmix_ci_squared,
    sort_ci_scores,
)
from graspkit.CSFs_processor.CSFs_choosing import (
    CSFs_sort_by_mix_coefficient,
    batch_asfs_mix_square_above_threshold,
    batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum,
    single_asf_mix_square_above_threshold,
)
from graspkit.utils.data_modules import MixCoefficientBlock, MixCoefficientData


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


def test_ci_squared_returns_float64_square_for_1d_coefficients() -> None:
    coefficients = np.array([0.5, -0.25, 0.0])

    result = ci_squared(coefficients)

    np.testing.assert_allclose(result, np.array([0.25, 0.0625, 0.0]))
    assert result.dtype == np.float64


def test_ci_squared_returns_float64_square_for_2d_coefficients() -> None:
    coefficients = np.array([[0.5, -0.25], [0.1, -0.2]])

    result = ci_squared(coefficients)

    expected = np.array([[0.25, 0.0625], [0.01, 0.04]])
    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_ci_squared_rejects_empty_coefficients() -> None:
    with pytest.raises(ValueError, match="coefficients cannot be empty"):
        ci_squared(np.array([]))


def test_ci_squared_rejects_more_than_two_dimensions() -> None:
    with pytest.raises(ValueError, match="1D or 2D"):
        ci_squared(np.zeros((1, 1, 1)))


def test_aggregate_ci_squared_sum_max_and_mean_for_2d_input() -> None:
    values = np.array([[0.25, 0.0625, 0.0], [0.01, 0.04, 0.09]])

    np.testing.assert_allclose(
        aggregate_ci_squared(values, method="sum"),
        np.array([0.26, 0.1025, 0.09]),
    )
    np.testing.assert_allclose(
        aggregate_ci_squared(values, method="max"),
        np.array([0.25, 0.0625, 0.09]),
    )
    np.testing.assert_allclose(
        aggregate_ci_squared(values, method="mean"),
        np.array([0.13, 0.05125, 0.045]),
    )


def test_aggregate_ci_squared_returns_1d_input_unchanged_as_float64() -> None:
    values = np.array([0.25, 0.0625, 0.0])

    result = aggregate_ci_squared(values)

    np.testing.assert_allclose(result, np.array([0.25, 0.0625, 0.0]))
    assert result.dtype == np.float64


def test_aggregate_ci_squared_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unsupported aggregation method"):
        aggregate_ci_squared(np.array([[0.25]]), method="median")  # type: ignore[arg-type]


def test_sort_ci_scores_descending_returns_original_indices_and_scores() -> None:
    scores = np.array([0.04, 0.25, 0.01])

    indices, sorted_scores = sort_ci_scores(scores)

    np.testing.assert_array_equal(indices, np.array([1, 0, 2]))
    np.testing.assert_allclose(sorted_scores, np.array([0.25, 0.04, 0.01]))


def test_sort_ci_scores_ascending_returns_original_indices_and_scores() -> None:
    scores = np.array([0.04, 0.25, 0.01])

    indices, sorted_scores = sort_ci_scores(scores, descending=False)

    np.testing.assert_array_equal(indices, np.array([2, 0, 1]))
    np.testing.assert_allclose(sorted_scores, np.array([0.01, 0.04, 0.25]))


def test_filter_ci_scores_by_threshold_uses_strict_cutoff_by_default() -> None:
    scores = np.array([0.04, 0.25, 0.01])

    result = filter_ci_scores_by_threshold(scores, threshold=0.04)

    np.testing.assert_array_equal(result, np.array([1]))


def test_filter_ci_scores_by_threshold_can_include_equal_values() -> None:
    scores = np.array([0.04, 0.25, 0.01])

    result = filter_ci_scores_by_threshold(scores, threshold=0.04, inclusive=True)

    np.testing.assert_array_equal(result, np.array([0, 1]))


def test_filter_ci_scores_by_threshold_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be non-negative"):
        filter_ci_scores_by_threshold(np.array([0.04]), threshold=-0.1)


def test_filter_sorted_ci_scores_by_cumulative_keeps_minimum_prefix() -> None:
    sorted_scores = np.array([0.5, 0.3, 0.2])

    result = filter_sorted_ci_scores_by_cumulative(
        sorted_scores,
        cumulative_threshold=0.75,
    )

    np.testing.assert_array_equal(result, np.array([0, 1]))


def test_filter_sorted_ci_scores_by_cumulative_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="cumulative_threshold must be"):
        filter_sorted_ci_scores_by_cumulative(
            np.array([0.5, 0.3, 0.2]),
            cumulative_threshold=1.5,
        )


def test_filter_sorted_ci_scores_by_cumulative_returns_empty_for_zero_total() -> None:
    result = filter_sorted_ci_scores_by_cumulative(
        np.array([0.0, 0.0]),
        cumulative_threshold=0.5,
    )

    np.testing.assert_array_equal(result, np.array([], dtype=np.int64))


def test_load_rmix_ci_squared_returns_selected_asf_square_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_paths: list[Path] = []

    class DummyLoader:
        def __init__(self, file_path: str | Path) -> None:
            loaded_paths.append(Path(file_path))

        def load(self) -> MixCoefficientData:
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

    monkeypatch.setattr(rmix_data_processor, "MixCoefLoader", DummyLoader)

    result = load_rmix_ci_squared(
        "/tmp/example.m",
        select_asfs=[[1], [0]],
    )

    assert loaded_paths == [Path("/tmp/example.m")]
    assert isinstance(result, RmixCiSquaredData)
    assert result.block_indices == [0, 1]
    assert result.selected_asfs == [[1], [0]]
    np.testing.assert_allclose(result.ci_squared_list[0], np.array([[0.0, 0.16, 0.01]]))
    np.testing.assert_allclose(result.ci_squared_list[1], np.array([[0.09, 0.16]]))


def test_rmix_ci_squared_data_methods_return_csf_index_selections() -> None:
    data = RmixCiSquaredData(
        block_indices=[0],
        selected_asfs=[[0, 1]],
        ci_squared_list=[
            np.array(
                [
                    [0.25, 0.01, 0.04],
                    [0.0, 0.16, 0.01],
                ]
            )
        ],
    )

    sorted_result = data.sort_ci_scores()
    assert isinstance(sorted_result, RmixCsfIndexSelection)
    assert sorted_result.block_indices == [0]
    assert sorted_result.selected_asfs == [[0, 1]]
    assert sorted_result.csf_indices_list == [[[0, 2, 1], [1, 2, 0]]]

    threshold_result = data.filter_ci_scores_by_threshold(0.03)
    assert threshold_result.csf_indices_list == [[[0, 2], [1]]]

    cumulative_result = data.filter_sorted_ci_scores_by_cumulative(0.9)
    assert cumulative_result.csf_indices_list == [[[0, 2], [1]]]


def test_rmix_processor_api_is_exported_from_grasp_data_extractor_package() -> None:
    from graspkit.grasp_data_extractor import (
        RmixCiSquaredData,
        RmixCsfIndexSelection,
        aggregate_ci_squared,
        ci_squared,
        filter_ci_scores_by_threshold,
        filter_sorted_ci_scores_by_cumulative,
        load_rmix_ci_squared,
        sort_ci_scores,
    )

    assert RmixCiSquaredData.__name__ == "RmixCiSquaredData"
    assert RmixCsfIndexSelection.__name__ == "RmixCsfIndexSelection"
    assert callable(aggregate_ci_squared)
    assert callable(ci_squared)
    assert callable(filter_ci_scores_by_threshold)
    assert callable(filter_sorted_ci_scores_by_cumulative)
    assert callable(load_rmix_ci_squared)
    assert callable(sort_ci_scores)


def test_removed_rmix_analysis_api_is_not_exported() -> None:
    import graspkit.grasp_data_extractor as extractor

    assert not hasattr(extractor, "RmixAsfSelection")
    assert not hasattr(extractor, "RmixBlockSelection")
    assert not hasattr(extractor, "analyze_rmix_file")
    assert not hasattr(extractor, "select_block_ci_scores")


def test_legacy_single_asf_threshold_keeps_existing_tuple_index_shape() -> None:
    coefficients = np.array([0.5, -0.2, 0.1])

    result = single_asf_mix_square_above_threshold(coefficients, threshold=0.04)

    assert result == [(0,), (1,)]


def test_legacy_batch_threshold_returns_unique_indices_per_block() -> None:
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

    result = batch_asfs_mix_square_above_threshold(mix_data, threshold=0.04)

    np.testing.assert_array_equal(result[0], np.array([0, 1]))


def test_batch_threshold_uses_blocks_and_nested_list_positions() -> None:
    mix_data = MixCoefficientData(
        blocks=[
            _mix_block(
                10,
                np.array(
                    [
                        [0.5, 0.1, 0.0],
                        [0.0, 0.3, 0.1],
                    ]
                ),
            ),
            _mix_block(
                20,
                np.array(
                    [
                        [0.1, 0.6],
                        [0.4, 0.0],
                    ]
                ),
            ),
        ],
        level_list=[0.0, 0.1, 1.0, 1.1],
    )

    result = batch_asfs_mix_square_above_threshold(
        mix_data,
        asfs_position=[[1], [0]],
        threshold=0.04,
    )

    assert set(result) == {10, 20}
    np.testing.assert_array_equal(result[10], np.array([1]))
    np.testing.assert_array_equal(result[20], np.array([1]))


def test_batch_coupling_sum_uses_blocks_and_nested_list_positions() -> None:
    mix_data = MixCoefficientData(
        blocks=[
            _mix_block(
                10,
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
    blocks_csfs = [
        [
            ["line1", "line2", "a b J0"],
            ["line1", "line2", "a b J1"],
            ["line1", "line2", "a b J1"],
        ]
    ]

    result = batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum(
        blocks_CSFs_list=blocks_csfs,
        asfs_mix_data=mix_data,
        asfs_position=[[1]],
        coupling_level=1,
    )

    assert set(result) == {10}
    assert result[10][("J0",)]["sum_ci"] == [0.0]
    assert result[10][("J1",)]["sum_ci"] == [0.10]


def test_legacy_csf_sort_by_mix_coefficient_preserves_sorted_csf_records() -> None:
    csfs_block = [["csf0"], ["csf1"], ["csf2"]]
    coefficients = np.array(
        [
            [0.5, 0.1, 0.2],
            [0.0, 0.4, 0.1],
        ]
    )

    result = CSFs_sort_by_mix_coefficient(csfs_block, coefficients, threshold=0.2)

    assert result == [["csf0"], ["csf1"], ["csf2"]]


def test_legacy_csf_sort_by_mix_coefficient_uses_direct_block_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import graspkit.CSFs_processor.CSFs_choosing as csfs_choosing

    def fail_aggregate(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "CSFs_sort_by_mix_coefficient should not aggregate via legacy helper"
        )

    monkeypatch.setattr(
        csfs_choosing,
        "aggregate_ci_squared",
        fail_aggregate,
        raising=False,
    )
    csfs_block = [["csf0"], ["csf1"], ["csf2"]]
    coefficients = np.array(
        [
            [0.5, 0.1, 0.2],
            [0.0, 0.4, 0.1],
        ]
    )

    result = CSFs_sort_by_mix_coefficient(csfs_block, coefficients, threshold=0.2)

    assert result == [["csf0"], ["csf1"], ["csf2"]]
