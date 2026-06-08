from pathlib import Path

import numpy as np
import pytest

from graspkit.grasp_data_extractor import rmix_data_processor
from graspkit.grasp_data_extractor.rmix_data_processor import (
    RmixAsfSelection,
    RmixBlockSelection,
    aggregate_ci_squared,
    analyze_rmix_file,
    ci_squared,
    filter_ci_scores_by_threshold,
    filter_sorted_ci_scores_by_cumulative,
    select_block_ci_scores,
    sort_ci_scores,
)
from graspkit.CSFs_processor.CSFs_choosing import (
    CSFs_sort_by_mix_coefficient,
    batch_asfs_mix_square_above_threshold,
    single_asf_mix_square_above_threshold,
)
from graspkit.utils.data_modules import MixCoefficientData


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


def test_select_block_ci_scores_returns_per_asf_scores_without_aggregation() -> None:
    coefficients = np.array(
        [
            [0.5, 0.1, 0.2],
            [0.0, 0.4, 0.1],
        ]
    )

    result = select_block_ci_scores(coefficients, block_index=2)

    assert isinstance(result, RmixBlockSelection)
    assert result.block_index == 2
    np.testing.assert_array_equal(result.selected_asf_indices, np.array([0, 1]))
    np.testing.assert_allclose(
        result.ci_squared,
        np.array([[0.25, 0.01, 0.04], [0.0, 0.16, 0.01]]),
    )

    assert len(result.asf_selections) == 2
    assert isinstance(result.asf_selections[0], RmixAsfSelection)
    assert result.asf_selections[0].asf_index == 0
    np.testing.assert_array_equal(
        result.asf_selections[0].selected_csf_indices,
        np.array([0, 2, 1]),
    )
    np.testing.assert_allclose(
        result.asf_selections[0].scores,
        np.array([0.25, 0.04, 0.01]),
    )
    np.testing.assert_allclose(
        result.asf_selections[0].selected_cumulative_scores,
        np.array([0.25 / 0.3, 0.29 / 0.3, 1.0]),
    )


def test_select_block_ci_scores_supports_select_asfs() -> None:
    coefficients = np.array(
        [
            [0.5, 0.1, 0.2],
            [0.0, 0.4, 0.1],
        ]
    )

    result = select_block_ci_scores(coefficients, select_asfs=[1])

    np.testing.assert_array_equal(result.selected_asf_indices, np.array([1]))
    np.testing.assert_allclose(result.ci_squared, np.array([[0.0, 0.16, 0.01]]))
    assert result.asf_selections[0].asf_index == 1
    np.testing.assert_array_equal(
        result.asf_selections[0].selected_csf_indices,
        np.array([1, 2, 0]),
    )
    np.testing.assert_allclose(
        result.asf_selections[0].scores,
        np.array([0.16, 0.01, 0.0]),
    )


def test_select_block_ci_scores_combines_threshold_and_cumulative_limits() -> None:
    coefficients = np.array(
        [
            [0.5, 0.1, 0.2],
            [0.0, 0.4, 0.1],
        ]
    )

    result = select_block_ci_scores(
        coefficients,
        score_threshold=0.1,
        cumulative_threshold=0.9,
    )

    assert len(result.asf_selections) == 2
    np.testing.assert_array_equal(
        result.asf_selections[0].selected_csf_indices,
        np.array([0]),
    )
    np.testing.assert_allclose(
        result.asf_selections[0].selected_scores,
        np.array([0.25]),
    )
    np.testing.assert_array_equal(
        result.asf_selections[1].selected_csf_indices,
        np.array([1]),
    )
    np.testing.assert_allclose(
        result.asf_selections[1].selected_scores,
        np.array([0.16]),
    )


def test_select_block_ci_scores_supports_top_k() -> None:
    coefficients = np.array([[0.5, 0.1, 0.2]])

    result = select_block_ci_scores(coefficients, top_k=2)

    np.testing.assert_array_equal(
        result.asf_selections[0].selected_csf_indices,
        np.array([0, 2]),
    )
    np.testing.assert_allclose(
        result.asf_selections[0].selected_scores,
        np.array([0.25, 0.04]),
    )


def test_select_block_ci_scores_supports_top_ratio_with_ceiling() -> None:
    coefficients = np.array([[0.5, 0.1, 0.2]])

    result = select_block_ci_scores(coefficients, top_ratio=0.34)

    np.testing.assert_array_equal(
        result.asf_selections[0].selected_csf_indices,
        np.array([0, 2]),
    )


def test_select_block_ci_scores_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        select_block_ci_scores(np.array([[0.5]]), top_k=0)


def test_select_block_ci_scores_rejects_invalid_top_ratio() -> None:
    with pytest.raises(ValueError, match="top_ratio must be"):
        select_block_ci_scores(np.array([[0.5]]), top_ratio=1.2)


def test_analyze_rmix_file_loads_data_and_selects_each_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_paths: list[Path] = []

    class DummyLoader:
        def __init__(self, file_path: str | Path) -> None:
            loaded_paths.append(Path(file_path))

        def load(self) -> MixCoefficientData:
            return MixCoefficientData(
                block_num=2,
                block_idx_list=[0, 1],
                block_CSFs_nums=[3, 2],
                block_energy_count_list=[1, 1],
                level_J_value_list=["0", "1"],
                parity_list=[1, 1],
                block_levels_idx_list=[np.array([0]), np.array([0])],
                block_energy_list=[0.0, 0.0],
                block_level_energy_list=[np.array([0.0]), np.array([0.0])],
                mix_coefficient_list=[
                    np.array([[0.5, 0.1, 0.2], [0.0, 0.4, 0.1]]),
                    np.array([[0.3, 0.4], [0.5, 0.1]]),
                ],
                level_list=[0.0, 0.1],
            )

    monkeypatch.setattr(rmix_data_processor, "MixCoefLoader", DummyLoader)

    result = analyze_rmix_file(
        "/tmp/example.m",
        select_asfs=[np.array([1]), [0]],
        top_k=1,
    )

    assert loaded_paths == [Path("/tmp/example.m")]
    assert len(result) == 2
    assert result[0].block_index == 0
    np.testing.assert_array_equal(result[0].selected_asf_indices, np.array([1]))
    np.testing.assert_array_equal(
        result[0].asf_selections[0].selected_csf_indices,
        np.array([1]),
    )
    assert result[1].block_index == 1
    np.testing.assert_array_equal(result[1].selected_asf_indices, np.array([0]))
    np.testing.assert_array_equal(
        result[1].asf_selections[0].selected_csf_indices,
        np.array([1]),
    )


def test_rmix_processor_api_is_exported_from_grasp_data_extractor_package() -> None:
    from graspkit.grasp_data_extractor import (
        RmixAsfSelection,
        RmixBlockSelection,
        aggregate_ci_squared,
        analyze_rmix_file,
        ci_squared,
        filter_ci_scores_by_threshold,
        filter_sorted_ci_scores_by_cumulative,
        select_block_ci_scores,
        sort_ci_scores,
    )

    assert RmixAsfSelection.__name__ == "RmixAsfSelection"
    assert RmixBlockSelection.__name__ == "RmixBlockSelection"
    assert callable(aggregate_ci_squared)
    assert callable(analyze_rmix_file)
    assert callable(ci_squared)
    assert callable(filter_ci_scores_by_threshold)
    assert callable(filter_sorted_ci_scores_by_cumulative)
    assert callable(select_block_ci_scores)
    assert callable(sort_ci_scores)


def test_legacy_single_asf_threshold_keeps_existing_tuple_index_shape() -> None:
    coefficients = np.array([0.5, -0.2, 0.1])

    result = single_asf_mix_square_above_threshold(coefficients, threshold=0.04)

    assert result == [(0,), (1,)]


def test_legacy_batch_threshold_returns_unique_indices_per_block() -> None:
    mix_data = MixCoefficientData(
        block_num=1,
        block_idx_list=[0],
        block_CSFs_nums=[3],
        block_energy_count_list=[2],
        level_J_value_list=["0"],
        parity_list=[1],
        block_levels_idx_list=[np.array([0, 1])],
        block_energy_list=[0.0],
        block_level_energy_list=[np.array([0.0, 0.1])],
        mix_coefficient_list=[
            np.array(
                [
                    [0.5, 0.1, 0.0],
                    [0.0, 0.3, 0.1],
                ]
            )
        ],
        level_list=[0.0, 0.1],
    )

    result = batch_asfs_mix_square_above_threshold(mix_data, threshold=0.04)

    np.testing.assert_array_equal(result[0], np.array([0, 1]))


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
