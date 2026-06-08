import numpy as np
import pytest

from graspkit.grasp_data_extractor.rmix_data_processor import (
    aggregate_ci_squared,
    ci_squared,
    filter_ci_scores_by_threshold,
    filter_sorted_ci_scores_by_cumulative,
    sort_ci_scores,
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
