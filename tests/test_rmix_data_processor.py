import numpy as np
import pytest

from graspkit.grasp_data_extractor.rmix_data_processor import (
    aggregate_ci_squared,
    ci_squared,
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
