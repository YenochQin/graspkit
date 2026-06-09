import logging
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from graspkit.ml_module.ml_selection import (
    ContributionSource,
    log_reference_level_order_diagnostics,
    select_cumulative_contribution_indices,
    select_ml_candidate_indices,
)
from graspkit.ml_module.ml_regression_trainer import predict_regression_model
from graspkit.utils.data_modules import MLDataCounts


def test_cumulative_contribution_selects_single_asf_threshold() -> None:
    contributions = np.array([[0.50, 0.30, 0.15, 0.05]], dtype=float)
    global_indices = np.array([10, 11, 12, 13], dtype=np.int64)

    result = select_cumulative_contribution_indices(
        global_indices,
        contributions,
        threshold=0.80,
        logger=logging.getLogger("test-single-asf"),
    )

    np.testing.assert_array_equal(result.selected_indices, np.array([10, 11]))
    assert result.per_asf_counts == [2]
    assert result.coverages == pytest.approx([0.80])
    assert result.union_coverage == pytest.approx([0.80])


def test_cumulative_contribution_uses_union_across_asfs() -> None:
    contributions = np.array(
        [
            [0.50, 0.30, 0.10, 0.05, 0.05],
            [0.05, 0.05, 0.55, 0.30, 0.05],
        ],
        dtype=float,
    )
    global_indices = np.array([20, 21, 22, 23, 24], dtype=np.int64)

    result = select_cumulative_contribution_indices(
        global_indices,
        contributions,
        threshold=0.80,
        logger=logging.getLogger("test-multi-asf"),
    )

    np.testing.assert_array_equal(result.selected_indices, np.array([20, 21, 22, 23]))
    assert result.per_asf_counts == [2, 2]
    assert result.union_coverage == pytest.approx([0.95, 0.95])
    assert result.overlap_matrix[0, 1] == 0


def test_cumulative_contribution_union_grows_when_asf_overlap_is_small() -> None:
    contributions = np.array(
        [
            [0.60, 0.30, 0.05, 0.05],
            [0.05, 0.05, 0.60, 0.30],
        ],
        dtype=float,
    )
    global_indices = np.array([1, 2, 3, 4], dtype=np.int64)

    result = select_cumulative_contribution_indices(
        global_indices,
        contributions,
        threshold=0.90,
        logger=logging.getLogger("test-low-overlap"),
    )

    assert result.per_asf_counts == [2, 2]
    assert len(result.selected_indices) == 4


def test_classifier_only_cannot_claim_cumulative_physical_contribution(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = SimpleNamespace(
        cal_settings=SimpleNamespace(
            selection_mode="cumulative_contribution",
            cumulative_contribution_threshold=0.995,
            exploration_ratio=0.0,
        )
    )
    candidate_indices = np.array([1, 2, 3], dtype=np.int64)
    classifier_probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.7]])
    final_score = np.max(classifier_probabilities, axis=1)

    with caplog.at_level(logging.WARNING):
        selected = select_ml_candidate_indices(
            candidate_indices,
            per_level_scores=classifier_probabilities,
            final_score=final_score,
            target_count=2,
            config=config,
            logger=logging.getLogger("test-classifier-fallback"),
            source=ContributionSource.CLASSIFIER_PROBABILITY,
            physical_contributions=None,
        )

    np.testing.assert_array_equal(selected, np.array([1, 2]))
    assert "classification probability is not a physical cumulative contribution" in caplog.text


def test_fixed_ratio_selection_still_uses_top_ranked_candidates() -> None:
    config = SimpleNamespace(
        cal_settings=SimpleNamespace(
            selection_mode="fixed_ratio",
            cumulative_contribution_threshold=0.995,
            exploration_ratio=0.0,
        )
    )
    candidate_indices = np.array([10, 11, 12], dtype=np.int64)
    scores = np.array([0.2, 0.9, 0.5])

    selected = select_ml_candidate_indices(
        candidate_indices,
        per_level_scores=scores[:, None],
        final_score=scores,
        target_count=2,
        config=config,
        logger=logging.getLogger("test-fixed-ratio"),
        source=ContributionSource.REGRESSION_LOG_CI_SQUARED,
        physical_contributions=10.0 ** scores[:, None],
    )

    np.testing.assert_array_equal(selected, np.array([11, 12]))


def test_regression_cumulative_mode_selects_union_from_predicted_log_ci(
    tmp_path,
) -> None:
    config = SimpleNamespace(
        cal_settings=SimpleNamespace(
            expansion_ratio=1.0,
            sampling_ratio=1.0,
            selection_mode="cumulative_contribution",
            cumulative_contribution_threshold=0.80,
            exploration_ratio=0.0,
            reference_energy_levels=[],
            reference_energy_mode="monitor",
            reference_energy_score_weight=0.0,
            spectral_term=["ASF0", "ASF1"],
        ),
        cal_path=SimpleNamespace(results_path=tmp_path),
    )

    class FixedLogCiModel:
        def predict_batch(self, descriptors: np.ndarray, batch_size: int) -> np.ndarray:
            return np.log10(
                np.array(
                    [
                        [0.50, 0.05],
                        [0.30, 0.05],
                        [0.10, 0.55],
                        [0.05, 0.30],
                        [0.05, 0.05],
                    ],
                    dtype=float,
                )
            )

    result = predict_regression_model(
        FixedLogCiModel(),
        np.zeros((5, 1), dtype=float),
        np.array([20, 21, 22, 23, 24], dtype=np.int64),
        np.array([100], dtype=np.int64),
        config,
        MLDataCounts(total_csfs_count=6, cal_csfs_count=1),
        logging.getLogger("test-regression-cumulative"),
    )

    np.testing.assert_array_equal(result[0], np.array([20, 21, 22, 23]))


def test_cumulative_mode_fills_to_target_count_with_exploration_budget() -> None:
    config = SimpleNamespace(
        cal_settings=SimpleNamespace(
            selection_mode="cumulative_contribution",
            cumulative_contribution_threshold=0.80,
            exploration_ratio=0.0,
        )
    )
    candidate_indices = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    physical_contributions = np.array([[0.80], [0.10], [0.04], [0.03], [0.03]])
    scores = np.array([0.80, 0.20, 0.90, 0.70, 0.60])

    selected = select_ml_candidate_indices(
        candidate_indices,
        per_level_scores=np.log10(physical_contributions),
        final_score=scores,
        target_count=3,
        config=config,
        logger=logging.getLogger("test-cumulative-target-fill"),
        source=ContributionSource.REGRESSION_LOG_CI_SQUARED,
        physical_contributions=physical_contributions,
    )

    np.testing.assert_array_equal(selected, np.array([10, 12, 13]))


def test_reference_level_order_diagnostics_warns_without_enforcing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    energy_data = pl.DataFrame(
        {"configuration_raw": ["low", "high"], "EnergyLevel": [100.0, 0.0]}
    )

    with caplog.at_level(logging.WARNING):
        log_reference_level_order_diagnostics(
            selected_energy_data=energy_data,
            spectral_terms=["low", "high"],
            reference_energy_levels=[0.0, 100.0],
            logger=logging.getLogger("test-level-order"),
        )

    assert "cross-J/reference level-order violation diagnostics" in caplog.text
