# -*- encoding: utf-8 -*-
"""CSF candidate selection helpers for ML-guided calculation loops."""

from dataclasses import dataclass
from enum import StrEnum
import logging
import math

import numpy as np
from numpy.typing import NDArray
import polars as pl


class ContributionSource(StrEnum):
    """Source semantics for per-ASF candidate scores."""

    CLASSIFIER_PROBABILITY = "classifier_probability"
    REGRESSION_LOG_CI_SQUARED = "regression_log_ci_squared"
    PHYSICAL_CI_SQUARED = "physical_ci_squared"


@dataclass(frozen=True)
class CumulativeContributionSelection:
    """Diagnostics for per-ASF cumulative-contribution selection."""

    selected_indices: NDArray[np.int64]
    per_asf_indices: list[NDArray[np.int64]]
    per_asf_counts: list[int]
    coverages: list[float]
    union_coverage: list[float]
    overlap_matrix: NDArray[np.int64]
    exploration_indices: NDArray[np.int64]


def _as_candidate_by_asf(values: np.ndarray, n_candidates: int) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.shape[0] == n_candidates:
        return array
    if array.shape[1] == n_candidates:
        return array.T
    raise ValueError(
        "贡献矩阵维度必须与候选 CSF 数匹配: "
        f"n_candidates={n_candidates}, shape={array.shape}"
    )


def _normalized_physical_contributions(
    contributions: NDArray[np.float64],
) -> NDArray[np.float64]:
    if np.any(~np.isfinite(contributions)):
        raise ValueError("贡献度矩阵包含非有限值")
    if np.any(contributions < 0):
        raise ValueError("贡献度矩阵不能包含负值")

    totals = np.sum(contributions, axis=0)
    normalized = np.zeros_like(contributions, dtype=np.float64)
    positive_totals = totals > 0
    normalized[:, positive_totals] = (
        contributions[:, positive_totals] / totals[positive_totals]
    )
    return normalized


def _top_ranked_indices(
    candidate_indices: NDArray[np.int64],
    scores: NDArray[np.float64],
    target_count: int,
) -> NDArray[np.int64]:
    if target_count <= 0 or candidate_indices.size == 0:
        return np.array([], dtype=np.int64)
    n_select = min(target_count, candidate_indices.size)
    ranked_local = np.argsort(scores, kind="stable")[::-1][:n_select]
    return candidate_indices[ranked_local].astype(np.int64, copy=False)


def _coverage_for_indices(
    local_indices: NDArray[np.int64],
    normalized: NDArray[np.float64],
) -> list[float]:
    if local_indices.size == 0:
        return [0.0 for _ in range(normalized.shape[1])]
    return np.sum(normalized[local_indices], axis=0).astype(float).tolist()


def select_cumulative_contribution_indices(
    candidate_indices: np.ndarray,
    contributions: np.ndarray,
    *,
    threshold: float,
    logger: logging.Logger,
    exploration_scores: np.ndarray | None = None,
    exploration_ratio: float = 0.0,
    target_count: int = 0,
) -> CumulativeContributionSelection:
    """Select CSFs per ASF by cumulative normalized contribution, then union.

    Args:
        candidate_indices: Global CSF indices for candidate rows.
        contributions: Physical non-negative contribution matrix. Accepts either
            ``(n_asf, n_candidates)`` or ``(n_candidates, n_asf)``.
        threshold: Per-ASF cumulative contribution threshold in ``(0, 1]``.
        logger: Logger used for per-J/per-ASF diagnostics.
        exploration_scores: Optional ranking scores for adding a small number of
            candidates outside the cumulative-contribution union.
        exploration_ratio: Extra exploration budget as a fraction of the union
            size.
        target_count: Minimum total selected count for iterative learning. The
            cumulative contribution union is never truncated when it already
            exceeds this value.

    Returns:
        Selected global indices and diagnostic coverage/overlap details.
    """
    if not (0 < threshold <= 1):
        raise ValueError("cumulative_contribution_threshold 必须在 (0, 1] 范围内")
    if exploration_ratio < 0:
        raise ValueError("exploration_ratio 必须大于等于 0")

    candidate_indices_array = np.asarray(candidate_indices, dtype=np.int64)
    contribution_matrix = _as_candidate_by_asf(
        np.asarray(contributions, dtype=np.float64),
        candidate_indices_array.size,
    )
    normalized = _normalized_physical_contributions(contribution_matrix)

    logger.info("候选 CSF 总数: %s", candidate_indices_array.size)

    selected_local_sets: list[set[int]] = []
    per_asf_indices: list[NDArray[np.int64]] = []
    per_asf_counts: list[int] = []
    per_asf_coverages: list[float] = []

    for asf_idx in range(normalized.shape[1]):
        q_values = normalized[:, asf_idx]
        ranked = np.argsort(q_values, kind="stable")[::-1]
        cumulative = np.cumsum(q_values[ranked])
        if cumulative.size == 0 or cumulative[-1] <= 0:
            selected_local = np.array([], dtype=np.int64)
            coverage = 0.0
        else:
            cutoff_position = int(
                np.searchsorted(cumulative, threshold - 1e-12, side="left")
            )
            cutoff_position = min(cutoff_position, ranked.size - 1)
            selected_local = ranked[: cutoff_position + 1].astype(np.int64, copy=False)
            coverage = float(cumulative[cutoff_position])

        selected_local_sets.append({int(idx) for idx in selected_local.tolist()})
        per_asf_indices.append(candidate_indices_array[selected_local])
        per_asf_counts.append(int(selected_local.size))
        per_asf_coverages.append(coverage)
        logger.info(
            "ASF %s 累计贡献选择: selected=%s, coverage=%.6f, threshold=%.6f",
            asf_idx,
            selected_local.size,
            coverage,
            threshold,
        )

    union_local_set: set[int] = set()
    for local_set in selected_local_sets:
        union_local_set.update(local_set)
    union_local = np.array(sorted(union_local_set), dtype=np.int64)

    exploration_local = np.array([], dtype=np.int64)
    ratio_target = math.ceil(union_local.size * (1.0 + exploration_ratio))
    desired_total = min(
        candidate_indices_array.size,
        max(union_local.size, target_count, ratio_target),
    )
    if desired_total > union_local.size:
        if exploration_scores is None:
            rank_scores = np.max(normalized, axis=1)
        else:
            rank_scores = np.asarray(exploration_scores, dtype=np.float64)
            if rank_scores.shape != (candidate_indices_array.size,):
                raise ValueError("exploration_scores 必须是一维数组并匹配候选 CSF 数")
        exploration_count = desired_total - union_local.size
        outside_mask = np.ones(candidate_indices_array.size, dtype=bool)
        outside_mask[union_local] = False
        outside_local = np.where(outside_mask)[0]
        ranked_outside = outside_local[
            np.argsort(rank_scores[outside_local], kind="stable")[::-1]
        ]
        exploration_local = ranked_outside[:exploration_count].astype(
            np.int64,
            copy=False,
        )
        union_local = np.array(
            sorted(set(union_local.tolist()) | set(exploration_local.tolist())),
            dtype=np.int64,
        )

    overlap_matrix = np.zeros((len(selected_local_sets), len(selected_local_sets)), dtype=np.int64)
    for row, left_set in enumerate(selected_local_sets):
        for col, right_set in enumerate(selected_local_sets):
            overlap_matrix[row, col] = len(left_set & right_set)

    union_coverage = _coverage_for_indices(union_local, normalized)
    logger.info("同一 J 内 ASF 并集大小 |S_J|: %s", union_local.size)
    logger.info("每个 ASF 的并集覆盖度: %s", [round(value, 6) for value in union_coverage])
    logger.info("多 ASF CSF overlap matrix: %s", overlap_matrix.tolist())
    if exploration_local.size > 0:
        logger.info("exploration budget 额外加入 CSF 数: %s", exploration_local.size)
    logger.info("最终 selected CSF 数: %s", union_local.size)

    return CumulativeContributionSelection(
        selected_indices=candidate_indices_array[union_local],
        per_asf_indices=per_asf_indices,
        per_asf_counts=per_asf_counts,
        coverages=per_asf_coverages,
        union_coverage=union_coverage,
        overlap_matrix=overlap_matrix,
        exploration_indices=candidate_indices_array[exploration_local],
    )


def select_ml_candidate_indices(
    candidate_indices: np.ndarray,
    *,
    per_level_scores: np.ndarray,
    final_score: np.ndarray,
    target_count: int,
    config: object,
    logger: logging.Logger,
    source: ContributionSource,
    physical_contributions: np.ndarray | None,
) -> NDArray[np.int64]:
    """Select ML candidates using configured fixed-ratio or cumulative mode."""
    candidate_indices_array = np.asarray(candidate_indices, dtype=np.int64)
    final_score_array = np.asarray(final_score, dtype=np.float64)
    if final_score_array.shape != (candidate_indices_array.size,):
        raise ValueError("final_score 必须是一维数组并匹配候选 CSF 数")

    cal_settings = getattr(config, "cal_settings")
    selection_mode = getattr(cal_settings, "selection_mode", "fixed_ratio")

    if selection_mode == "fixed_ratio":
        return _top_ranked_indices(candidate_indices_array, final_score_array, target_count)

    if selection_mode != "cumulative_contribution":
        raise ValueError(
            "selection_mode 必须是 'fixed_ratio' 或 'cumulative_contribution'，"
            f"当前值: {selection_mode}"
        )

    if (
        source == ContributionSource.CLASSIFIER_PROBABILITY
        or physical_contributions is None
    ):
        logger.warning(
            "classification probability is not a physical cumulative contribution; "
            "falling back to fixed_ratio/top-k selection"
        )
        return _top_ranked_indices(candidate_indices_array, final_score_array, target_count)

    threshold = float(getattr(cal_settings, "cumulative_contribution_threshold", 0.995))
    exploration_ratio = float(getattr(cal_settings, "exploration_ratio", 0.02))
    selection = select_cumulative_contribution_indices(
        candidate_indices_array,
        physical_contributions,
        threshold=threshold,
        logger=logger,
        exploration_scores=final_score_array,
        exploration_ratio=exploration_ratio,
        target_count=target_count,
    )
    return selection.selected_indices


def log_reference_level_order_diagnostics(
    *,
    selected_energy_data: pl.DataFrame | None,
    spectral_terms: list[str],
    reference_energy_levels: list[float],
    logger: logging.Logger,
) -> None:
    """Log cross-level order differences against reference data without gating."""
    if (
        selected_energy_data is None
        or len(spectral_terms) <= 1
        or len(reference_energy_levels) != len(spectral_terms)
        or "configuration_raw" not in selected_energy_data.columns
        or "EnergyLevel" not in selected_energy_data.columns
    ):
        return

    term_to_calc_energy: dict[str, float] = {}
    for row in selected_energy_data.select(["configuration_raw", "EnergyLevel"]).iter_rows(
        named=True
    ):
        term = str(row["configuration_raw"])
        if term in spectral_terms:
            term_to_calc_energy[term] = float(row["EnergyLevel"])

    if len(term_to_calc_energy) < 2:
        return

    violations: list[str] = []
    for left_idx, left_term in enumerate(spectral_terms):
        if left_term not in term_to_calc_energy:
            continue
        for right_idx in range(left_idx + 1, len(spectral_terms)):
            right_term = spectral_terms[right_idx]
            if right_term not in term_to_calc_energy:
                continue
            reference_delta = reference_energy_levels[left_idx] - reference_energy_levels[right_idx]
            calculated_delta = term_to_calc_energy[left_term] - term_to_calc_energy[right_term]
            if reference_delta == 0 or calculated_delta == 0:
                continue
            if np.sign(reference_delta) != np.sign(calculated_delta):
                violations.append(
                    f"{left_term} vs {right_term}: "
                    f"calc_delta={calculated_delta:.6g}, ref_delta={reference_delta:.6g}"
                )

    if violations:
        logger.warning(
            "cross-J/reference level-order violation diagnostics: %s. "
            "This is diagnostic only; selection is not forced to match reference order.",
            violations,
        )
    else:
        logger.info("cross-J/reference level-order diagnostics: no order violations detected")
