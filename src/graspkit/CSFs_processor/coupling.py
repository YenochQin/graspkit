# -*- encoding: utf-8 -*-
"""Coupling-J pattern collection and CI-square summaries for CSF blocks."""

from __future__ import annotations

import logging
from collections import Counter
from typing import TypedDict

import numpy as np

from ..utils.data_modules import MixCoefficientData
from .validation import validate_coupling_level, validate_csf_records

logger = logging.getLogger(__name__)


class CouplingJInfo(TypedDict):
    """Summary for CSFs sharing the same final coupling-J pattern.

    Attributes:
        count: Number of CSFs that match the coupling pattern.
        idxs: CSF indices in the source block.
    """

    count: int
    idxs: list[int]


class CouplingJInfoWithSumCi(TypedDict):
    """Coupling-J summary with one accumulated CI-square contribution.

    Attributes:
        count: Number of CSFs that match the coupling pattern.
        idxs: CSF indices in the source block.
        sum_ci: Sum of squared CI coefficients for the matching CSFs.
    """

    count: int
    idxs: list[int]
    sum_ci: float


class CouplingJInfoWithSumCiList(TypedDict):
    """Coupling-J summary with CI-square contributions for multiple ASFs.

    Attributes:
        count: Number of CSFs that match the coupling pattern.
        idxs: CSF indices in the source block.
        sum_ci: Per-selected-ASF sums of squared CI coefficients.
    """

    count: int
    idxs: list[int]
    sum_ci: list[float]


def single_block_csfs_final_coupling_J_collector(
    block_csfs: list[list[str]], coupling_level: int | None = None
) -> dict[tuple[str, ...], CouplingJInfo]:
    """Collect final coupling-J patterns from one CSF block.

    Args:
        block_csfs: CSF records whose third line contains whitespace-separated
            coupling tokens.
        coupling_level: Number of trailing coupling tokens to keep. If None,
            all tokens are used. Shorter CSF records fall back to all tokens.

    Returns:
        Mapping from coupling-token pattern to its occurrence count and CSF
        indices.

    Raises:
        ValueError: If ``coupling_level`` is not ``None`` or positive, or a
            CSF record does not have exactly 3 lines.
    """
    coupling_level = validate_coupling_level(coupling_level)
    validate_csf_records(block_csfs)

    all_tokens: list[tuple[str, ...]] = [
        tuple(csf[2].lstrip().split()) for csf in block_csfs
    ]

    if coupling_level is None:
        selected_tokens = all_tokens
    else:
        selected_tokens = [
            tokens[-coupling_level:] if len(tokens) >= coupling_level else tokens
            for tokens in all_tokens
        ]

    coupling_J_counts = Counter(selected_tokens)
    coupling_J_collection: dict[tuple[str, ...], CouplingJInfo] = {
        pattern: {"count": cnt, "idxs": []}
        for pattern, cnt in coupling_J_counts.items()
    }
    for idx, pattern in enumerate(selected_tokens):
        coupling_J_collection[pattern]["idxs"].append(idx)

    return coupling_J_collection


def batch_blocks_csfs_final_coupling_J_collection(
    blocks_csfs_list: list[list[list[str]]], coupling_level: int | None = None
) -> dict[int, dict[tuple[str, ...], CouplingJInfo]]:
    """Collect final coupling-J patterns for every CSF block.

    Args:
        blocks_csfs_list: CSF blocks, where each block contains three-line CSF
            records.
        coupling_level: Number of trailing coupling tokens to keep for each
            CSF pattern.

    Returns:
        Dictionary keyed by block index with per-pattern coupling summaries.
    """
    blocks_coupling_J_collection: dict[int, dict[tuple[str, ...], CouplingJInfo]] = {}
    for block, block_csfs in enumerate(blocks_csfs_list):
        logger.info(f"Block {block + 1}: 包含 {len(block_csfs)} 个 CSF")
        block_coupling_J_collection = single_block_csfs_final_coupling_J_collector(
            block_csfs, coupling_level
        )
        blocks_coupling_J_collection[block] = block_coupling_J_collection
    return blocks_coupling_J_collection


def single_asf_csfs_final_coupling_J_mix_coefficient_sum(
    block_csfs_coupling_J_collection_dict: dict[tuple[str, ...], CouplingJInfo],
    mix_coefficient_list: list[float] | np.ndarray,
) -> dict[tuple[str, ...], CouplingJInfoWithSumCi]:
    """Sum squared CI coefficients for each coupling-J pattern in one ASF.

    Args:
        block_csfs_coupling_J_collection_dict: Coupling summaries for a CSF
            block.
        mix_coefficient_list: CI coefficients aligned with the CSFs in the
            same block.

    Returns:
        Coupling summaries augmented with the summed squared CI contribution.
    """
    coeff_array = np.asarray(mix_coefficient_list)
    result: dict[tuple[str, ...], CouplingJInfoWithSumCi] = {}
    for pattern, info in block_csfs_coupling_J_collection_dict.items():
        idxs = info["idxs"]
        sum_ci = float(np.sum(coeff_array[idxs] ** 2))
        logger.debug(f"{pattern=}  count={info['count']}  sum_ci={sum_ci}")
        result[pattern] = {"count": info["count"], "idxs": idxs, "sum_ci": sum_ci}
    return result


def single_block_batch_asfs_CSFs_final_coupling_J_collection(
    block_CSFs: list[list[str]],
    block_asfs_mix_coefficient_list: list[np.ndarray] | np.ndarray,
    block_asfs_position: list[int] | np.ndarray | None = None,
    coupling_level: int | None = None,
) -> dict[tuple[str, ...], CouplingJInfoWithSumCiList]:
    """Collect coupling-J contributions for selected ASFs in one CSF block.

    Args:
        block_CSFs: CSF records for a single block.
        block_asfs_mix_coefficient_list: Matrix-like ASF by CSF CI
            coefficients for the block.
        block_asfs_position: ASF indices to include. If None, all ASFs are
            included.
        coupling_level: Number of trailing coupling tokens used to define each
            coupling pattern.

    Returns:
        Mapping from coupling pattern to counts, CSF indices, and
        per-selected-ASF summed squared CI coefficients.
    """
    if block_asfs_position is None:
        block_asfs_position = list(range(len(block_asfs_mix_coefficient_list)))

    coeff_matrix = np.asarray(block_asfs_mix_coefficient_list)

    base_coupling_dict = single_block_csfs_final_coupling_J_collector(
        block_CSFs, coupling_level
    )
    result: dict[tuple[str, ...], CouplingJInfoWithSumCiList] = {}
    for pattern, info in base_coupling_dict.items():
        idxs = np.asarray(info["idxs"])
        sum_ci_list: list[float] = []
        for asf_idx in block_asfs_position:
            asf_coeff = coeff_matrix[int(asf_idx)]
            sum_ci_list.append(float(np.sum(asf_coeff[idxs] ** 2)))
        result[pattern] = {"count": info["count"], "idxs": info["idxs"], "sum_ci": sum_ci_list}

    return result


def batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum(
    blocks_CSFs_list: list[list[list[str]]],
    asfs_mix_data: MixCoefficientData,
    asfs_position: list[list[int]] | None = None,
    coupling_level: int | None = None,
) -> dict[int, dict[tuple[str, ...], CouplingJInfoWithSumCiList]]:
    """Collect coupling-J CI-square summaries for all CSF blocks.

    Args:
        blocks_CSFs_list: CSF blocks aligned with ``asfs_mix_data``.
        asfs_mix_data: Parsed ASF mixing-coefficient data.
        asfs_position: Optional selected ASF indices for each block. Defaults
            to each block's ``level_indices``.
        coupling_level: Number of trailing coupling tokens used to define each
            coupling pattern.

    Returns:
        Nested dictionary keyed first by block index and then by coupling
        pattern.

    Raises:
        ValueError: If ASF positions are inconsistent with the mixing data or
            CSF and coefficient lengths do not match.
    """
    if asfs_position is None:
        normalized_positions: list[list[int]] = [
            block.level_indices.astype(np.int64).tolist()
            for block in asfs_mix_data.blocks
        ]
    else:
        normalized_positions = [
            [int(pos) for pos in block_positions]
            for block_positions in asfs_position
        ]

    if len(normalized_positions) != len(asfs_mix_data.blocks):
        raise ValueError(
            f"asfs_position 第一层长度 {len(normalized_positions)} "
            f"与 blocks {len(asfs_mix_data.blocks)} 不一致。"
        )

    if len(blocks_CSFs_list) != len(asfs_mix_data.blocks):
        raise ValueError(
            f"blocks_CSFs_list 长度 {len(blocks_CSFs_list)} "
            f"与 mix blocks {len(asfs_mix_data.blocks)} 不一致。"
        )

    blocks_asfs_coupling_J_sum_ci: dict[int, dict[tuple[str, ...], CouplingJInfoWithSumCiList]] = {}
    for block_offset, (block_csfs, mix_block, selected_positions) in enumerate(
        zip(blocks_CSFs_list, asfs_mix_data.blocks, normalized_positions, strict=True)
    ):
        allowed_positions = set(mix_block.level_indices.astype(np.int64).tolist())
        selected_set = {int(pos) for pos in selected_positions}
        if not selected_set.issubset(allowed_positions):
            raise ValueError(
                f"asfs_position 第 {block_offset} 层元素 {sorted(selected_set)} "
                f"不是 block.level_indices 对应层 {sorted(allowed_positions)} 的子集。"
            )

        logger.info(f"Block {mix_block.block_index + 1}: 包含 {len(mix_block.mix_coefficients)} 个 ASF")
        if any(len(asf_mix) != len(block_csfs) for asf_mix in mix_block.mix_coefficients):
            raise ValueError(
                f"Block {mix_block.block_index}: block_CSFs 长度 {len(block_csfs)} 与 "
                f"block_asfs_mix_coefficient 长度不匹配。"
            )

        blocks_asfs_coupling_J_sum_ci[mix_block.block_index] = (
            single_block_batch_asfs_CSFs_final_coupling_J_collection(
                block_CSFs=block_csfs,
                block_asfs_mix_coefficient_list=mix_block.mix_coefficients,
                block_asfs_position=selected_positions,
                coupling_level=coupling_level,
            )
        )

    return blocks_asfs_coupling_J_sum_ci
