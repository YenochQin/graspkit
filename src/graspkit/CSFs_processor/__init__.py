# -*- encoding: utf-8 -*-
"""CSF parsing, coupling-J analysis, and deterministic selection algorithms."""

from .coupling import (
    CouplingJInfo,
    CouplingJInfoWithSumCi,
    CouplingJInfoWithSumCiList,
    batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum,
    batch_blocks_csfs_final_coupling_J_collection,
    collect_coupling_groups,
    select_csfs_by_coupling_theme,
    single_asf_csfs_final_coupling_J_mix_coefficient_sum,
    single_block_batch_asfs_CSFs_final_coupling_J_collection,
    single_block_csfs_final_coupling_J_collector,
    summarize_coupling_ci_squared,
)
from .extraction import (
    CsfDocument,
    SelectedCsfsBlock,
    create_csf_document,
    merge_and_write_csfs_blocks,
    select_csfs_block,
    write_csf_documents,
)
from .selection import (
    CSFs_block_get_CSF,
    CSFs_sort_by_mix_coefficient,
    batch_asfs_mix_square_above_threshold,
    generate_unique_random_numbers,
    radom_choose_csfs,
    random_choose_csfs,
    rmix_cumulative_selected_row_idxs,
    select_csfs_rows,
    sort_csfs_by_mix_coefficient,
    single_asf_mix_square_above_threshold,
    union_lists_with_order,
)
from .validation import (
    load_selection_idxs,
    normalize_asf_positions,
    validate_coupling_level,
    validate_csf_records,
    validate_header_lines,
    validate_headers_match,
    validate_output_path_disjoint,
    validate_selection_idxs,
)

__all__ = [
    # coupling
    "CouplingJInfo",
    "CouplingJInfoWithSumCi",
    "CouplingJInfoWithSumCiList",
    "collect_coupling_groups",
    "select_csfs_by_coupling_theme",
    "summarize_coupling_ci_squared",
    "single_block_csfs_final_coupling_J_collector",
    "batch_blocks_csfs_final_coupling_J_collection",
    "single_asf_csfs_final_coupling_J_mix_coefficient_sum",
    "single_block_batch_asfs_CSFs_final_coupling_J_collection",
    "batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum",
    # selection
    "single_asf_mix_square_above_threshold",
    "batch_asfs_mix_square_above_threshold",
    "CSFs_block_get_CSF",
    "union_lists_with_order",
    "CSFs_sort_by_mix_coefficient",
    "sort_csfs_by_mix_coefficient",
    "generate_unique_random_numbers",
    "radom_choose_csfs",
    "random_choose_csfs",
    "select_csfs_rows",
    "rmix_cumulative_selected_row_idxs",
    # extraction
    "CsfDocument",
    "create_csf_document",
    "write_csf_documents",
    "SelectedCsfsBlock",
    "select_csfs_block",
    "merge_and_write_csfs_blocks",
    # validation
    "validate_selection_idxs",
    "load_selection_idxs",
    "normalize_asf_positions",
    "validate_header_lines",
    "validate_headers_match",
    "validate_output_path_disjoint",
    "validate_coupling_level",
    "validate_csf_records",
]
