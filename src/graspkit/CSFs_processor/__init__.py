# -*- encoding: utf-8 -*-
"""CSF parsing, coupling-J analysis, and deterministic selection algorithms."""

from .coupling import (
    collect_coupling_groups,
    select_csfs_by_coupling_theme,
    summarize_coupling_ci_squared,
)
from .extraction import (
    CsfDocument,
    create_csf_document,
    write_csf_documents,
)
from .coverage import (
    OrbitalCoverageResult,
    complete_orbital_coverage,
    count_orbital_coverage,
)
from .selection import (
    CumulativeCsfSelection,
    generate_unique_random_numbers,
    random_choose_csfs,
    rmix_cumulative_selected_row_idxs,
    select_csf_indices_above_ci_squared_cutoff,
    select_csf_indices_by_ci_squared_cutoff,
    select_csfs_rows,
    sort_csfs_by_mix_coefficient,
    select_csf_indices_by_cumulative_weight,
    select_ci_cumulative_local_idxs,
    project_local_idxs_to_full_pool_idxs,
    iter_unselected_csf_indices,
)
from .validation import (
    load_selection_idxs,
    validate_coupling_level,
    validate_header_lines,
    validate_headers_match,
    validate_output_path_disjoint,
    validate_selection_idxs,
)

__all__ = [
    # coupling
    "collect_coupling_groups",
    "select_csfs_by_coupling_theme",
    "summarize_coupling_ci_squared",
    # selection
    "select_csf_indices_above_ci_squared_cutoff",
    "select_csf_indices_by_ci_squared_cutoff",
    "sort_csfs_by_mix_coefficient",
    "generate_unique_random_numbers",
    "random_choose_csfs",
    "select_csfs_rows",
    "rmix_cumulative_selected_row_idxs",
    "CumulativeCsfSelection",
    "select_csf_indices_by_cumulative_weight",
    "select_ci_cumulative_local_idxs",
    "project_local_idxs_to_full_pool_idxs",
    "iter_unselected_csf_indices",
    # orbital coverage
    "OrbitalCoverageResult",
    "complete_orbital_coverage",
    "count_orbital_coverage",
    # extraction
    "CsfDocument",
    "create_csf_document",
    "write_csf_documents",
    # validation
    "validate_selection_idxs",
    "load_selection_idxs",
    "validate_header_lines",
    "validate_headers_match",
    "validate_output_path_disjoint",
    "validate_coupling_level",
]
