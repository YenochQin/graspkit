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
from .selection import (
    generate_unique_random_numbers,
    random_choose_csfs,
    rmix_cumulative_selected_row_idxs,
    select_csf_indices_above_ci_squared_cutoff,
    select_csf_indices_by_ci_squared_cutoff,
    select_csfs_rows,
    sort_csfs_by_mix_coefficient,
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
