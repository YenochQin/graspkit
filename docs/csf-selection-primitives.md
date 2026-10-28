# CI weights, CSF indices and orbital coverage

Available in the prepared GraspKit 3.5.2 source version.

The public functions below belong to `graspkit.CSFs_processor`. They do not
depend on ML configuration, PyTorch, scikit-learn or a calculation workflow.
Callers choose thresholds and budgets and decide what to do with the returned
diagnostics. Descriptor generation and the V2 format remain owned by rCSFs.

## Cumulative selection

```python
import numpy as np
from graspkit.CSFs_processor import select_csf_indices_by_cumulative_weight

selection = select_csf_indices_by_cumulative_weight(
    np.array([[0.90, 0.09, 0.01], [0.05, 0.90, 0.05]]),
    csf_indices=np.array([10, 20, 30], dtype=np.int64),
    cumulative_threshold=0.99,
)
```

The matrix always has ASF rows and CSF columns; a one-dimensional array denotes
one ASF. Each row is sorted independently, the first item crossing the target
is included, and the per-ASF prefixes are unioned. Equal weights use descending
CSF index as a deterministic tie breaker. `csf_indices` optionally maps columns
to a full pool; without it, indices refer to local columns.

`CumulativeCsfSelection` contains `indices`, `per_asf_indices`, original row
`totals`, and per-ASF `retained_weight`/`discarded_weight` fractions. By default
the union is sorted by index. `index_order="importance"` preserves the first
occurrence while visiting each ASF's descending prefix; it is not a joint
cross-state ranking.

- `normalization="unit"` is the default for complete physical ASF coefficients.
  Rows must sum to one within an absolute numerical tolerance of `1e-8`.
  Only rounding-level discrepancies are renormalized.
- `normalization="relative"` explicitly permits arbitrary finite nonnegative
  totals for post-processing. Zero-total rows yield empty prefixes and zero
  retained/discarded fractions.
- A target of exactly one keeps every positive input weight. It does not select
  trailing zero entries or overrun the last row because of cumulative rounding.

Neither mode certifies energies, other observables or an uncomputed CSF space.
Do not interpret mixed observations from different CI spaces as one normalized
wavefunction.

`select_ci_cumulative_local_idxs` applies the same kernel to a sequence of
blocks, with optional per-block ASF **matrix row** selections and concatenated
block offsets. It uses relative normalization and importance traversal order.
The existing `rmix_cumulative_selected_row_idxs(RmixCiSquaredData, ...)` delegates
to this block interface; its loader has already selected the requested ASF rows.

## Indices

- `validate_selection_idxs` checks one-dimensional integer indices, bounds,
  duplicates and signed-int64 range.
- `project_local_idxs_to_full_pool_idxs(source_indices, local_indices)` maps
  local selections through an explicit source-to-pool mapping while preserving
  request order. Both arrays are validated; `pool_row_count` can additionally
  bound the global mapping.
- `iter_unselected_csf_indices(total_count, excluded_indices, batch_size)` yields
  the sorted complement in bounded chunks, without allocating a full-pool mask.

## Orbital coverage

```python
from graspkit.CSFs_processor import complete_orbital_coverage

occupied = np.array([[True, False], [False, True], [True, True]])
coverage = complete_orbital_coverage(
    np.array([0], dtype=np.int64),
    read_occupations=lambda indices: occupied[indices],
    total_count=3,
    orbital_count=2,
    batch_size=2,
)
assert coverage.added_indices.tolist() == [1]
```

The reader returns a boolean `(requested CSF, orbital)` matrix in the requested
index order. A descriptor consumer extracts validated occupation columns;
models, scaled feature arrays and non-occupation channels do not determine the
meaning of these masks. Kit neither guesses a descriptor version nor depends
on the reader's storage implementation.

`count_orbital_coverage` counts selected CSFs occupying each requested orbital.
`complete_orbital_coverage` retains the current selection and visits the index
complement in source order, adding only rows that cover a remaining gap. This
is deterministic, not a globally minimum set-cover solver.

`OrbitalCoverageResult` returns the selected union, added indices, occupation
counts and missing orbitals before/after scanning. Uncoverable orbitals remain
in `missing_after`; EOF is not success. The caller decides whether partial
coverage is an error. Empty initial selections are supported. If no additions
are needed, input order is preserved; otherwise the union is index-sorted.

This function does not impose a CSF cap, choose which ML candidates may be
withdrawn, preserve exploration quotas, or change workflow state. Those
decisions remain in the consuming application.
