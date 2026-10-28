"""Deterministic orbital-coverage operations independent of ML and workflows."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .selection import iter_unselected_csf_indices
from .validation import validate_selection_idxs

type OccupationReader = Callable[[NDArray[np.int64]], NDArray[np.bool_]]


@dataclass(frozen=True)
class OrbitalCoverageResult:
    selected_indices: NDArray[np.int64]
    added_indices: NDArray[np.int64]
    occupation_counts: NDArray[np.int64]
    missing_before: NDArray[np.int64]
    missing_after: NDArray[np.int64]


def _read_occupations(
    reader: OccupationReader,
    indices: NDArray[np.int64],
    orbital_count: int,
) -> NDArray[np.bool_]:
    occupied = np.asarray(reader(indices))
    if occupied.dtype != np.bool_ or occupied.shape != (len(indices), orbital_count):
        raise ValueError(
            "Occupation reader must return a boolean (requested CSF, orbital) matrix in requested order"
        )
    return occupied


def count_orbital_coverage(
    indices: np.ndarray,
    *,
    read_occupations: OccupationReader,
    orbital_count: int,
    total_count: int | None = None,
    batch_size: int = 100_000,
) -> NDArray[np.int64]:
    """Count selected CSFs occupying each orbital using bounded reads.

    The reader maps the supplied CSF indices to validated occupation masks.
    It is responsible for extracting occupation columns from the source format;
    no descriptor version, feature scaling, model or configuration is inferred.
    """
    if orbital_count < 0 or batch_size <= 0:
        raise ValueError("orbital_count must be nonnegative and batch_size positive")
    selected = validate_selection_idxs(indices, row_count=total_count)
    counts = np.zeros(orbital_count, dtype=np.int64)
    for start in range(0, len(selected), batch_size):
        batch = selected[start : start + batch_size]
        counts += np.count_nonzero(
            _read_occupations(read_occupations, batch, orbital_count), axis=0
        )
    return counts


def complete_orbital_coverage(
    selected_indices: np.ndarray,
    *,
    read_occupations: OccupationReader,
    total_count: int,
    orbital_count: int,
    batch_size: int = 100_000,
) -> OrbitalCoverageResult:
    """Add the first source-order CSFs covering still-missing orbitals.

    Already selected CSFs are retained. No budget is imposed and no model score
    is used. Uncoverable orbitals are returned in ``missing_after`` for callers
    to report or reject; scanning to EOF is not proof of complete coverage.
    Empty initial selections are allowed. With no additions, input order is
    preserved; after additions the union is sorted by source index.
    """
    if total_count < 0:
        raise ValueError("total_count must be nonnegative")
    selected = validate_selection_idxs(selected_indices, row_count=total_count)
    counts = count_orbital_coverage(
        selected,
        read_occupations=read_occupations,
        orbital_count=orbital_count,
        total_count=total_count,
        batch_size=batch_size,
    )
    missing_before = np.flatnonzero(counts == 0).astype(np.int64)
    added: list[int] = []
    if missing_before.size:
        for candidates in iter_unselected_csf_indices(
            total_count, selected, batch_size
        ):
            masks = _read_occupations(read_occupations, candidates, orbital_count)
            for index, occupied in zip(candidates, masks, strict=True):
                if np.any(occupied & (counts == 0)):
                    added.append(int(index))
                    counts += occupied
                if np.all(counts > 0):
                    break
            if np.all(counts > 0):
                break
    additions = np.asarray(added, dtype=np.int64)
    final = np.union1d(selected, additions) if additions.size else selected.copy()
    return OrbitalCoverageResult(
        final,
        additions,
        counts,
        missing_before,
        np.flatnonzero(counts == 0).astype(np.int64),
    )
