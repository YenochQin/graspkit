"""Exact jj-CSF MR selection, independent of CLI and CSF reader backends.

Targets use zero-based block indices and the loader's ``ivec - 1`` level IDs.
The caller supplies an in-memory document read without truncation, and remains
responsible for file snapshots/provenance. Structural agreement cannot establish
historical pairing: legacy mixing files contain no CSF-content digest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ..utils.data_modules import MixCoefficientData
from .extraction import CsfDocument, create_csf_document
from .selection import select_csf_indices_by_ci_squared_cutoff

# GRASP orbital lettering omits j. This parses occupation metadata only; the
# original coupling records are never reconstructed from these labels.
_LETTERS = "spdfghiklmnoqrtuvwxyz"
_ORBITAL = re.compile(r"([1-9]\d*)([" + _LETTERS + r"])(-?)")
_OCCUPATION = re.compile(r"([1-9]\d*[" + _LETTERS + r"]-?)\s*\(\s*(\d+)\)")
_TOTAL_J = re.compile(r"(?:^|\s)(\d+(?:/2)?)\s*([+-])\s*$")
NORM_TOLERANCE = 1e-8


@dataclass(frozen=True)
class MrTarget:
    """A source block index and loader level ID, both zero-based."""

    block_index: int
    level_id: int


@dataclass(frozen=True)
class MrSelection:
    """Selected original records and source-indexed, JSON-compatible audit rows."""

    document: CsfDocument
    selected_rows: list[dict[str, object]]
    configuration_families: list[dict[str, object]]
    target_summaries: list[dict[str, object]]
    block_mapping: list[dict[str, int]]
    cutoff_mode: str
    cutoff: float
    omitted_core_subshells: tuple[str, ...]
    source_csf_count: int

    def summary(self) -> dict[str, object]:
        """Return numerical checks and index conventions for an audit manifest."""
        source_records = {
            int(cast(int, row["source_row_index_0based"])): {
                name: row[name]
                for name in (
                    "source_block_1based",
                    "output_block_1based",
                    "csf_index_in_block_1based",
                    "source_row_index_0based",
                )
            }
            for row in self.selected_rows
        }
        return {
            "threshold": {
                "mode": self.cutoff_mode,
                "value": self.cutoff,
                "comparison": ">",
            },
            "source_csf_count": self.source_csf_count,
            "selected_csf_count": self.document.csfs_df.height,
            "selected_family_count": len(
                {row["configuration_family"] for row in self.configuration_families}
            ),
            "targets": self.target_summaries,
            "block_mapping": self.block_mapping,
            "selected_source_records": [
                source_records[index] for index in sorted(source_records)
            ],
            "omitted_closed_core_subshells": list(self.omitted_core_subshells),
            "family_definition": "Sum jj subshell occupations at fixed n,l; omit only header core. Reporting only, not a coupling specification.",
            "checks": {
                "block_dimensions_and_order": "passed",
                "j_parity_all_records": "passed",
                "electrons_all_records": "passed",
                "header_orbital_count": "passed",
                "coefficients_finite": "passed",
                "target_norms": "passed",
                "norm_absolute_tolerance": NORM_TOLERANCE,
            },
            "index_conventions": {
                "source_block_1based": "original file block order",
                "level_id_1based": "original GRASP ivec, not matrix row",
                "asf_row_index_0based": "resolved using row_for_level_id(ivec - 1)",
                "source_row_index_0based": "sum of preceding source block lengths plus local index; ignores dataframe idx",
                "csf_index_in_block_1based": "original CSF position in source block",
            },
            "interpretation": "Weights describe the original expansion, not a re-solved wavefunction or an energy/HFS/isotope-shift error bound.",
        }


def _orbital_capacity(label: str) -> int:
    match = _ORBITAL.fullmatch(label)
    if match is None:
        raise ValueError(f"Invalid relativistic subshell: {label!r}")
    angular = _LETTERS.index(match[2])
    if int(match[1]) <= angular or (angular == 0 and match[3]):
        raise ValueError(f"Invalid relativistic subshell: {label!r}")
    return 2 * angular if match[3] else 2 * angular + 2


def _occupations(line: str, peel: set[str]) -> dict[str, int]:
    matches = list(_OCCUPATION.finditer(line))
    if not matches or _OCCUPATION.sub("", line).strip():
        raise ValueError(f"Malformed CSF occupation line: {line!r}")
    occupations: dict[str, int] = {}
    for match in matches:
        label, count = match[1], int(match[2])
        if label not in peel or label in occupations:
            raise ValueError(f"Unknown or duplicate peel subshell: {label}")
        if not 1 <= count <= _orbital_capacity(label):
            raise ValueError(f"Invalid occupation for {label}: {count}")
        occupations[label] = count
    return occupations


def _family(occupations: dict[str, int]) -> str:
    merged: dict[str, int] = {}
    for label, count in occupations.items():
        nonrelativistic = label.rstrip("-")
        merged[nonrelativistic] = merged.get(nonrelativistic, 0) + count
    labels = sorted(
        merged, key=lambda label: (int(label[:-1]), _LETTERS.index(label[-1]))
    )
    return " ".join(f"{label}({merged[label]})" for label in labels)


def _validate_pair(
    document: CsfDocument, block_lengths: Sequence[int], mixing: MixCoefficientData
) -> list[str]:
    frame = document.csfs_df
    if list(block_lengths) != [block.csf_count for block in mixing.blocks]:
        raise ValueError("CSF/mixing block dimensions disagree")
    if sum(block_lengths) != frame.height or "block_id" not in frame.columns:
        raise ValueError("CSF row count or block_id missing/inconsistent")
    if [document.header_lines[i].strip() for i in (0, 2, 4)] != [
        "Core subshells:",
        "Peel subshells:",
        "CSF(s):",
    ]:
        raise ValueError("Invalid CSF header labels")
    core = document.header_lines[1].split()
    peel = document.header_lines[3].split()
    peel_set = set(peel)
    orbitals = core + peel
    if len(set(orbitals)) != len(orbitals):
        raise ValueError("Duplicate subshell in CSF header")
    core_electrons = sum(_orbital_capacity(label) for label in core)
    for label in peel:
        _orbital_capacity(label)
    if mixing.orbital_count != len(orbitals):
        raise ValueError("CSF header/mixing orbital count mismatch")
    families: list[str] = []
    offset = 0
    for block_index, block in enumerate(mixing.blocks):
        if block.block_index != block_index:
            raise ValueError("Mixing block order mismatch")
        if np.any(block.level_ids < 0):
            raise ValueError("Negative loader level ID")
        if not np.all(np.isfinite(block.mix_coefficients)):
            raise ValueError("Non-finite mixing coefficient")
        if not math.isfinite(block.base_energy) or not np.all(
            np.isfinite(block.level_energies)
        ):
            raise ValueError("Non-finite mixing energy")
        rows = cast(
            list[tuple[int, str, str]],
            frame.slice(offset, block.csf_count)
            .select("block_id", "line1", "line3")
            .rows(),
        )
        for source_block, line1, line3 in rows:
            if source_block != block_index:
                raise ValueError("CSF block_id disagrees with source block offsets")
            occupations = _occupations(line1, peel_set)
            if core_electrons + sum(occupations.values()) != mixing.electron_count:
                raise ValueError("CSF/mixing electron count mismatch")
            match = _TOTAL_J.search(line3)
            if match is None:
                raise ValueError(f"Missing total J/parity in coupling line: {line3!r}")
            two_j = 2 * Fraction(match[1])
            parity = 1 if match[2] == "+" else 2
            occupation_parity = (
                sum(
                    _LETTERS.index(label.rstrip("-")[-1]) * count
                    for label, count in occupations.items()
                )
                % 2
                + 1
            )
            if (
                two_j != block.j_value_location - 1
                or parity != block.parity
                or parity != occupation_parity
            ):
                raise ValueError("CSF/mixing J/parity mismatch")
            families.append(_family(occupations))
        offset += block.csf_count
    return families


def select_mr_csfs(
    document: CsfDocument,
    block_lengths: Sequence[int],
    mixing: MixCoefficientData,
    targets: Sequence[MrTarget],
    *,
    amplitude_cutoff: float | None = None,
    weight_cutoff: float | None = None,
) -> MrSelection:
    """Validate pairing structurally and select the per-target strict-cutoff union.

    No normalization, cumulative weighting, configuration expansion, file writes
    or physical convergence claims are performed. Requested empty blocks fail.
    The document must come from a strict, non-truncating reader.
    """
    if (amplitude_cutoff is None) == (weight_cutoff is None):
        raise ValueError("Specify exactly one amplitude_cutoff or weight_cutoff")
    cutoff = (
        amplitude_cutoff if amplitude_cutoff is not None else cast(float, weight_cutoff)
    )
    if not math.isfinite(cutoff) or not 0 <= cutoff < 1:
        raise ValueError("Cutoff must be finite and within [0, 1)")
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Targets must be nonempty and unique")
    requested: dict[int, list[int]] = {}
    for target in targets:
        if (
            type(target.block_index) is not int
            or type(target.level_id) is not int
            or target.level_id < 0
            or not 0 <= target.block_index < mixing.block_num
        ):
            raise ValueError(f"Invalid target: {target}")
        block = mixing.blocks[target.block_index]
        row = block.row_for_level_id(target.level_id)
        requested.setdefault(target.block_index, []).append(row)
    families = _validate_pair(document, block_lengths, mixing)
    source_records = document.csfs_df.select("line1", "line2", "line3")
    selected_global: list[int] = []
    selected_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    mapping: list[dict[str, int]] = []
    offset = 0
    for block in mixing.blocks:
        block_index = block.block_index
        if block_index not in requested:
            offset += block.csf_count
            continue
        rows = requested[block_index]
        coefficients = block.mix_coefficients[rows]
        squared = np.square(coefficients)
        norms = np.sum(squared, axis=1)
        if np.any(np.abs(norms - 1.0) > NORM_TOLERANCE):
            raise ValueError(
                f"Block {block_index + 1}: target ASF norm differs from 1 by more than {NORM_TOLERANCE}"
            )
        triggered = (
            (np.abs(coefficients) > cutoff)
            if amplitude_cutoff is not None
            else (squared > cutoff)
        )
        selected: NDArray[np.int64]
        if weight_cutoff is not None:
            selected = select_csf_indices_by_ci_squared_cutoff(
                MixCoefficientData(blocks=[block], sorted_level_energies=[]),
                asf_row_indices=[rows],
                ci_squared_cutoff=cutoff,
            )[block_index]
        else:
            selected = np.flatnonzero(np.any(triggered, axis=0)).astype(np.int64)
        if selected.size == 0:
            raise ValueError(f"Block {block_index + 1}: empty selection")
        output_block = len(mapping) + 1
        mapping.append(
            {
                "source_block_1based": block_index + 1,
                "output_block_1based": output_block,
                "source_csf_count": block.csf_count,
                "selected_csf_count": int(selected.size),
            }
        )
        selected_global.extend(offset + int(index) for index in selected)
        block_families = families[offset : offset + block.csf_count]
        family_all: dict[str, list[int]] = {}
        family_kept: dict[str, list[int]] = {}
        for index, family in enumerate(block_families):
            family_all.setdefault(family, []).append(index)
        for index in selected:
            local = int(index)
            family_kept.setdefault(block_families[local], []).append(local)
        for position, row in enumerate(rows):
            level = int(block.level_ids[row]) + 1
            identity: dict[str, object] = {
                "source_block_1based": block_index + 1,
                "output_block_1based": output_block,
                "level_id_1based": level,
                "loader_level_id_0based": level - 1,
                "asf_row_index_0based": row,
                "j": block.j_value,
                "parity": "+" if block.parity == 1 else "-",
            }
            kept = float(squared[position, selected].sum())
            summaries.append(
                {
                    **identity,
                    "energy_hartree": float(
                        block.base_energy + block.level_energies[row]
                    ),
                    "weight_all": float(norms[position]),
                    "triggered_csf_count": int(triggered[position].sum()),
                    "weight_triggered": float(
                        squared[position, triggered[position]].sum()
                    ),
                    "weight_kept": kept,
                    "weight_omitted": float(norms[position]) - kept,
                }
            )
            for index in selected:
                local = int(index)
                source = offset + local
                record = cast(
                    tuple[str, str, str],
                    source_records.row(source),
                )
                selected_rows.append(
                    {
                        **identity,
                        "csf_index_in_block_1based": local + 1,
                        "source_row_index_0based": source,
                        "coefficient": float(coefficients[position, local]),
                        "squared_weight": float(squared[position, local]),
                        "triggered_selection": bool(triggered[position, local]),
                        "triggering_level_ids_1based": ";".join(
                            str(int(block.level_ids[rows[p]]) + 1)
                            for p in range(len(rows))
                            if triggered[p, local]
                        ),
                        "configuration_family": block_families[local],
                        **dict(zip(("line1", "line2", "line3"), record, strict=True)),
                    }
                )
            for family, kept_indices in family_kept.items():
                all_indices = family_all[family]
                family_rows.append(
                    {
                        **identity,
                        "configuration_family": family,
                        "source_csf_count": len(all_indices),
                        "selected_csf_count": len(kept_indices),
                        "family_weight_all": float(
                            squared[position, all_indices].sum()
                        ),
                        "family_weight_selected": float(
                            squared[position, kept_indices].sum()
                        ),
                    }
                )
        offset += block.csf_count
    return MrSelection(
        document=create_csf_document(
            header_lines=document.header_lines,
            csfs_df=document.csfs_df[selected_global],
            source_path=document.source_path,
        ),
        selected_rows=selected_rows,
        configuration_families=family_rows,
        target_summaries=summaries,
        block_mapping=mapping,
        cutoff_mode="amplitude" if amplitude_cutoff is not None else "squared_weight",
        cutoff=cutoff,
        omitted_core_subshells=tuple(document.header_lines[1].split()),
        source_csf_count=document.csfs_df.height,
    )
