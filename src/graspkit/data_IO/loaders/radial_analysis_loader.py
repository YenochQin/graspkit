"""Strict input adapter for radial post-processing; never edits input arrays."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from numpy.typing import NDArray

from .radial_wavefunction_loader import RWFNFileLoader

FloatArray = NDArray[np.float64]
LETTERS = "spdfghiklmnoqrtuvwxyz"


def parse_label(label: str) -> tuple[str, int, int]:
    """GRASP labels: 2p- has kappa=1, 2p (or 2p+) has kappa=-2."""
    match = re.fullmatch(r"(\d+)([a-z])([-+]?)", label.strip())
    if match is None or match[2] not in LETTERS:
        raise ValueError(f"Invalid GRASP orbital label: {label!r}")
    n, l = int(match[1]), LETTERS.index(match[2])
    if n <= l or (l == 0 and match[3] == "-"):
        raise ValueError(f"Invalid quantum numbers in label: {label!r}")
    minus = match[3] == "-"
    return f"{n}{LETTERS[l]}{'-' if minus else ''}", n, l if minus else -l - 1


@dataclass(frozen=True)
class RadialOrbital:
    label: str
    n: int
    kappa: int
    r: FloatArray
    p: FloatArray
    q: FloatArray
    source: Path
    provenance: str = "native"

    def validate(self) -> None:
        if not (self.r.ndim == self.p.ndim == self.q.ndim == 1):
            raise ValueError(f"{self.label}: arrays must be one-dimensional")
        if len(self.r) < 3 or not len(self.r) == len(self.p) == len(self.q):
            raise ValueError(f"{self.label}: unequal lengths or fewer than 3 points")
        if not all(bool(np.isfinite(a).all()) for a in (self.r, self.p, self.q)):
            raise ValueError(f"{self.label}: non-finite data")
        if self.r[0] < 0 or not bool((np.diff(self.r) > 0).all()):
            raise ValueError(
                f"{self.label}: negative radius, duplicate or non-increasing grid"
            )
        if self.kappa == 0 or self.n <= (
            self.kappa if self.kappa > 0 else -self.kappa - 1
        ):
            raise ValueError(f"{self.label}: invalid n/kappa")


def load_radial_input(
    path: Path,
    *,
    radius_scale: float = 1.0,
    amplitude_scale: float = 1.0,
    radius_column: str = "r(a.u)",
) -> list[RadialOrbital]:
    """Convert explicitly declared units to bohr and bohr**(-1/2).

    Defaults implement the existing read_radial_wavefunction.py CSV contract.
    Native binary support is the existing native-endian, 4-byte record format.
    CSV trailing zeros are preserved: their provenance cannot be recovered.
    """
    if not all(np.isfinite(v) and v > 0 for v in (radius_scale, amplitude_scale)):
        raise ValueError("Unit conversion factors must be finite and positive")
    path = path.expanduser().resolve()
    result: list[RadialOrbital] = []
    if path.suffix.lower() != ".csv":
        if (radius_scale, amplitude_scale) != (1.0, 1.0):
            raise ValueError("rwfn already uses atomic units; unit scaling is CSV-only")
        for (n, kappa), orbital in RWFNFileLoader(path).load_orbitals().items():
            l = kappa if kappa > 0 else -kappa - 1
            if not 0 <= l < len(LETTERS):
                raise ValueError(f"Unsupported kappa: {kappa}")
            label = f"{n}{LETTERS[l]}{'-' if kappa > 0 else ''}"
            result.append(
                RadialOrbital(label, n, kappa, orbital.r, orbital.p, orbital.q, path)
            )
    else:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader, list[str]())
            if len(header) != len(set(header)):
                raise ValueError(f"Duplicate CSV columns in {path}")
            if header.count(radius_column) != 1:
                raise ValueError(f"Expected radius column {radius_column!r}")
            rows = list(reader)
        if not rows or any(len(row) != len(header) for row in rows):
            raise ValueError("Empty CSV or inconsistent row lengths")
        data = np.asarray(rows, dtype=np.float64)
        columns: dict[tuple[str, str], int] = {}
        labels: dict[str, tuple[int, int]] = {}
        for index, name in enumerate(header):
            if name == radius_column:
                continue
            match = re.fullmatch(r"([PQ])\(([^)]+)\)", name.strip())
            if match is None:
                raise ValueError(
                    f"Unrecognized column (meaning/units unknown): {name!r}"
                )
            label, n, kappa = parse_label(match[2])
            key = (label, match[1])
            if key in columns:
                raise ValueError(f"Duplicate canonical orbital column: {key}")
            columns[key] = index
            labels[label] = n, kappa
        for label, (n, kappa) in labels.items():
            if (label, "P") not in columns or (label, "Q") not in columns:
                raise ValueError(f"Missing P/Q component for {label}")
            result.append(
                RadialOrbital(
                    label,
                    n,
                    kappa,
                    data[:, header.index(radius_column)] * radius_scale,
                    data[:, columns[label, "P"]] * amplitude_scale,
                    data[:, columns[label, "Q"]] * amplitude_scale,
                    path,
                    "csv_shared_grid_native_lengths_and_export_precision_unknown",
                )
            )
    if not result:
        raise ValueError(f"No orbitals in {path}")
    for orbital in result:
        orbital.validate()
    return result
