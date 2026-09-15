"""Derived diagnostics for orbitals loaded from GRASP G92RWF files."""

from __future__ import annotations

import math

import numpy as np

from graspkit.data_IO import RWFNOrbitalData


def radial_norm(orbital: RWFNOrbitalData) -> float:
    """Integrate ``P**2 + Q**2`` on the orbital's native grid."""

    density = orbital.p * orbital.p + orbital.q * orbital.q
    return float(np.trapezoid(density, orbital.r))


def mean_radial_radius(orbital: RWFNOrbitalData) -> float:
    """Return the density-weighted mean radius for one orbital."""

    density = orbital.p * orbital.p + orbital.q * orbital.q
    norm = radial_norm(orbital)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"non-positive radial norm for {orbital.n},{orbital.kappa}")
    return float(np.trapezoid(orbital.r * density, orbital.r)) / norm


def radial_overlap_proxy(
    anchor: RWFNOrbitalData, current: RWFNOrbitalData
) -> float:
    """Return a normalized same-grid radial overlap proxy.

    This quantity compares one-electron radial functions.  It is not a
    many-electron ASF overlap and it deliberately fails when grids or quantum
    numbers differ.
    """

    if (anchor.n, anchor.kappa) != (current.n, current.kappa):
        raise ValueError(
            "radial quantum numbers differ: "
            f"{anchor.n},{anchor.kappa} != {current.n},{current.kappa}"
        )
    points = min(len(anchor.r), len(current.r))
    if points < 2 or not np.array_equal(anchor.r[:points], current.r[:points]):
        raise ValueError(f"radial grids differ for orbital {anchor.n},{anchor.kappa}")
    anchor_norm = radial_norm(anchor)
    current_norm = radial_norm(current)
    if (
        not math.isfinite(anchor_norm)
        or not math.isfinite(current_norm)
        or anchor_norm <= 0.0
        or current_norm <= 0.0
    ):
        raise ValueError(f"non-positive radial norm for {anchor.n},{anchor.kappa}")
    product = (
        anchor.p[:points] * current.p[:points]
        + anchor.q[:points] * current.q[:points]
    )
    raw = float(np.trapezoid(product, anchor.r[:points]))
    return raw / math.sqrt(anchor_norm * current_norm)


def count_radial_nodes(orbital: RWFNOrbitalData) -> int:
    """Count sign changes in the significant large radial component."""

    maximum = float(np.max(np.abs(orbital.p), initial=0.0))
    if maximum == 0.0:
        return 0
    significant = orbital.p[np.abs(orbital.p) > maximum * 1.0e-10]
    return int(np.count_nonzero(significant[:-1] * significant[1:] < 0.0))
