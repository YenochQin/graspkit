"""Nuclear quadrupole deformation calculations."""

from __future__ import annotations

import math


NUCLEAR_RADIUS_COEFFICIENT_FM = 1.2
BARN_TO_SQUARE_FEMTOMETER = 100.0


def calculate_deformation(
    atomic_number: int,
    mass_number: int,
    nuclear_spin: float,
    spectroscopic_quadrupole_moment_barn: float,
) -> dict[str, float]:
    """Calculate nuclear intrinsic quadrupole moment and deformation parameter.

    Args:
        atomic_number: Proton count.
        mass_number: Nuclear mass number.
        nuclear_spin: Nuclear spin quantum number; must be greater than 0.5.
        spectroscopic_quadrupole_moment_barn: Spectroscopic quadrupole moment,
            in barns.

    Returns:
        Intrinsic quadrupole moment, deformation parameter, nuclear radius, and
        the spectroscopic-to-intrinsic conversion factor.

    Raises:
        ValueError: If a nuclear number is non-positive or the spin cannot
            produce a finite quadrupole conversion factor.
    """
    if atomic_number <= 0:
        raise ValueError("atomic_number must be positive")
    if mass_number <= 0:
        raise ValueError("mass_number must be positive")
    if nuclear_spin <= 0.5:
        raise ValueError(
            "nuclear_spin must be greater than 0.5 for quadrupole deformation"
        )

    spectroscopic_quadrupole_moment_fm2 = (
        spectroscopic_quadrupole_moment_barn * BARN_TO_SQUARE_FEMTOMETER
    )
    nuclear_radius_fm = NUCLEAR_RADIUS_COEFFICIENT_FM * mass_number ** (1 / 3)
    nuclear_radius_squared_fm2 = nuclear_radius_fm**2
    spectroscopic_to_intrinsic_conversion_factor = (
        nuclear_spin * (2 * nuclear_spin - 1)
    ) / ((nuclear_spin + 1) * (2 * nuclear_spin + 3))
    intrinsic_quadrupole_moment_fm2 = (
        spectroscopic_quadrupole_moment_fm2
        / spectroscopic_to_intrinsic_conversion_factor
    )
    intrinsic_quadrupole_moment_barn = (
        intrinsic_quadrupole_moment_fm2 / BARN_TO_SQUARE_FEMTOMETER
    )
    quadrupole_deformation_beta2 = (
        math.sqrt(5 * math.pi)
        * intrinsic_quadrupole_moment_fm2
        / (3 * atomic_number * nuclear_radius_squared_fm2)
    )

    return {
        "intrinsic_quadrupole_moment_barn": intrinsic_quadrupole_moment_barn,
        "quadrupole_deformation_beta2": quadrupole_deformation_beta2,
        "nuclear_radius_fm": nuclear_radius_fm,
        "spectroscopic_to_intrinsic_conversion_factor": (
            spectroscopic_to_intrinsic_conversion_factor
        ),
    }
