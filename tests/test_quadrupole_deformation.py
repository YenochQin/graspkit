import math

import pytest

from graspkit.utils.quadrupole_deformation import calculate_deformation


def test_calculate_deformation_returns_named_physical_quantities() -> None:
    result = calculate_deformation(
        atomic_number=60,
        mass_number=152,
        nuclear_spin=2.0,
        spectroscopic_quadrupole_moment_barn=1.5,
    )

    assert result["intrinsic_quadrupole_moment_barn"] == pytest.approx(5.25)
    assert result["quadrupole_deformation_beta2"] == pytest.approx(0.2819, rel=1e-3)
    assert result["nuclear_radius_fm"] == pytest.approx(1.2 * math.pow(152, 1 / 3))
    assert result["spectroscopic_to_intrinsic_conversion_factor"] == pytest.approx(
        2 / 7
    )


@pytest.mark.parametrize("nuclear_spin", [0.0, 0.5, -1.0])
def test_calculate_deformation_rejects_spin_without_a_valid_conversion(
    nuclear_spin: float,
) -> None:
    with pytest.raises(ValueError, match="greater than 0.5"):
        calculate_deformation(
            atomic_number=60,
            mass_number=152,
            nuclear_spin=nuclear_spin,
            spectroscopic_quadrupole_moment_barn=1.5,
        )


@pytest.mark.parametrize(
    ("atomic_number", "mass_number"),
    [(0, 152), (-1, 152), (60, 0), (60, -152)],
)
def test_calculate_deformation_rejects_non_positive_nuclear_numbers(
    atomic_number: int,
    mass_number: int,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        calculate_deformation(
            atomic_number=atomic_number,
            mass_number=mass_number,
            nuclear_spin=2.0,
            spectroscopic_quadrupole_moment_barn=1.5,
        )
