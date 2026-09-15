from __future__ import annotations

import numpy as np
import pytest

from graspkit.data_IO import RWFNOrbitalData
from graspkit.grasp_data_extractor import (
    count_radial_nodes,
    mean_radial_radius,
    radial_norm,
    radial_overlap_proxy,
)


def _orbital(
    p: tuple[float, ...],
    *,
    n: int = 4,
    kappa: int = -2,
    grid: tuple[float, ...] = (0.0, 0.5, 1.0),
) -> RWFNOrbitalData:
    return RWFNOrbitalData(
        n=n,
        kappa=kappa,
        energy=-0.5,
        a0=0.0,
        p=np.asarray(p, dtype=np.float64),
        q=np.zeros(len(p), dtype=np.float64),
        r=np.asarray(grid, dtype=np.float64),
    )


def test_radial_diagnostics_use_native_orbital_data() -> None:
    orbital = _orbital((0.0, 1.0, 0.0))

    assert radial_norm(orbital) == pytest.approx(0.5)
    assert mean_radial_radius(orbital) == pytest.approx(0.5)
    assert radial_overlap_proxy(orbital, orbital) == pytest.approx(1.0)
    assert count_radial_nodes(_orbital((1.0, -1.0, 1.0))) == 2


def test_radial_overlap_proxy_rejects_incompatible_inputs() -> None:
    anchor = _orbital((0.0, 1.0, 0.0))

    with pytest.raises(ValueError, match="quantum numbers differ"):
        radial_overlap_proxy(anchor, _orbital((0.0, 1.0, 0.0), kappa=1))
    with pytest.raises(ValueError, match="radial grids differ"):
        radial_overlap_proxy(
            anchor,
            _orbital(
                (0.0, 1.0, 0.0),
                grid=(0.0, 0.25, 1.0),
            ),
        )


def test_radial_overlap_proxy_accepts_a_shared_grid_prefix() -> None:
    anchor = _orbital((0.0, 1.0, 0.0))
    current = _orbital(
        (0.0, 1.0, 0.0, 0.0),
        grid=(0.0, 0.5, 1.0, 1.5),
    )

    assert radial_overlap_proxy(anchor, current) == pytest.approx(1.0)
