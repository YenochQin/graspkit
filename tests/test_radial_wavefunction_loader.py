from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from graspkit.data_IO import RWFNFileLoader


def _record(payload: bytes) -> bytes:
    marker = struct.pack("i", len(payload))
    return marker + payload + marker


def _write_rwfn(path: Path) -> None:
    content = bytearray(_record(b"G92RWF"))
    for n, kappa, grid in ((4, 1, (0.0, 1.0)), (4, -2, (0.0, 0.5, 1.0))):
        points = len(grid)
        content.extend(_record(struct.pack("iidi", n, kappa, -0.5, points)))
        values = (0.0, *(1.0 for _ in grid), *(0.0 for _ in grid))
        content.extend(_record(struct.pack(f"{len(values)}d", *values)))
        content.extend(_record(struct.pack(f"{points}d", *grid)))
    path.write_bytes(content)


def test_load_orbitals_preserves_kappa_and_native_grids(tmp_path: Path) -> None:
    path = tmp_path / "sample.w"
    _write_rwfn(path)

    orbitals = RWFNFileLoader(path).load_orbitals()

    assert set(orbitals) == {(4, 1), (4, -2)}
    assert orbitals[(4, 1)].energy == -0.5
    np.testing.assert_array_equal(orbitals[(4, 1)].r, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(
        orbitals[(4, -2)].r, np.array([0.0, 0.5, 1.0])
    )


def test_dataframe_load_reuses_structured_orbitals(tmp_path: Path) -> None:
    path = tmp_path / "sample.w"
    _write_rwfn(path)

    frame = RWFNFileLoader(path).load()

    assert frame.columns == ["r(a.u)", "P(4p-)", "Q(4p-)", "P(4p )", "Q(4p )"]
    assert frame.height == 3
