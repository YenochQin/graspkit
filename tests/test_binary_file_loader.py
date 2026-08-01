import struct
import sys
from pathlib import Path
from typing import override

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graspkit.data_IO.loaders.binary_file_loader import (  # noqa: E402
    BinaryFileLoader,
)


class ConcreteBinaryFileLoader(BinaryFileLoader):
    @override
    def load(self) -> object:
        return object()


def _write_fortran_record(path: Path, payload: bytes) -> None:
    marker = struct.pack("i", len(payload))
    _ = path.write_bytes(marker + payload + marker)


def test_binary_loader_reads_single_type_record(tmp_path: Path) -> None:
    binary_file = tmp_path / "single.bin"
    expected = np.array([3, 5], dtype=np.int32)
    _write_fortran_record(binary_file, expected.tobytes())
    loader = ConcreteBinaryFileLoader(binary_file)

    with binary_file.open("rb") as file:
        actual = loader.read_fortran_record(file, np.int32, count=2)

    np.testing.assert_array_equal(actual, expected)


def test_binary_loader_reads_mixed_scalars_and_arrays(tmp_path: Path) -> None:
    scalar_file = tmp_path / "scalars.bin"
    scalar_payload = struct.pack("i", 7) + struct.pack("d", 1.5)
    _write_fortran_record(scalar_file, scalar_payload)

    array_file = tmp_path / "arrays.bin"
    array_payload = struct.pack("2i", 2, 4) + struct.pack("2d", 0.5, 1.5)
    _write_fortran_record(array_file, array_payload)

    loader = ConcreteBinaryFileLoader(scalar_file)
    with scalar_file.open("rb") as file:
        scalars = loader.read_mixed_scalars(file, ["i", "d"])
    with array_file.open("rb") as file:
        arrays = loader.read_mixed_arrays(file, [("i", 2), ("d", 2)])

    assert scalars == [7, 1.5]
    assert arrays == [[2, 4], [0.5, 1.5]]


def test_binary_loader_rejects_truncated_float64_field(tmp_path: Path) -> None:
    binary_file = tmp_path / "truncated.bin"
    _ = binary_file.write_bytes(struct.pack("i", 8) + b"\x00" * 4)
    loader = ConcreteBinaryFileLoader(binary_file)

    with binary_file.open("rb") as file:
        with pytest.raises(ValueError, match="Expected 8 bytes for float64"):
            _ = loader.read_mixed_scalars(file, ["d"])
