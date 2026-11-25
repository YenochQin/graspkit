"""Write one standalone block of mixing coefficients."""

import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

from ..utils.data_modules import MixCoefficientBlock
from . import mixing_stream_layout as stream_layout


def _write_record(file: BinaryIO, data: NDArray[np.generic]) -> None:
    marker = struct.pack("=i", data.nbytes)
    _ = file.write(marker)
    data.tofile(file)
    _ = file.write(marker)


def write_mix_coefficient_block(
    file_path: str | Path,
    block: MixCoefficientBlock,
    *,
    electron_count: int,
    orbital_count: int,
    basis_id: str | None = None,
) -> Path:
    """Write a Suite stream if bound to a descriptor, otherwise write G92MIX.

    The output has one block numbered 1. GRASP level IDs are restored to
    one-based values; energies and coefficient rows retain their input order.
    As in rasfsplit, a block with no levels has no following data records.
    Uses native byte order, int32 record markers/integers, and float64 data,
    matching :class:`MixCoefLoader`. An existing output is overwritten.
    Python parity (1=even, 2=odd) is written as GRASP ISPAR (+1/-1).
    """
    output = Path(file_path)
    if basis_id is not None:
        return _write_stream_block(output, block, electron_count, orbital_count, basis_id)
    coefficient_count = block.csf_count * block.level_count
    counts = (
        electron_count,
        block.csf_count,
        orbital_count,
        block.level_count,
        coefficient_count,
        block.j_value_location,
    )
    if any(value < 0 or value > np.iinfo(np.int32).max for value in counts):
        raise ValueError("Mix coefficient counts must fit non-negative int32")
    if block.csf_count < 1 or block.j_value_location < 1 or block.parity not in (1, 2):
        raise ValueError("Invalid mix coefficient block header")
    if np.any(block.level_ids < 0) or np.any(block.level_ids >= np.iinfo(np.int32).max):
        raise ValueError("GRASP level IDs must fit positive int32")
    if max(coefficient_count * 8, (block.level_count + 1) * 8) > np.iinfo(np.int32).max:
        raise ValueError("Mix coefficient record exceeds the int32 marker size")

    with output.open("wb") as file:
        _write_record(file, np.frombuffer(b"G92MIX", dtype=np.uint8))
        _write_record(
            file,
            np.array(
                [
                    electron_count,
                    block.csf_count,
                    orbital_count,
                    block.level_count,
                    coefficient_count,
                    1,
                ],
                dtype=np.int32,
            ),
        )
        _write_record(
            file,
            np.array(
                [
                    1,
                    block.csf_count,
                    block.level_count,
                    block.j_value_location,
                    1 if block.parity == 1 else -1,
                ],
                dtype=np.int32,
            ),
        )
        if block.level_count:
            _write_record(file, (block.level_ids + 1).astype(np.int32))
            _write_record(
                file,
                np.concatenate((np.array([block.base_energy]), block.level_energies)),
            )
            _write_record(file, np.asarray(block.mix_coefficients, dtype=np.float64))
    return output


def _write_stream_block(
    path: Path,
    block: MixCoefficientBlock,
    electrons: int,
    orbitals: int,
    basis_id: str,
) -> Path:
    if len(basis_id) != 64 or any(char not in "0123456789abcdef" for char in basis_id):
        raise ValueError("Invalid mixing stream basis digest")
    csfs, levels = block.csf_count, block.level_count
    if (
        min(electrons, orbitals, levels, block.j_value_location) < 1
        or levels > csfs
        or block.parity not in (1, 2)
        or np.any(block.level_ids < 0)
        or np.any(block.level_ids >= csfs)
    ):
        raise ValueError("Invalid Suite mixing block")
    table = np.empty(levels, dtype=stream_layout.LEVEL_DTYPE)
    table["root"] = block.level_ids + 1
    table["energy"] = block.base_energy + block.level_energies
    if not np.isfinite(table["energy"]).all():
        raise ValueError("Nonfinite ASF energy")
    offset = stream_layout.HEADER.size + stream_layout.DIRECTORY_RECORD.size + table.nbytes
    header = stream_layout.HEADER.pack(
        stream_layout.SIGNATURE, electrons, orbitals, 1, csfs, levels, basis_id.encode("ascii")
    )
    directory = stream_layout.DIRECTORY_RECORD.pack(
        block.j_value_location - 1, 1 if block.parity == 1 else -1, csfs, levels, 0, offset
    )
    with path.open("wb") as file:
        _ = file.write(header)
        _ = file.write(directory)
        table.tofile(file)
        np.asarray(block.mix_coefficients, dtype="<f8").tofile(file)
    return path
