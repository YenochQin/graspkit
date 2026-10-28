"""GRASP mixing file compatibility at the binary record boundary."""

import struct
from pathlib import Path

import numpy as np
import pytest

from graspkit.data_IO import MixCoefLoader, write_mix_coefficient_block


def _records(payloads: list[bytes]) -> bytes:
    return b"".join(
        struct.pack("=i", len(payload)) + payload + struct.pack("=i", len(payload))
        for payload in payloads
    )


@pytest.mark.parametrize("file_parity, expected_parity", [(1, 1), (-1, 2), (2, 2)])
@pytest.mark.parametrize("level_count", [0, 1])
def test_signed_grasp_parity_is_normalized_and_written_in_native_format(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    file_parity: int,
    expected_parity: int,
    level_count: int,
) -> None:
    # GRASP writes ISPAR = +/-1; the public Python model uses 1=even, 2=odd.
    path = tmp_path / "native.cm"
    payloads = [
        b"G92MIX",
        struct.pack("=6i", 2, 1, 2, level_count, level_count, 1),
        struct.pack("=5i", 1, 1, level_count, 3, file_parity),
    ]
    if level_count:
        payloads.extend(
            [
                struct.pack("=i", 2),
                struct.pack("=2d", -2.0, 0.04),
                struct.pack("=d", 1.0),
            ]
        )
    path.write_bytes(_records(payloads))

    data = MixCoefLoader(path).load()
    assert data.parity_list == [expected_parity]
    assert data.blocks[0].mix_coefficients.shape == (level_count, 1)
    display = capsys.readouterr().out
    if level_count:
        assert ("+" if expected_parity == 1 else "-") in display

    output = tmp_path / "roundtrip.cm"
    write_mix_coefficient_block(
        output, data.blocks[0], electron_count=2, orbital_count=2
    )
    native_parity = 1 if expected_parity == 1 else -1
    payloads[2] = struct.pack("=5i", 1, 1, level_count, 3, native_parity)
    assert output.read_bytes() == _records(payloads)
    assert MixCoefLoader(output).load(show_levels=False).parity_list == [
        expected_parity
    ]


def test_loader_handles_empty_block_before_populated_block_and_high_j(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "name.cm"
    path.write_bytes(
        _records(
            [
                b"G92MIX",
                struct.pack("=6i", 12, 5, 7, 1, 3, 2),
                struct.pack("=5i", 1, 2, 0, 48, 2),
                struct.pack("=5i", 2, 3, 1, 3, 1),
                struct.pack("=i", 6),
                struct.pack("=2d", -2.0, 0.04),
                struct.pack("=3d", 0.2, -0.3, 0.4),
            ]
        )
    )
    data = MixCoefLoader(path).load(show_levels=False)
    assert capsys.readouterr().out == ""
    assert data.electron_count == 12
    assert data.orbital_count == 7
    assert data.blocks[0].j_value == "47/2"
    assert data.blocks[0].mix_coefficients.shape == (0, 2)
    assert data.blocks[0].level_ids.size == 0
    assert data.blocks[1].j_value == "1"
    np.testing.assert_array_equal(data.blocks[1].level_ids, [5])
    np.testing.assert_array_equal(data.blocks[1].mix_coefficients, [[0.2, -0.3, 0.4]])
    assert data.sorted_level_energies == [-1.96]

    output = tmp_path / "name_2.cm"
    write_mix_coefficient_block(
        output, data.blocks[1], electron_count=12, orbital_count=7
    )
    assert output.read_bytes() == _records(
        [
            b"G92MIX",
            struct.pack("=6i", 12, 3, 7, 1, 3, 1),
            struct.pack("=5i", 1, 3, 1, 3, 1),
            struct.pack("=i", 6),
            struct.pack("=2d", -2.0, 0.04),
            struct.pack("=3d", 0.2, -0.3, 0.4),
        ]
    )


def test_empty_mix_file_prints_and_writes_no_level_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "empty.m"
    expected = _records(
        [
            b"G92MIX",
            struct.pack("=6i", 12, 2, 7, 0, 0, 1),
            struct.pack("=5i", 1, 2, 0, 1, 1),
        ]
    )
    path.write_bytes(expected)
    data = MixCoefLoader(path).load()
    assert "Total Levels: 0" in capsys.readouterr().out
    assert data.sorted_level_energies == []
    output = tmp_path / "empty_0.m"
    write_mix_coefficient_block(
        output, data.blocks[0], electron_count=12, orbital_count=7
    )
    assert output.read_bytes() == expected
    output.write_bytes(b"old output")
    with pytest.raises(ValueError, match="int32"):
        write_mix_coefficient_block(
            output, data.blocks[0], electron_count=-1, orbital_count=7
        )
    assert output.read_bytes() == b"old output"


@pytest.mark.parametrize(
    "fault",
    [
        "total_csfs",
        "total_levels",
        "total_size",
        "trailing",
        "negative_id",
        "truncated",
        "marker",
        "parity",
    ],
)
def test_loader_rejects_inconsistent_or_incomplete_file(
    tmp_path: Path, fault: str
) -> None:
    totals = [2, 1, 1, 1, 1, 1]
    if fault.startswith("total_"):
        totals[{"total_csfs": 1, "total_levels": 3, "total_size": 4}[fault]] = 2
    records = [
        b"G92MIX",
        struct.pack("=6i", *totals),
        struct.pack("=5i", 1, 1, 1, 1, 0 if fault == "parity" else 1),
        struct.pack("=i", 0 if fault == "negative_id" else 1),
        struct.pack("=2d", -1, 0),
        struct.pack("=d", 1),
    ]
    content = _records(records)
    if fault == "trailing":
        content += b"extra"
    elif fault == "truncated":
        content = content[:-1]
    elif fault == "marker":
        content = content[:-4] + struct.pack("=i", 7)
    path = tmp_path / "bad.m"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        MixCoefLoader(path).load(show_levels=False)
