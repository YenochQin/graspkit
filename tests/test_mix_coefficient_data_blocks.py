from pathlib import Path
from typing import Any

import numpy as np
import pytest

from graspkit.data_IO.loaders.mix_coef_loader import MixCoefLoader

from graspkit.utils.data_modules import MixCoefficientBlock, MixCoefficientData


def test_mix_coefficient_data_derives_legacy_lists_from_blocks() -> None:
    block0 = MixCoefficientBlock(
        block_index=0,
        csf_count=3,
        level_count=2,
        j_value_location=1,
        j_value="0",
        parity=1,
        level_indices=np.array([0, 1], dtype=np.int64),
        base_energy=0.5,
        level_energies=np.array([0.0, 0.1], dtype=np.float64),
        mix_coefficients=np.array(
            [
                [0.5, 0.1, 0.2],
                [0.0, 0.4, 0.1],
            ],
            dtype=np.float64,
        ),
    )
    block1 = MixCoefficientBlock(
        block_index=1,
        csf_count=2,
        level_count=1,
        j_value_location=2,
        j_value="1/2",
        parity=2,
        level_indices=np.array([0], dtype=np.int64),
        base_energy=0.8,
        level_energies=np.array([0.05], dtype=np.float64),
        mix_coefficients=np.array([[0.3, 0.4]], dtype=np.float64),
    )

    data = MixCoefficientData(
        blocks=[block0, block1],
        level_list=[0.5, 0.6, 0.85],
    )

    assert data.block_num == 2
    assert data.block_idx_list == [0, 1]
    assert data.block_CSFs_nums == [3, 2]
    assert data.block_energy_count_list == [2, 1]
    assert data.level_J_value_list == ["0", "1/2"]
    assert data.parity_list == [1, 2]
    assert data.block_energy_list == [0.5, 0.8]
    np.testing.assert_array_equal(data.block_levels_idx_list[0], np.array([0, 1]))
    np.testing.assert_allclose(data.block_level_energy_list[1], np.array([0.05]))
    np.testing.assert_allclose(
        data.mix_coefficient_list[0],
        np.array([[0.5, 0.1, 0.2], [0.0, 0.4, 0.1]]),
    )


def test_mix_coef_loader_builds_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records = iter(
        [
            np.frombuffer(b"G92MIX", dtype="S1"),
            np.array([2, 3, 4, 5, 6, 1], dtype=np.int32),
            np.array([1, 3, 2, 1, 1], dtype=np.int32),
            np.array([1, 2], dtype=np.int32),
            np.array([0.5, 0.0, 0.1], dtype=np.float64),
            np.array([0.5, 0.1, 0.2, 0.0, 0.4, 0.1], dtype=np.float64),
        ]
    )

    def fake_read_fortran_record(
        self: MixCoefLoader,
        file: Any,
        dtype: str,
        count: int,
    ) -> np.ndarray:
        return next(records)

    def fake_read_mixed_scalars(
        self: MixCoefLoader,
        file: Any,
        field_specs: list[str],
    ) -> np.ndarray:
        return next(records)

    printed: list[MixCoefficientData] = []
    monkeypatch.setattr(MixCoefLoader, "read_fortran_record", fake_read_fortran_record)
    monkeypatch.setattr(MixCoefLoader, "read_mixed_scalars", fake_read_mixed_scalars)
    monkeypatch.setattr(
        "graspkit.data_IO.loaders.mix_coef_loader.print_mix_coef_levels_rich",
        lambda data: printed.append(data),
    )

    mix_path = tmp_path / "example.m"
    mix_path.write_bytes(b"dummy")

    result = MixCoefLoader(mix_path).load()

    assert len(result.blocks) == 1
    block = result.blocks[0]
    assert block.block_index == 0
    assert block.csf_count == 3
    assert block.level_count == 2
    assert block.j_value_location == 1
    assert block.j_value == "0"
    assert block.parity == 1
    assert block.base_energy == 0.5
    np.testing.assert_array_equal(block.level_indices, np.array([0, 1]))
    np.testing.assert_allclose(block.level_energies, np.array([0.0, 0.1]))
    np.testing.assert_allclose(
        block.mix_coefficients,
        np.array([[0.5, 0.1, 0.2], [0.0, 0.4, 0.1]]),
    )
    assert printed == [result]
