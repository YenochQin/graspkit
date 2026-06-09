from pathlib import Path

import numpy as np
import polars as pl

from graspkit.ml_module.ml_initializer import check_configuration_coupling
from graspkit.utils.data_modules import MixCoefficientBlock, MixCoefficientData


class DummyLogger:
    def info(self, *args: object, **kwargs: object) -> None:
        return None

    def error(self, *args: object, **kwargs: object) -> None:
        return None


class DummyCalPath:
    def __init__(self, cal_loop_path: Path, loop_file_name: str) -> None:
        self.cal_loop_path = cal_loop_path
        self.loop_file_name = loop_file_name


def _block(coefficients: np.ndarray) -> MixCoefficientBlock:
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    return MixCoefficientBlock(
        block_index=5,
        csf_count=coefficient_array.shape[1],
        level_count=coefficient_array.shape[0],
        j_value_location=1,
        j_value="0",
        parity=1,
        level_indices=np.arange(coefficient_array.shape[0], dtype=np.int64),
        base_energy=0.0,
        level_energies=np.arange(coefficient_array.shape[0], dtype=np.float64),
        mix_coefficients=coefficient_array,
    )


def test_check_configuration_coupling_reads_first_mix_block(tmp_path: Path) -> None:
    energy_data = pl.DataFrame(
        {
            "configuration_raw": ["term-a", "term-b", "term-c"],
            "energy": [0.0, 1.0, 2.0],
        }
    )
    mix_data = MixCoefficientData(
        blocks=[
            _block(
                np.array(
                    [
                        [0.5, 0.1],
                        [0.2, 0.3],
                        [0.4, 0.0],
                    ]
                )
            )
        ],
        level_list=[0.0, 1.0, 2.0],
    )

    ok, selected_energy_data, correct_levels_ci = check_configuration_coupling(
        paths_cfg=DummyCalPath(tmp_path, "loop"),
        energy_level_data=energy_data,
        rmix_file_data=mix_data,
        spectral_term=["term-b", "term-c"],
        cal_loop_num=1,
        logger=DummyLogger(),
    )

    assert ok is True
    assert selected_energy_data is not None
    assert selected_energy_data["configuration_raw"].to_list() == ["term-b", "term-c"]
    np.testing.assert_allclose(
        correct_levels_ci,
        np.array([[0.2, 0.3], [0.4, 0.0]]),
    )
    assert (tmp_path / "loop_correct_levels.csv").is_file()
