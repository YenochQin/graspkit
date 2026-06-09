import numpy as np

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
