# -*- encoding: utf-8 -*-
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MixCoefficientBlock:
    """Parsed mixing coefficients and metadata for one rmix block.

    ``level_ids`` contains GRASP identifiers from ``ivec``. It is metadata,
    not a matrix index. Rows of ``level_energies`` and ``mix_coefficients``
    are addressed only by zero-based ``asf_row_indices``.
    """

    block_index: int
    csf_count: int
    level_count: int
    j_value_location: int
    j_value: str
    parity: int
    level_ids: NDArray[np.int64]
    base_energy: float
    level_energies: NDArray[np.float64]
    mix_coefficients: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.level_ids.shape != (self.level_count,):
            raise ValueError("level_ids shape must match level_count")
        if len(np.unique(self.level_ids)) != self.level_count:
            raise ValueError("level_ids must be unique within a block")
        if self.level_energies.shape != (self.level_count,):
            raise ValueError("level_energies shape must match level_count")
        if self.mix_coefficients.shape != (self.level_count, self.csf_count):
            raise ValueError(
                "mix_coefficients shape must be (level_count, csf_count)"
            )

    @property
    def asf_row_indices(self) -> NDArray[np.int64]:
        """Return the matrix row indices for all ASFs in this block."""
        return np.arange(self.level_count, dtype=np.int64)

    def row_for_level_id(self, level_id: int) -> int:
        """Resolve one GRASP level ID to its ASF matrix row."""
        matches = np.flatnonzero(self.level_ids == level_id)
        if matches.size == 0:
            raise ValueError(
                f"Block {self.block_index}: unknown level ID {level_id}"
            )
        return int(matches[0])


@dataclass(frozen=True)
class MixCoefficientData:
    """Container for parsed ASF mixing coefficients grouped by block.

    Attributes:
        blocks: Parsed block-level rmix data.
        sorted_level_energies: Absolute level energies sorted ascending.
    """

    blocks: list[MixCoefficientBlock]
    sorted_level_energies: list[float]

    @property
    def block_num(self) -> int:
        return len(self.blocks)

    @property
    def block_idx_list(self) -> list[int]:
        return [block.block_index for block in self.blocks]

    @property
    def block_CSFs_nums(self) -> list[int]:
        return [block.csf_count for block in self.blocks]

    @property
    def block_energy_count_list(self) -> list[int]:
        return [block.level_count for block in self.blocks]

    @property
    def level_J_value_list(self) -> list[str]:
        return [block.j_value for block in self.blocks]

    @property
    def parity_list(self) -> list[int]:
        return [block.parity for block in self.blocks]

    @property
    def block_level_ids_list(self) -> list[NDArray[np.int64]]:
        return [block.level_ids for block in self.blocks]

    @property
    def block_energy_list(self) -> list[float]:
        return [block.base_energy for block in self.blocks]

    @property
    def block_level_energy_list(self) -> list[NDArray[np.float64]]:
        return [block.level_energies for block in self.blocks]

    @property
    def mix_coefficient_list(self) -> list[NDArray[np.float64]]:
        return [block.mix_coefficients for block in self.blocks]
