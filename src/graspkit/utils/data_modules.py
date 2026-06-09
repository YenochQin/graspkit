# -*- encoding: utf-8 -*-
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MixCoefficientBlock:
    """Parsed mixing coefficients and metadata for one rmix block."""

    block_index: int
    csf_count: int
    level_count: int
    j_value_location: int
    j_value: str
    parity: int
    level_indices: NDArray[np.int64]
    base_energy: float
    level_energies: NDArray[np.float64]
    mix_coefficients: NDArray[np.float64]


@dataclass(frozen=True)
class MixCoefficientData:
    """Container for parsed ASF mixing coefficients grouped by block.

    Attributes:
        blocks: Parsed block-level rmix data.
        level_list: Flat list of level identifiers.
    """

    blocks: list[MixCoefficientBlock]
    level_list: list[float]

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
    def block_levels_idx_list(self) -> list[NDArray[np.int64]]:
        return [block.level_indices for block in self.blocks]

    @property
    def block_energy_list(self) -> list[float]:
        return [block.base_energy for block in self.blocks]

    @property
    def block_level_energy_list(self) -> list[NDArray[np.float64]]:
        return [block.level_energies for block in self.blocks]

    @property
    def mix_coefficient_list(self) -> list[NDArray[np.float64]]:
        return [block.mix_coefficients for block in self.blocks]


class CSFsDict(TypedDict, total=False):
    """Dictionary representation of parsed CSF file data.

    Attributes:
        subshell_info_raw: Raw header lines describing subshells.
        CSFs_block_j_value: J values grouped by CSF block.
        parity: Parity label for the CSF data.
        CSFs_block_data: Three-line CSF records grouped by block.
        CSFs_block_length: Number of CSFs in each block.
        block_num: Number of CSF blocks.
    """

    subshell_info_raw: list[str]
    CSFs_block_j_value: list[str]
    parity: str
    CSFs_block_data: list[list[list[str]]]
    CSFs_block_length: list[int]
    block_num: int


@dataclass
class CSFs:
    """Structured CSF file data grouped by symmetry block.

    Attributes:
        subshell_info_raw: Raw header lines describing subshells.
        CSFs_block_j_value: J values grouped by CSF block.
        parity: Parity label for the CSF data.
        CSFs_block_data: Three-line CSF records grouped by block.
        CSFs_block_length: Number of CSFs in each block.
        block_num: Number of CSF blocks.
    """

    subshell_info_raw: list[str]
    CSFs_block_j_value: list[str]
    parity: str
    CSFs_block_data: list[
        list[list[str]]
    ]  # list of blocks, each block is list of CSFs (3 lines each)
    CSFs_block_length: list[int]
    block_num: int

    @classmethod
    def from_dict(cls, data: CSFsDict) -> "CSFs":
        """Build a CSFs instance from a dictionary representation.

        Args:
            data: Partial or complete CSF dictionary returned by legacy
                loaders.

        Returns:
            CSFs instance with missing fields filled by safe defaults.
        """
        return cls(
            subshell_info_raw=data.get("subshell_info_raw", []),
            CSFs_block_j_value=data.get("CSFs_block_j_value", []),
            parity=data.get("parity", ""),
            CSFs_block_data=data.get("CSFs_block_data", []),
            CSFs_block_length=data.get("CSFs_block_length", []),
            block_num=data.get("block_num", 0),
        )


@dataclass
class MLDataCounts:
    """Counters and retention metrics tracked during ML-guided CSF selection.

    Attributes:
        total_csfs_count: Total number of CSFs in the candidate space.
        cal_csfs_count: Number of CSFs included in the current calculation.
        important_csfs_count: Count of verified important CSFs.
        ml_sampled_count: Count of CSFs selected by ML for the next loop.
        ml_predicted_count: Count of CSFs predicted as important by ML.
        final_sampled_count: Final number of CSFs sampled for calculation.
        important_retention_rate: Retention rate for verified important CSFs.
        screening_retention_rate: Retention rate after explicit calculation.
        ml_retention_rate: Retention rate after ML prediction and cutoff.
        iteration_retention_rate: Relative growth between calculation loops.
    """

    total_csfs_count: int
    cal_csfs_count: int

    important_csfs_count: int | None = None
    ml_sampled_count: int | None = None
    ml_predicted_count: int | None = None
    final_sampled_count: int | None = None

    # 重要组态留存率 (important Retention Rate)：本轮计算的重要组态/上一轮计算的重要组态
    important_retention_rate: float | None = None
    # 验证留存率 (Screening Retention Rate) / 良品率:经过实际计算（或仿真/实验）后，有多少数据被认为是“好”的并保留下来。
    screening_retention_rate: float | None = None
    # ML 预测留存率 (ML Selection Retention Rate) : 进行预测并截断时产生的留存率,模型对未知空间的探索力度。
    ml_retention_rate: float | None = None
    # 迭代增长率 (Iteration Growth/Retention Rate): 下一次计算的规模相对于这一次的变化, 控制计算成本。如果 $>1$，计算量在发散；如果 $<1$，计算量在收敛。
    iteration_retention_rate: float | None = None
