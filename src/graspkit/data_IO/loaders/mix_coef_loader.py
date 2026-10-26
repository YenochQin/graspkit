# -*- encoding: utf-8 -*-
from typing import cast, override

import numpy as np
from numpy import bytes_, dtype, ndarray
from numpy.typing import NDArray
from rich.align import Align
from rich.console import Console, Group
from rich.table import Table

from ...utils.data_modules import MixCoefficientBlock, MixCoefficientData
from .binary_file_loader import BinaryFileLoader

# 能级显示相关的常量
_PARITY_LIST: list[str] = ["+", "-"]
_RYDBERG_CONSTANT: float = 109737.31568508


def _collect_level_metadata(
    blocks: list[MixCoefficientBlock],
) -> tuple[list[int], list[str], list[int], list[float]]:
    """Collect strongly typed level metadata from NumPy-backed blocks."""
    level_ids: list[int] = []
    j_values: list[str] = []
    parity_indices: list[int] = []
    energies: list[float] = []

    for block in blocks:
        block_level_ids = cast(list[int], block.level_ids.tolist())
        for row_index, level_id in enumerate(block_level_ids):
            level_energy = cast(
                np.float64,
                block.level_energies[row_index],
            ).item()
            level_ids.append(level_id)
            j_values.append(block.j_value)
            parity_indices.append(block.parity)
            energies.append(block.base_energy + level_energy)

    return level_ids, j_values, parity_indices, energies


def _sorted_level_order(energies: list[float]) -> list[int]:
    """Return level indices sorted by ascending energy."""
    return sorted(range(len(energies)), key=lambda index: energies[index])


class MixCoefLoader(BinaryFileLoader):
    """混合系数文件加载器 (.[c]m文件）

    用于加载 GRASP2018 生成的混合系数二进制文件。
    能级数据使用 rich 库进行格式化输出。
    """

    @override
    def load(self, *, show_levels: bool = True) -> MixCoefficientData:
        """加载混合系数数据

        Args:
            show_levels: 是否打印能级表；CLI 批处理可设为 False。

        Returns:
            MixCoefficientData对象，包含：
                - block_num: 数据块数量
                - block_idx_list: 块索引列表
                - block_CSFs_nums: 每块的CSF数量
                - block_energy_count_list: 每块的能级数量
                - level_J_value_list: J值列表
                - parity_list: 宇称列表
                - block_level_ids_list: 块中 GRASP 能级 ID 列表
                - block_energy_list: 块能量列表
                - block_level_energy_list: 块中能级能量列表
                - mix_coefficient_list: 混合系数列表
                - sorted_level_energies: 排序后的绝对能级列表

        Raises:
            ValueError: 文件格式不正确
            IOError: 读取错误
        """
        with open(file=self.file_path, mode="rb") as binary_file:
            # 读取文件头标识（G92MIX 前后的记录标记）
            # 使用 S1 读取每个字节，然后拼接
            header: NDArray[bytes_] = self.read_fortran_record(binary_file, "S1", 6)
            g92mix: str = b"".join(header).decode("utf-8").strip()

            if g92mix != "G92MIX":
                raise ValueError(f"Not a mix coefficient file: {g92mix}")

            # READ (nfmix) nelec, ncftot, nw, nvectot, nvecsiz, nblock
            # nelec -> num_electron, ncftot -> total_num_configuration,
            # nw -> nw, ncmin -> ncmin, nvecsiz -> nvecsiz, nblock -> num_block
            nelec, ncftot, nw, ncmin, nvecsiz, nblock = tuple[int, ...](
                int(x)
                for x in self.read_mixed_scalars(
                    file=binary_file, field_specs=["i", "i", "i", "i", "i", "i"]
                )
            )

            if min(nelec, ncftot, nw, ncmin, nvecsiz, nblock) < 0:
                raise ValueError("Negative count in mix coefficient file header")

            blocks: list[MixCoefficientBlock] = []

            # 使用进度条处理数据块
            for jblock in range(1, nblock + 1):
                # READ (nfmix) nb, ncfblk, nevblk, iatjp, iaspa
                nb, ncfblk, nevblk, iatjp, iaspa = tuple[int, ...](
                    int(x)
                    for x in self.read_mixed_scalars(
                        file=binary_file, field_specs=["i", "i", "i", "i", "i"]
                    )
                )

                block_index = nb - 1

                if jblock != nb:
                    raise ValueError(f"jblock ({jblock}) != nb ({nb})")

                if ncfblk < 1 or nevblk < 0 or iatjp < 1 or iaspa not in (1, 2):
                    raise ValueError(f"Invalid mix coefficient block {nb} header")

                # GRASP omits all three data records when nevblk == 0.
                if nevblk:
                    ivec: NDArray[np.int32] = self.read_fortran_record(
                        file=binary_file, dtype="int32", count=nevblk
                    )
                    ivec_array = ivec.astype(np.int64) - 1
                    if np.any(ivec_array < 0):
                        raise ValueError("GRASP level IDs must be positive")
                    eva_evals: NDArray[np.float64] = self.read_fortran_record(
                        file=binary_file, dtype="float64", count=nevblk + 1
                    )
                    eav = cast(np.float64, eva_evals[0]).item()
                    evals = eva_evals[1:]
                    evecsblock: NDArray[np.float64] = self.read_fortran_record(
                        file=binary_file, dtype="float64", count=nevblk * ncfblk
                    )
                    evecs: ndarray[tuple[int, int], dtype[np.float64]] = (
                        evecsblock.reshape(nevblk, ncfblk)
                    )
                else:
                    ivec_array = np.empty(0, dtype=np.int64)
                    eav = 0.0
                    evals = np.empty(0, dtype=np.float64)
                    evecs = np.empty((0, ncfblk), dtype=np.float64)

                if ncfblk != evecs.shape[1]:
                    raise ValueError(
                        f"{ncfblk=}: number of configuration functions in block should equal {evecs.shape[1]=}"
                    )

                blocks.append(
                    MixCoefficientBlock(
                        block_index=block_index,
                        csf_count=ncfblk,
                        level_count=nevblk,
                        j_value_location=iatjp,
                        j_value=(
                            str((iatjp - 1) // 2)
                            if (iatjp - 1) % 2 == 0
                            else f"{iatjp - 1}/2"
                        ),
                        parity=iaspa,
                        level_ids=ivec_array.astype(np.int64, copy=False),
                        base_energy=eav,
                        level_energies=evals,
                        mix_coefficients=evecs,
                    )
                )

            if (
                sum(block.csf_count for block in blocks) != ncftot
                or sum(block.level_count for block in blocks) != ncmin
                or sum(block.csf_count * block.level_count for block in blocks)
                != nvecsiz
            ):
                raise ValueError("Mix header totals disagree with block dimensions")
            if binary_file.read(1):
                raise ValueError("Unexpected trailing data in mix coefficient file")

            # 收集能级能量
            temp_energy = _collect_level_metadata(blocks)[3]

            # 按能量排序
            level_order = _sorted_level_order(temp_energy)
            level_energy_list = [temp_energy[index] for index in level_order]

            # 创建 MixCoefficientData 对象
            data = MixCoefficientData(
                blocks=blocks,
                sorted_level_energies=level_energy_list,
                electron_count=nelec,
                orbital_count=nw,
            )

            # 使用 rich 打印能级数据
            if show_levels:
                print_mix_coef_levels_rich(data)

            return data

    def get_block_count(self) -> int:
        """获取块数量

        Returns:
            数据块总数
        """
        data: MixCoefficientData = self.load()
        return int(data.block_num)


def print_mix_coef_levels_rich(
    data: MixCoefficientData,
    Rydberg: float = _RYDBERG_CONSTANT,
) -> None:
    """使用 rich 库打印能级数据

    Args:
        data: MixCoefficientData 对象
        console: Rich Console 对象，如果为 None 则创建新的
        title: 面板标题
        Rydberg: 里德伯常数
    """

    console = Console()

    # 创建表格
    table = Table()
    table.add_column("No", justify="right", width=4)
    table.add_column("Pos", justify="right", width=4)
    table.add_column("J", justify="right", width=4)
    table.add_column("Parity", justify="center", width=6)
    table.add_column("Energy (a.u.)", justify="right", width=18)
    table.add_column("Levels (cm⁻¹)", justify="right", width=14)

    # 重建能级数据（复现 load() 中的 block-level 能量整理逻辑）
    temp_pos, temp_j, temp_parity_idx, temp_energy = _collect_level_metadata(
        data.blocks
    )

    # 按能量排序
    level_order = _sorted_level_order(temp_energy)

    # 填充表格
    base_energy = temp_energy[level_order[0]] if level_order else 0.0

    for number, index in enumerate(level_order, start=1):
        pos = temp_pos[index]
        j_val = temp_j[index]
        parity = _PARITY_LIST[temp_parity_idx[index] - 1]
        energy_au = temp_energy[index]

        if number == 1:
            levels_cm = 0.0
        else:
            energy_diff = energy_au - base_energy
            levels_cm = energy_diff * Rydberg * 2

        table.add_row(
            str(number),
            str(pos),
            j_val,
            parity,
            f"{energy_au:14.7f}",
            f"{levels_cm:12.2f}",
        )

    # 输出
    group = Group(
        f"  Rydberg constant is  {Rydberg}  \n  Blocks: {data.block_num}  |  Total Levels: {len(data.sorted_level_energies):,}",
        table,
    )
    console.print(Align.left(renderable=group))
