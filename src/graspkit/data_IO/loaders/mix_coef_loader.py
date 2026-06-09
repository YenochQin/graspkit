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
_J_VALUE_LIST: list[str] = [
    "0",
    "1/2",
    "1",
    "3/2",
    "2",
    "5/2",
    "3",
    "7/2",
    "4",
    "9/2",
    "5",
    "11/2",
    "6",
    "13/2",
    "7",
    "15/2",
    "8",
    "17/2",
    "9",
    "19/2",
    "10",
    "21/2",
    "11",
    "23/2",
    "12",
    "25/2",
    "13",
    "27/2",
    "14",
    "29/2",
    "15",
    "31/2",
    "16",
    "33/2",
    "17",
    "35/2",
    "18",
    "37/2",
    "19",
    "39/2",
    "20",
    "41/2",
    "21",
    "43/2",
    "22",
]
_PARITY_LIST: list[str] = ["+", "-"]
_RYDBERG_CONSTANT: float = 109737.31568508


class MixCoefLoader(BinaryFileLoader):
    """混合系数文件加载器 (.[c]m文件）

    用于加载 GRASP2018 生成的混合系数二进制文件。
    能级数据使用 rich 库进行格式化输出。
    """

    @override
    def load(self) -> MixCoefficientData:
        """加载混合系数数据

        Returns:
            MixCoefficientData对象，包含：
                - block_num: 数据块数量
                - block_idx_list: 块索引列表
                - block_CSFs_nums: 每块的CSF数量
                - block_energy_count_list: 每块的能级数量
                - level_J_value_list: J值列表
                - parity_list: 宇称列表
                - block_levels_idx_list: 块中能级索引列表
                - block_energy_list: 块能量列表
                - block_level_energy_list: 块中能级能量列表
                - mix_coefficient_list: 混合系数列表
                - level_list: 能级列表

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
            _nelec, _ncftot, _nw, _ncmin, _nvecsiz, nblock = tuple[int, ...](
                int(x) for x in self.read_mixed_scalars(
                    file=binary_file,
                    field_specs=["i", "i", "i", "i", "i", "i"]
                )
            )

            blocks: list[MixCoefficientBlock] = []

            # 使用进度条处理数据块
            for jblock in range(1, nblock + 1):
                # READ (nfmix) nb, ncfblk, nevblk, iatjp, iaspa
                nb, ncfblk, nevblk, iatjp, iaspa = tuple[int, ...](
                    int(x) for x in self.read_mixed_scalars(
                        file=binary_file,
                        field_specs=["i", "i", "i", "i", "i"]
                    )
                )

                block_index = nb - 1

                if jblock != nb:
                    raise ValueError(f"jblock ({jblock}) != nb ({nb})")

                # READ (nfmix) ivec
                ivec: NDArray[np.int32] = self.read_fortran_record(file=binary_file, dtype="int32", count=nevblk)

                ivec_array: NDArray[np.int32] = (
                    np.array(ivec) - 1
                )  # use python idx method not fortran idx method

                # READ (nfmix) eav, (eval(i+ncountState), i = 1, nevblk)
                eva_evals: NDArray[np.float64] = self.read_fortran_record(file=binary_file, dtype="float64", count=nevblk + 1)

                eav: float = cast(np.float64, eva_evals[0]).item()
                evals: NDArray[np.float64] = eva_evals[1:]

                # READ (nfmix) (evec, i = 1, ncfblk*nevblk)
                evecsblock: NDArray[np.float64] = self.read_fortran_record(file=binary_file, dtype="float64", count=nevblk * ncfblk)

                evecs: ndarray[tuple[int, int], dtype[np.float64]] = evecsblock.reshape(nevblk, ncfblk)

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
                        j_value=_J_VALUE_LIST[iatjp - 1],
                        parity=iaspa,
                        level_indices=ivec_array.astype(np.int64, copy=False),
                        base_energy=eav,
                        level_energies=evals,
                        mix_coefficients=evecs,
                    )
                )

            # 收集能级数据用于打印
            temp_pos: list[int] = []
            temp_J: list[str] = []
            temp_parity_idx: list[int] = []
            temp_energy: list[float] = []

            for block in blocks:
                for pos in block.level_indices.tolist():
                    temp_pos.append(pos)
                    temp_J.append(block.j_value)
                    temp_parity_idx.append(block.parity)
                    temp_energy.append(float(block.base_energy + block.level_energies[pos]))

            # 按能量排序
            level_idx: NDArray[np.int_] = np.argsort(temp_energy)
            level_energy_list: list[float] = [temp_energy[int(i)] for i in level_idx]

            # 创建 MixCoefficientData 对象
            data = MixCoefficientData(
                blocks=blocks,
                level_list=level_energy_list,
            )

            # 使用 rich 打印能级数据
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

    # 重建能级数据（复现 load() 中第 249-273 行的逻辑）
    temp_pos: list[int] = []
    temp_J: list[str] = []
    temp_parity_idx: list[int] = []
    temp_energy: list[float] = []

    for jblock in range(data.block_num):
        for pos in data.block_levels_idx_list[jblock].tolist():
            temp_pos.append(pos)
            temp_J.append(data.level_J_value_list[jblock])
            temp_parity_idx.append(data.parity_list[jblock])
            temp_energy.append(
                float(data.block_energy_list[jblock] + data.block_level_energy_list[jblock][pos])
            )

    # 按能量排序
    level_idx = np.argsort(temp_energy)

    # 填充表格
    base_energy = temp_energy[int(level_idx[0])]

    for i, idx in enumerate(level_idx):
        pos = temp_pos[int(idx)]
        j_val = temp_J[int(idx)]
        parity = _PARITY_LIST[temp_parity_idx[int(idx)] - 1]
        energy_au = temp_energy[int(idx)]

        if i == 0:
            levels_cm = 0.0
        else:
            energy_diff = energy_au - base_energy
            levels_cm = energy_diff * Rydberg * 2

        table.add_row(
            str(i + 1),
            str(pos),
            j_val,
            parity,
            f"{energy_au:14.7f}",
            f"{levels_cm:12.2f}",
        )

    # 输出
    group = Group(
        f"  Rydberg constant is  {Rydberg}  \n  Blocks: {data.block_num}  |  Total Levels: {len(data.level_list):,}",
        table
    )
    console.print(Align.left(renderable=group))
