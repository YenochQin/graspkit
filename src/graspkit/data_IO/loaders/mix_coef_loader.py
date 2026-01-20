# -*- encoding: utf-8 -*-
"""
@Id :mix_coef_loader.py
@date :2026/01/19
@author :YenochQin (秦毅)
"""

from typing import Any

import numpy as np

from ...utils.data_modules import MixCoefficientData
from ...utils.progress_manager import wrap_iterator
from .binary_file_loader import BinaryFileLoader

# 能级显示相关的常量
_J_VALUE_LIST = [
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
_PARITY_LIST = ["+", "-"]
_RYDBERG_CONSTANT = 109737.31568508


class MixCoefLoader(BinaryFileLoader):
    """混合系数文件加载器 (.[c]m文件）

    用于加载 GRASP2018 生成的混合系数二进制文件。
    保留原有的打印信息用于调试和验证。
    """

    def _level_print_title(self, Rydberg: float = _RYDBERG_CONSTANT) -> None:
        """打印能级表格标题

        Args:
            Rydberg: 里德伯常数
        """
        print(
            f"""
    Energy levels for ...
Rydberg constant is  {Rydberg}

---------------------------------------------
 No Pos  J  Parity   Energy Total    Levels
                      (a.u.)         (cm^-1)
---------------------------------------------
"""
        )

    def _level_J_value(self, j_idx: int) -> str:
        """根据 J 值索引返回对应的 J 值字符串

        Args:
            j_idx: J 值索引

        Returns:
            J 值字符串
        """
        return _J_VALUE_LIST[j_idx - 1]

    def _level_parity(self, parity_idx: int) -> str:
        """根据宇称索引返回对应的宇称符号

        Args:
            parity_idx: 宇称索引

        Returns:
            宇称符号 (+ 或 -)
        """
        return _PARITY_LIST[parity_idx - 1]

    def _energy_au_cm(
        self, energy_au: float, Rydberg: float = _RYDBERG_CONSTANT
    ) -> float:
        """将能量从原子单位转换为 cm^-1

        Args:
            energy_au: 能量（原子单位）
            Rydberg: 里德伯常数

        Returns:
            能量（cm^-1）
        """
        return energy_au * Rydberg * 2

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
        with open(self.file_path, "rb") as binary_file:
            # 读取文件头标识（G92MIX 前后的记录标记）
            # 使用 S1 读取每个字节，然后拼接
            header = self.read_fortran_record(binary_file, "S1", count=6)
            g92mix = b"".join(header).decode("utf-8").strip()

            if g92mix != "G92MIX":
                raise ValueError(f"Not a mix coefficient file: {g92mix}")
            print(f"g92mix: {g92mix}")  # 调试信息

            # READ (nfmix) nelec, ncftot, nw, nvectot, nvecsiz, nblock
            header_data = self.read_fortran_record(binary_file, "int32", count=6)

            # nelec -> num_electron, ncftot -> total_num_configuration,
            # nw -> nw, ncmin -> ncmin, nvecsiz -> nvecsiz, nblock -> num_block
            nelec, ncftot, nw, ncmin, nvecsiz, nblock = header_data

            print(
                f" nblock = {nblock},       ncftot =   {ncftot},          nw =  {nw},            nelec =   {nelec}"
            )

            idx_block_list = []
            ncfblk_list = []
            block_energy_count_list = []
            j_value_location_list = []
            parity_list = []
            ivec_list = []
            block_energy_list = []
            block_level_energy_list = []
            mix_coefficient_list = []

            # 使用进度条处理数据块
            for jblock in wrap_iterator(range(1, nblock + 1), desc="处理数据块"):
                print("cycle jblock =", jblock)

                # READ (nfmix) nb, ncfblk, nevblk, iatjp, iaspa
                block_data = self.read_fortran_record(binary_file, "int32", count=5)

                nb, ncfblk, nevblk, iatjp, iaspa = block_data
                print(
                    f" Block no. = {nb}, 2J+1 = {iatjp}, Parity = {iaspa}, No. of eigenvalues = {nevblk}, No. of CSFs = {ncfblk}"
                )

                idx_block_list.append(
                    nb - 1
                )  # use python idx method not fortran idx method
                ncfblk_list.append(ncfblk)
                block_energy_count_list.append(nevblk)
                j_value_location_list.append(iatjp)
                parity_list.append(iaspa)

                if jblock != nb:
                    raise ValueError(f"jblock ({jblock}) != nb ({nb})")

                # READ (nfmix) ivec
                ivec = self.read_fortran_record(binary_file, "int32", count=nevblk)

                ivec_array = (
                    np.array(ivec) - 1
                )  # use python idx method not fortran idx method

                ivec_list.append(ivec_array)

                # READ (nfmix) eav, (eval(i+ncountState), i = 1, nevblk)
                eva_evals = self.read_fortran_record(
                    binary_file, "float64", count=nevblk + 1
                )

                eav = eva_evals[0]
                evals = eva_evals[1:]

                block_energy_list.append(eav)
                block_level_energy_list.append(evals)

                # READ (nfmix) (evec, i = 1, ncfblk*nevblk)
                evecsblock = self.read_fortran_record(
                    binary_file, "float64", count=nevblk * ncfblk
                )

                evecs = evecsblock.reshape(nevblk, ncfblk)

                if ncfblk != len(evecs[0]):
                    raise ValueError(
                        f"ncfblk: number of configuration functions in block "
                        f"len(evecs[0]) should equal len(evecs[0])"
                    )

                mix_coefficient_list.append(evecs)

            # 打印能级标题和信息
            self._level_print_title()
            temp_pos = []
            temp_J = []
            temp_parity = []
            temp_energy = []

            # 收集所有能级数据
            for jblock in range(nblock):
                for pos in ivec_list[jblock]:
                    temp_pos.append(pos)
                    temp_J.append(self._level_J_value(j_value_location_list[jblock]))
                    temp_parity.append(self._level_parity(parity_list[jblock]))
                    temp_energy.append(
                        block_energy_list[jblock] + block_level_energy_list[jblock][pos]
                    )

            # 按能量排序
            level_idx = np.argsort(temp_energy)
            level_energy_list = []

            # 打印能级信息
            for i in range(len(level_idx)):
                if i == 0:
                    print(
                        f"{i + 1:3}{temp_pos[level_idx[i]]:3}{temp_J[level_idx[i]]:>4}   "
                        f"{temp_parity[level_idx[i]]:1}    {temp_energy[level_idx[i]]:14.7f}"
                        f"{0.0000000:12.2f}"
                    )
                    level_energy_list.append(temp_energy[level_idx[i]])
                else:
                    print(
                        f"{i + 1:3}{temp_pos[level_idx[i]]:3}{temp_J[level_idx[i]]:>4}   "
                        f"{temp_parity[level_idx[i]]:1}    {temp_energy[level_idx[i]]:14.7f}"
                        f"{self._energy_au_cm(temp_energy[level_idx[i]] - temp_energy[level_idx[0]]):12.2f}"
                    )
                    level_energy_list.append(temp_energy[level_idx[i]])

            return MixCoefficientData(
                block_num=nblock,
                block_idx_list=idx_block_list,
                block_CSFs_nums=ncfblk_list,
                block_energy_count_list=block_energy_count_list,
                level_J_value_list=temp_J,
                parity_list=parity_list,
                block_levels_idx_list=ivec_list,
                block_energy_list=block_energy_list,
                block_level_energy_list=block_level_energy_list,
                mix_coefficient_list=mix_coefficient_list,
                level_list=level_energy_list,
            )

    def get_block_data(self, block_idx: int) -> dict[str, Any]:
        """获取指定块的数据

        Args:
            block_idx: 块索引（0-based）

        Returns:
            包含块数据的字典

        Raises:
            IndexError: 块索引超出范围
        """
        data = self.load()

        if block_idx < 0 or block_idx >= data.block_num:
            raise IndexError(
                f"Block index {block_idx} out of range [0, {data.block_num})"
            )

        return {
            "block_index": block_idx,
            "ncfblk": data.block_CSFs_nums[block_idx],
            "nevblk": data.block_energy_count_list[block_idx],
            "j_value": data.level_J_value_list[block_idx],
            "parity": data.parity_list[block_idx],
            "evec": data.mix_coefficient_list[block_idx],
            "eval": data.block_level_energy_list[block_idx],
            "ivec": data.block_levels_idx_list[block_idx],
        }

    def get_block_count(self) -> int:
        """获取块数量

        Returns:
            数据块总数
        """
        data = self.load()
        return int(data.block_num)
