# -*- encoding: utf-8 -*-
"""
@Id :radial_wavefunction_loader.py
@date :2026/01/19
@author :YenochQin (秦毅)
"""

import struct

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...utils.tool_function import align_2d_list_columns, int_nl_2_str_nl
from .binary_file_loader import BinaryFileLoader


class RadialWavefunctionLoader(BinaryFileLoader):
    """径向波函数二进制文件加载器 (.w文件）

    用于加载 GRASP2018 生成的径向波函数二进制文件。

    文件格式对应 Fortran 代码：
        write (3) 'G92RWF'
        write (3) nn, laky, energy, npts
        write (3) a0, (pg(j,i), j=1, npts), (qg(j,i), j=1, npts)
        write (3) (rg(j,i), j=1, npts)
    """

    def load(self) -> pd.DataFrame:
        """加载径向波函数数据

        Returns:
            DataFrame对象，包含：
                - r(a.u): 径向坐标
                - P(nl): 各轨道的大分量
                - Q(nl): 各轨道的小分量

        Raises:
            ValueError: 文件格式不正确
            IOError: 读取错误
        """
        nn_list = []
        laky_list = []
        energy_list = []
        npts_list = []
        a0_list = []
        pg_list = []
        qg_list = []
        rg_list = []

        with open(self.file_path, "rb") as binary_file:
            # 读取文件头标识（G92RWF 前后的记录标记）
            header = self.read_fortran_record(binary_file, "S1", count=6)
            g92rwf = b"".join(header).decode("utf-8").strip()

            if g92rwf != "G92RWF":
                raise ValueError(f"Not a radial wavefunction file: {g92rwf}")

            # 读取轨道数据（每个轨道有三条 Fortran 记录）
            while True:
                try:
                    # 第一条记录: read (3) nn, laky, energy, npts
                    # 4字节整数 + 4字节整数 + 8字节浮点数 + 4字节整数 = 20字节
                    nn, laky, energy, npts = self.read_mixed_fortran_record(
                        binary_file, ["i", "i", "d", "i"]
                    )
                    nn_list.append(nn)
                    laky_list.append(laky)
                    energy_list.append(energy)
                    npts_list.append(npts)

                except (ValueError, struct.error):
                    # 读取失败，说明到达文件末尾或格式错误
                    break

                # 第二条记录: read (3) a0, (pg(j,i), j=1, npts), (qg(j,i), j=1, npts)
                # 8字节浮点数 + npts*8字节 + npts*8字节
                a0, pg, qg = self.read_mixed_fortran_record(
                    binary_file, [("d", 1), ("d", npts), ("d", npts)]
                )
                a0_list.append(a0)
                pg_list.append(np.array(pg))
                qg_list.append(np.array(qg))

                # 第三条记录: read (3) (rg(j,i), j=1, npts)
                # npts*8字节
                rg = self.read_mixed_fortran_record(binary_file, [("d", npts)])[0]
                rg_list.append(np.array(rg))

        # 对齐所有列表的列数
        rg_list_len = [len(rg_list[i]) for i in range(len(rg_list))]
        max_rg_idx = rg_list_len.index(max(rg_list_len))
        pg_aligned_list = align_2d_list_columns(pg_list)
        qg_aligned_list = align_2d_list_columns(qg_list)

        # 收集所有列到字典中，避免 DataFrame 碎片化
        columns_data = {"r(a.u)": rg_list[max_rg_idx]}

        for n in range(len(nn_list)):
            str_nl = int_nl_2_str_nl(nn_list[n], laky_list[n])
            columns_data[f"P({str_nl})"] = pg_aligned_list[n]
            columns_data[f"Q({str_nl})"] = qg_aligned_list[n]

        # 一次性创建 DataFrame
        rwfn_df = pd.DataFrame(columns_data)

        return rwfn_df

    def get_functions_for_orbital(
        self, orbital_n: int, orbital_l: int
    ) -> dict[str, NDArray[np.float64]]:
        """获取指定轨道的波函数数据

        Args:
            orbital_n: 主量子数
            orbital_l: 轨道角动量量子数

        Returns:
            包含 P(r) 和 Q(r) 值的字典
        """
        df = self.load()

        # 构建列名
        # 需要根据 l 值转换为对应的字母 (0=s, 1=p, 2=d, 3=f, ...)
        l_map = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}
        l_char = l_map.get(orbital_l, f"laky{orbital_l}")

        p_col = f"P({orbital_n}{l_char})"
        q_col = f"Q({orbital_n}{l_char})"

        if p_col in df.columns and q_col in df.columns:
            p_values = df[p_col].to_numpy()
            q_values = df[q_col].to_numpy()
            return {"P_r": p_values, "Q_r": q_values}
        else:
            raise ValueError(
                f"Orbital data not found: n={orbital_n}, l={orbital_l}. "
                f"Available columns: {list(df.columns)}"
            )

    def get_grid(self) -> NDArray[np.float64]:
        """获取径向网格

        Returns:
            径向坐标数组
        """
        df = self.load()
        if "r(a.u)" in df.columns:
            return df["r(a.u)"].to_numpy()
        else:
            raise ValueError("r(a.u) column not found in DataFrame")

    def get_all_orbitals(self) -> list[tuple[int, int]]:
        """获取所有轨道的量子数

        Returns:
            (主量子数, 轨道角动量量子数) 元组列表
        """
        df = self.load()

        orbitals = []
        for col in df.columns:
            if col.startswith("P(") and col.endswith(")"):
                # 解析列名，如 "P(4s)" 或 "P(5d)"
                orbital_str = col[2:-1]  # 去掉 "P(" 和 ")"

                # 提取数字部分
                n_str = ""
                l_str = ""
                for char in orbital_str:
                    if char.isdigit():
                        n_str += char
                    else:
                        l_str += char

                if n_str and l_str:
                    # 将字母映射回数字
                    l_map = {
                        "s": 0,
                        "p": 1,
                        "d": 2,
                        "f": 3,
                        "g": 4,
                        "h": 5,
                        "i": 6,
                    }
                    orbital_n = int(n_str)
                    orbital_l = l_map.get(l_str.lower(), -1)
                    if orbital_l >= 0:
                        orbitals.append((orbital_n, orbital_l))

        return sorted(set(orbitals))
