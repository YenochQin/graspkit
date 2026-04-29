# -*- encoding: utf-8 -*-
import struct

import numpy as np
import polars as pl
from numpy import bytes_
from numpy.typing import NDArray
from typing import override

from .binary_file_loader import BinaryFileLoader

def align_2d_list_columns(two_dimensional_list: list[NDArray[np.float64]]) -> list[NDArray[np.float64]]:
    """Pad one-dimensional arrays to the longest column length.

    Args:
        two_dimensional_list: List of one-dimensional arrays with possibly
            different lengths.

    Returns:
        New list where each array has been right-padded with zeros to the
        maximum input length.
    """
    if not two_dimensional_list:
        return []

    # 获取最长列的长度
    max_length: int = max(len(column) for column in two_dimensional_list)

    # 对每列进行处理，确保长度与最长列对齐
    aligned_list: list[NDArray[np.float64]] = []
    for column in two_dimensional_list:
        # 计算当前列需要补充的0的数量
        fill_count = max_length - len(column)
        # 进行补0操作并添加到结果列表中
        aligned_column: NDArray[np.float64] = np.pad(
            column, (0, fill_count), mode="constant", constant_values=0
        )
        aligned_list.append(aligned_column)

    return aligned_list

def int_nl_2_str_nl(n: int, kappa: int) -> str:
    r"""
    Convert integer quantum numbers to a GRASP orbital label.

    Args:
        n: Principal quantum number.
        kappa: Dirac kappa value.

    Returns:
        Orbital label such as ``"5s "`` or ``"4d-"``.

    Notes:
        For ``j = l + 1/2``, ``kappa = -(l + 1)``. For ``j = l - 1/2``,
        ``kappa = +l``.
    """
    l_list: list[str] = ["s", "p", "d", "f", "g", "h", "i"]
    str_nl: str = ""
    if kappa > 0:
        l = kappa
        str_nl = f"{n}{l_list[l]}-"
    elif kappa < 0:
        l = -kappa - 1
        str_nl = f"{n}{l_list[l]} "
    else:
        print("error: kappa should not be zero")

    return str_nl

class RWFNFileLoader(BinaryFileLoader):
    """径向波函数二进制文件加载器 (.w文件）

    用于加载 GRASP2018 生成的径向波函数二进制文件。

    文件格式对应 Fortran 代码：
        write (3) 'G92RWF'
        write (3) nn, laky, energy, npts
        write (3) a0, (pg(j,i), j=1, npts), (qg(j,i), j=1, npts)
        write (3) (rg(j,i), j=1, npts)
    """

    @override
    def load(self) -> pl.DataFrame:
        """加载径向波函数数据

        Returns:
            Polars DataFrame对象，包含：
                - r(a.u): 径向坐标
                - P(nl): 各轨道的大分量
                - Q(nl): 各轨道的小分量

        Raises:
            ValueError: 文件格式不正确
            IOError: 读取错误
        """
        nn_list: list[int] = []
        laky_list: list[int] = []
        energy_list: list[float] = []
        npts_list: list[int] = []
        a0_list: list[float] = []
        pg_list: list[NDArray[np.float64]] = []
        qg_list: list[NDArray[np.float64]] = []
        rg_list: list[NDArray[np.float64]] = []

        with open(file=self.file_path, mode="rb") as binary_file:
            # 读取文件头标识（G92RWF 前后的记录标记）
            header: NDArray[bytes_] = self.read_fortran_record(binary_file, "S1", 6)
            g92rwf: str = b"".join(header).decode(encoding="utf-8").strip()

            if g92rwf != "G92RWF":
                raise ValueError(f"Not a radial wavefunction file: {g92rwf}")

            # 读取轨道数据（每个轨道有三条 Fortran 记录）
            while True:
                try:
                    # 第一条记录: read (3) nn, laky, energy, npts
                    # 4字节整数 + 4字节整数 + 8字节浮点数 + 4字节整数 = 20字节
                    nn, laky, energy, npts = self.read_mixed_scalars(
                        file=binary_file, field_specs=["i", "i", "d", "i"]
                    )
                    nn_list.append(int(nn))
                    laky_list.append(int(laky))
                    energy_list.append(float(energy))
                    npts_list.append(int(npts))

                except (ValueError, struct.error):
                    # 读取失败，说明到达文件末尾或格式错误
                    break

                # 第二条记录: read (3) a0, (pg(j,i), j=1, npts), (qg(j,i), j=1, npts)
                # 8字节浮点数 + npts*8字节 + npts*8字节
                a0_data, pg, qg = self.read_mixed_arrays(
                    file=binary_file, field_specs=[("d", 1), ("d", int(npts)), ("d", int(npts))]
                )
                a0_list.append(float(a0_data[0]))
                pg_list.append(np.array(pg, dtype=np.float64))
                qg_list.append(np.array(qg, dtype=np.float64))

                # 第三条记录: read (3) (rg(j,i), j=1, npts)
                # npts*8字节
                rg = self.read_mixed_arrays(file=binary_file, field_specs=[("d", int(npts))])[0]
                rg_list.append(np.array(rg, dtype=np.float64))

        # 对齐所有列表的列数
        rg_list_len = [len(rg_list[i]) for i in range(len(rg_list))]
        max_rg_idx = rg_list_len.index(max(rg_list_len))
        pg_aligned_list = align_2d_list_columns(two_dimensional_list=pg_list)
        qg_aligned_list = align_2d_list_columns(two_dimensional_list=qg_list)

        # 收集所有列到字典中，避免 DataFrame 碎片化
        columns_data: dict[str, NDArray[np.float64]] = {"r(a.u)": rg_list[max_rg_idx]}

        for n in range(len(nn_list)):
            str_nl: str = int_nl_2_str_nl(n=nn_list[n], kappa=laky_list[n])
            columns_data[f"P({str_nl})"] = pg_aligned_list[n]
            columns_data[f"Q({str_nl})"] = qg_aligned_list[n]

        # 一次性创建 Polars DataFrame
        rwfn_df: pl.DataFrame = pl.DataFrame(data=columns_data)

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
        df: pl.DataFrame = self.load()

        # 构建列名
        # 需要根据 l 值转换为对应的字母 (0=s, 1=p, 2=d, 3=f, ...)
        l_map: dict[int, str] = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}
        l_char: str = l_map.get(orbital_l, f"laky{orbital_l}")

        p_col: str = f"P({orbital_n}{l_char})"
        q_col: str = f"Q({orbital_n}{l_char})"

        if p_col in df.columns and q_col in df.columns:
            p_values = df[p_col].to_numpy()
            q_values = df[q_col].to_numpy()
            return {"P_r": p_values, "Q_r": q_values}
        else:
            raise ValueError(
                f"Orbital data not found: n={orbital_n}, l={orbital_l}.\nAvailable columns: {df.columns}"
            )

    def get_grid(self) -> NDArray[np.float64]:
        """获取径向网格

        Returns:
            径向坐标数组
        """
        df: pl.DataFrame = self.load()
        if "r(a.u)" in df.columns:
            return df["r(a.u)"].to_numpy()
        else:
            raise ValueError("r(a.u) column not found in DataFrame")

    def get_all_orbitals(self) -> list[tuple[int, int]]:
        """获取所有轨道的量子数

        Returns:
            (主量子数, 轨道角动量量子数) 元组列表
        """
        df: pl.DataFrame = self.load()

        orbitals: list[tuple[int, int]] = []
        for col in df.columns:
            if col.startswith("P(") and col.endswith(")"):
                # 解析列名，如 "P(4s)" 或 "P(5d)"
                orbital_str: str = col[2:-1]  # 去掉 "P(" 和 ")"

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
                    l_map: dict[str, int] = {
                        "s": 0,
                        "p": 1,
                        "d": 2,
                        "f": 3,
                        "g": 4,
                        "h": 5,
                        "i": 6,
                    }
                    orbital_n: int = int(n_str)
                    orbital_l: int = l_map.get(l_str.lower(), -1)
                    if orbital_l >= 0:
                        orbitals.append((orbital_n, orbital_l))

        return sorted(set[tuple[int, int]](orbitals))
