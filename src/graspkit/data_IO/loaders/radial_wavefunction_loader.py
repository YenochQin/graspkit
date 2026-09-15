# -*- encoding: utf-8 -*-
import struct
from dataclasses import dataclass
from typing import override

import numpy as np
import polars as pl
from numpy import bytes_
from numpy.typing import NDArray

from .binary_file_loader import BinaryFileLoader


@dataclass(frozen=True)
class RWFNOrbitalData:
    """One orbital from a G92RWF file, with its native radial grid."""

    n: int
    kappa: int
    energy: float
    a0: float
    p: NDArray[np.float64]
    q: NDArray[np.float64]
    r: NDArray[np.float64]


def align_2d_list_columns(
    two_dimensional_list: list[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
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

    def load_orbitals(self) -> dict[tuple[int, int], RWFNOrbitalData]:
        """Load orbitals without discarding kappa or per-orbital grids."""

        orbitals: dict[tuple[int, int], RWFNOrbitalData] = {}
        with open(file=self.file_path, mode="rb") as binary_file:
            # 读取文件头标识（G92RWF 前后的记录标记）
            header: NDArray[bytes_] = self.read_fortran_record(binary_file, "S1", 6)
            g92rwf: str = b"".join(header).decode(encoding="utf-8").strip()

            if g92rwf != "G92RWF":
                raise ValueError(f"Not a radial wavefunction file: {g92rwf}")

            # Read three Fortran records per orbital.  Probe for EOF before
            # reading metadata so malformed records are not mistaken for EOF.
            while True:
                position = binary_file.tell()
                if not binary_file.read(1):
                    break
                binary_file.seek(position)
                nn, laky, energy, npts = self.read_mixed_scalars(
                    file=binary_file, field_specs=["i", "i", "d", "i"]
                )
                orbital_n = int(nn)
                orbital_kappa = int(laky)
                point_count = int(npts)
                if point_count <= 0:
                    raise ValueError(
                        f"Invalid radial point count for {orbital_n},{orbital_kappa}: "
                        f"{point_count}"
                    )

                a0_data, pg, qg = self.read_mixed_arrays(
                    file=binary_file,
                    field_specs=[
                        ("d", 1),
                        ("d", point_count),
                        ("d", point_count),
                    ],
                )
                (rg,) = self.read_mixed_arrays(
                    file=binary_file, field_specs=[("d", point_count)]
                )
                key = (orbital_n, orbital_kappa)
                if key in orbitals:
                    raise ValueError(
                        f"Duplicate orbital in radial wavefunction: {key}"
                    )
                orbitals[key] = RWFNOrbitalData(
                    n=orbital_n,
                    kappa=orbital_kappa,
                    energy=float(energy),
                    a0=float(a0_data[0]),
                    p=np.asarray(pg, dtype=np.float64),
                    q=np.asarray(qg, dtype=np.float64),
                    r=np.asarray(rg, dtype=np.float64),
                )

        if not orbitals:
            raise ValueError(f"No orbitals in radial wavefunction: {self.file_path}")
        return orbitals

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
        orbitals = list(self.load_orbitals().values())
        pg_list = [orbital.p for orbital in orbitals]
        qg_list = [orbital.q for orbital in orbitals]

        # 对齐所有列表的列数
        max_grid_orbital = max(orbitals, key=lambda orbital: len(orbital.r))
        pg_aligned_list = align_2d_list_columns(two_dimensional_list=pg_list)
        qg_aligned_list = align_2d_list_columns(two_dimensional_list=qg_list)

        # 收集所有列到字典中，避免 DataFrame 碎片化
        columns_data: dict[str, NDArray[np.float64]] = {
            "r(a.u)": max_grid_orbital.r
        }

        for index, orbital in enumerate(orbitals):
            str_nl = int_nl_2_str_nl(n=orbital.n, kappa=orbital.kappa)
            columns_data[f"P({str_nl})"] = pg_aligned_list[index]
            columns_data[f"Q({str_nl})"] = qg_aligned_list[index]

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
