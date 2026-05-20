# -*- encoding: utf-8 -*-
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

import polars as pl

from .base_loader import BaseLoader


@dataclass(frozen=True)
class NuclearParameters:
    """超精细结构计算使用的核参数

    Attributes:
        spin: 核自旋
        spin_unit: 核自旋单位
        magnetic_dipole_moment: 核磁偶极矩
        magnetic_dipole_moment_unit: 核磁偶极矩单位
        electric_quadrupole_moment: 核电四极矩
        electric_quadrupole_moment_unit: 核电四极矩单位
    """

    spin: float
    spin_unit: str
    magnetic_dipole_moment: float
    magnetic_dipole_moment_unit: str
    electric_quadrupole_moment: float
    electric_quadrupole_moment_unit: str


class HyperfineStructureLoader(BaseLoader[pl.DataFrame]):
    """超精细结构文件加载器

    用于加载 GRASP 超精细结构输出文件（通常为 .h 或 .ch 文本格式）。
    解析文件头中的核参数，以及 ``Interaction constants`` 表中的 A/B 常数
    和 Landé g 因子数据。
    """

    COLUMN_NAMES: ClassVar[list[str]] = [
        "Level",
        "J",
        "Parity",
        "A_MHz",
        "B_MHz",
        "g_J",
        "delta_g_J",
        "total_g_J",
    ]

    NUCLEAR_PARAMETER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"""
        ^\s*
        (Nuclear\s+spin|Nuclear\s+magnetic\s+dipole\s+moment|Nuclear\s+electric\s+quadrupole\s+moment)
        \s+(\S+)\s+(.+?)\s*$
        """,
        re.VERBOSE,
    )

    DATA_LINE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"""
        ^\s*
        (\d+)      \s+  # Level index
        ([\d/]+)   \s+  # J value
        ([+-])     \s+  # Parity
        (\S+)      \s+  # A (MHz)
        (\S+)      \s+  # B (MHz)
        (\S+)      \s+  # g_J
        (\S+)      \s+  # delta g_J
        (\S+)           # total g_J
        \s*$
        """,
        re.VERBOSE,
    )

    def __init__(self, file_path: str | Path) -> None:
        """初始化超精细结构文件加载器

        Args:
            file_path: 超精细结构输出文件路径（.h 或 .ch 文本文件）
        """
        super().__init__(file_path)
        self.nuclear_parameters: NuclearParameters
        self.df: pl.DataFrame
        self.nuclear_parameters, self.df = self._parse_hyperfine_file()

    @staticmethod
    def _parse_grasp_float(value: str) -> float:
        """解析 GRASP 文本输出中的浮点数

        GRASP 输出常使用 Fortran 的 ``D`` 指数记法，例如 ``1.23D+02``。
        """
        normalized_value = value.strip().replace("D", "E").replace("d", "E")
        if normalized_value.lower() == "nan":
            return math.nan
        return float(normalized_value)

    def _parse_hyperfine_file(self) -> tuple[NuclearParameters, pl.DataFrame]:
        """解析超精细结构文本文件

        Returns:
            核参数对象和包含 interaction constants 的 DataFrame

        Raises:
            ValueError: 文件缺少必要的核参数或数据表
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]

        nuclear_values: dict[str, tuple[float, str]] = {}
        data_rows: list[dict[str, int | str | float]] = []
        in_table = False

        for line in lines:
            nuclear_match = self.NUCLEAR_PARAMETER_PATTERN.match(line)
            if nuclear_match:
                label = nuclear_match.group(1)
                nuclear_values[label] = (
                    self._parse_grasp_float(nuclear_match.group(2)),
                    nuclear_match.group(3).strip(),
                )
                continue

            if line.strip().startswith("Level1"):
                in_table = True
                continue

            if not in_table or not line.strip():
                continue

            data_match = self.DATA_LINE_PATTERN.match(line)
            if data_match is None:
                continue

            data_rows.append(
                {
                    "Level": int(data_match.group(1)),
                    "J": data_match.group(2),
                    "Parity": data_match.group(3),
                    "A_MHz": self._parse_grasp_float(data_match.group(4)),
                    "B_MHz": self._parse_grasp_float(data_match.group(5)),
                    "g_J": self._parse_grasp_float(data_match.group(6)),
                    "delta_g_J": self._parse_grasp_float(data_match.group(7)),
                    "total_g_J": self._parse_grasp_float(data_match.group(8)),
                }
            )

        required_labels = [
            "Nuclear spin",
            "Nuclear magnetic dipole moment",
            "Nuclear electric quadrupole moment",
        ]
        missing_labels = [
            label for label in required_labels if label not in nuclear_values
        ]
        if missing_labels:
            raise ValueError(
                f"Missing nuclear parameter lines in hyperfine structure file: \n{', '.join(missing_labels)}"
            )
        if not data_rows:
            raise ValueError(
                "No interaction constants data found in hyperfine structure file"
            )

        spin, spin_unit = nuclear_values["Nuclear spin"]
        magnetic_dipole_moment, magnetic_dipole_moment_unit = nuclear_values[
            "Nuclear magnetic dipole moment"
        ]
        electric_quadrupole_moment, electric_quadrupole_moment_unit = nuclear_values[
            "Nuclear electric quadrupole moment"
        ]

        nuclear_parameters = NuclearParameters(
            spin=spin,
            spin_unit=spin_unit,
            magnetic_dipole_moment=magnetic_dipole_moment,
            magnetic_dipole_moment_unit=magnetic_dipole_moment_unit,
            electric_quadrupole_moment=electric_quadrupole_moment,
            electric_quadrupole_moment_unit=electric_quadrupole_moment_unit,
        )

        return nuclear_parameters, pl.DataFrame(data_rows, schema=self.COLUMN_NAMES)

    @override
    def load(self) -> pl.DataFrame:
        """加载数据（实现抽象方法）

        Returns:
            包含所有超精细结构 interaction constants 的 DataFrame
        """
        return self.df

    def get_interaction_constants(self) -> pl.DataFrame:
        """获取 interaction constants 表格数据

        Returns:
            包含 A/B 常数和 Landé g 因子的 DataFrame
        """
        return self.df

    def get_nuclear_parameters(self) -> NuclearParameters:
        """获取文件头中的核参数

        Returns:
            核参数对象
        """
        return self.nuclear_parameters

    def get_number_of_levels(self) -> int:
        """获取表格中的能级数量

        Returns:
            超精细结构表格行数
        """
        return len(self.df)
