# -*- encoding: utf-8 -*-
import math
import re
from pathlib import Path
from typing import ClassVar, override

import polars as pl

from .base_loader import BaseLoader
from .hyperfine_structure_loader import NuclearParameters


class GJFactorLoader(BaseLoader[pl.DataFrame]):
    """g_J 因子文件加载器

    用于加载 GRASP 相关程序输出的 .gj 文本文件。该文件格式与超精细结构
    interaction constants 输出类似，但表格中只包含 Landé g 因子相关数据，
    不包含超精细 A/B 常数。
    """

    COLUMN_NAMES: ClassVar[list[str]] = [
        "Level",
        "J",
        "Parity",
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
        (\S+)      \s+  # g_J
        (\S+)      \s+  # delta g_J
        (\S+)           # total g_J
        \s*$
        """,
        re.VERBOSE,
    )

    def __init__(self, file_path: str | Path) -> None:
        """初始化 g_J 因子文件加载器

        Args:
            file_path: g_J 因子输出文件路径（.gj 文本文件）
        """
        super().__init__(file_path)
        self.nuclear_parameters: NuclearParameters
        self.df: pl.DataFrame
        self.nuclear_parameters, self.df = self._parse_gj_file()

    @staticmethod
    def _parse_grasp_float(value: str) -> float:
        """解析 GRASP 文本输出中的浮点数"""
        normalized_value = value.strip().replace("D", "E").replace("d", "E")
        if normalized_value.lower() == "nan":
            return math.nan
        return float(normalized_value)

    def _parse_gj_file(self) -> tuple[NuclearParameters, pl.DataFrame]:
        """解析 .gj 文本文件

        Returns:
            核参数对象和包含 g_J 因子数据的 DataFrame

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
                    "g_J": self._parse_grasp_float(data_match.group(4)),
                    "delta_g_J": self._parse_grasp_float(data_match.group(5)),
                    "total_g_J": self._parse_grasp_float(data_match.group(6)),
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
                f"Missing nuclear parameter lines in g_J factor file: \n{', '.join(missing_labels)}"
            )
        if not data_rows:
            raise ValueError("No g_J factor data found in file")

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
            包含所有 g_J 因子数据的 DataFrame
        """
        return self.df

    def get_gj_factors(self) -> pl.DataFrame:
        """获取 g_J 因子表格数据

        Returns:
            包含 g_J、delta g_J 和 total g_J 的 DataFrame
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
            g_J 因子表格行数
        """
        return len(self.df)
