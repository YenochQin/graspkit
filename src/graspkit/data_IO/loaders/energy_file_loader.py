# -*- encoding: utf-8 -*-
"""
@Id :energy_file_loader.py
@date :2026/01/19
@author :YenochQin (秦毅)
"""

from pathlib import Path

import polars as pl

from .base_loader import BaseLoader


class EnergyFileLoader(BaseLoader[pl.DataFrame]):
    """能级文件加载器

    用于加载 GRASP2018 生成的能级数据文件（.level 文本格式）。
    直接解析文本文件为 DataFrame，不保存中间 CSV 文件。
    初始化时自动加载数据。
    """

    # 列名定义
    COLUMN_NAMES = [
        "No",
        "Pos",
        "J",
        "Parity",
        "EnergyTotal",
        "EnergyLevel",
        "splitting",
        "configuration_raw",
    ]

    def __init__(self, file_path: str | Path) -> None:
        """初始化能级文件加载器

        初始化时自动加载数据，self.df 始终是有效的 DataFrame。

        Args:
            file_path: 能级文件路径（.level 文本文件）

        Raises:
            ValueError: 文件格式不正确
            IOError: 读取错误
        """
        super().__init__(file_path)
        self.df: pl.DataFrame = self._parse_level_file()

    def _parse_level_file(self) -> pl.DataFrame:
        """解析 .level 文本文件

        Returns:
            能级 DataFrame
        """
        # 读取文本文件
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip() for line in f.readlines()]

        # 查找数据起始行（包含 "No Pos  J " 的行之后第3行）
        data_start_line = 0
        for i, line in enumerate(lines):
            if "No Pos  J " in line:
                data_start_line = i + 3
                break

        # 提取数据行（直到遇到 "-----"）
        data_lines: list[list[str]] = []
        for line in lines[data_start_line:]:
            if "-----" in line:
                break
            stripped_line: str = line.strip()
            if stripped_line:
                data_lines.append(stripped_line.split())

        # 构建数据字典
        data_dict: dict[str, list[str | float]] = {col: [] for col in self.COLUMN_NAMES}

        for line_parts in data_lines:
            for i, col in enumerate[str](self.COLUMN_NAMES):
                if i < len(line_parts):
                    data_dict[col].append(line_parts[i])
                else:
                    data_dict[col].append("")

        # 创建 DataFrame - polars 会自动推断类型
        df: pl.DataFrame = pl.DataFrame(data_dict)

        # 转换数值列 - polars 方式
        int_columns = ["No"]  # Pos 保留为字符串类型
        float_columns = ["EnergyTotal", "EnergyLevel", "splitting"]

        for col in int_columns:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Int64))

        for col in float_columns:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64))

        return df

    def load(self) -> pl.DataFrame:
        """加载数据（实现抽象方法）

        Returns:
            包含所有能级信息的DataFrame
        """
        return self.df

    def get_energy_levels(self) -> pl.DataFrame:
        """获取能级数据

        Returns:
            包含所有能级信息的DataFrame
        """
        return self.df

    def get_total_energy(self) -> float:
        """获取总能量

        Returns:
            总能量值（原子单位）

        Raises:
            ValueError: 数据格式不正确
        """
        if "EnergyTotal" not in self.df.columns:
            raise ValueError(
                "EnergyTotal column not found in DataFrame. "
                f"Available columns: {list(self.df.columns)}"
            )

        return float(self.df["EnergyTotal"][0])

    def get_number_of_levels(self) -> int:
        """获取能级数量

        Returns:
            能级数量
        """
        return len(self.df)

    def get_configurations(self) -> pl.Series:
        """获取电子组态配置

        Returns:
            电子组态配置Series
        """
        if "configuration_raw" not in self.df.columns:
            raise ValueError(
                f"configuration column not found. "
                f"Available columns: {list(self.df.columns)}"
            )

        return self.df["configuration_raw"]
