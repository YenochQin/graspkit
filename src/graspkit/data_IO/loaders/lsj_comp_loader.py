# -*- encoding: utf-8 -*-
"""
@Id :lsj_comp_loader.py
@date :2026/01/19
@author :YenochQin (秦毅)
"""

import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .base_loader import BaseLoader


@dataclass(frozen=True)
class CompositionUnit:
    """单个组成单元

    Attributes:
        coefficient: CI系数
        weight: CI系数平方（权重百分比）
        configuration: LS组态配置字符串
    """

    coefficient: float
    weight: float
    configuration: str


@dataclass
class LevelComposition:
    """单个能级的组成信息

    Attributes:
        pos: 能级位置索引
        j: J量子数
        parity: 宇称 (+/-)
        energy_total: 总能量
        composition_asf: ASF组成百分比
        compositions: 组成单元列表
    """

    pos: int
    j: str
    parity: str
    energy_total: float
    composition_asf: str
    compositions: list[CompositionUnit]

    def format_composition(self, min_comp: float = 0.01, show_comp_num: int = 0) -> str:
        """格式化组成为字符串

        Args:
            min_comp: 最小权重阈值（当show_comp_num=0时使用）
            show_comp_num: 显示的组成数量（0表示按权重过滤）

        Returns:
            格式化的组成字符串
        """
        parts = []
        count = 0

        for comp in self.compositions:
            # 过滤逻辑
            if show_comp_num == 0:
                if comp.weight <= min_comp:
                    continue
            else:
                if count >= show_comp_num:
                    continue
                count += 1

            # 格式: 权重 配置
            parts.append(f"{comp.weight:.3f} {comp.configuration}")

        return " + ".join(parts)


class LSJCompLoader(BaseLoader[pl.DataFrame]):
    """LSJ组成文件加载器

    用于加载 GRASP2018 生成的 LSJ 组成文件（.lsj.lbl）。
    解析能级的 LSJ 组成信息并构建为 polars DataFrame。

    文件格式示例:
        Pos   J   Parity      Energy Total      Comp. of ASF
          1    1     -        -11257.596876908      97.449%
             0.95591698    0.91377727   5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(7)8S0_8S.5d_7D
             0.11744206    0.01379264   5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(7)6P0_6P.5d_5P
    """

    # 正则表达式模式
    LEVEL_LINE_PATTERN = re.compile(
        r"^\s*(\d+)\s+"  # Pos (integer)
        r"([\d/]+)\s+"  # J (could be 1, 3/2, etc.)
        r"([+-])\s+"  # Parity (+ or -)
        r"(-?\d+\.\d+)\s+"  # Energy Total (float)
        r"([\d.]+)%\s*$"  # Comp. of ASF (percentage)
    )

    COMPOSITION_LINE_PATTERN = re.compile(
        r"^\s+"  # Must start with whitespace (not a level line)
        r"(-?\d+\.\d+)\s+"  # CI coefficient (float)
        r"(-?\d+\.\d+)\s+"  # CI coefficient squared / weight (float)
        r"(.+?)\s*$"  # Configuration string
    )

    def __init__(self, file_path: str | Path) -> None:
        """初始化LSJ文件加载器

        Args:
            file_path: LSJ文件路径（.lsj.lbl 文本文件）
        """
        super().__init__(file_path)
        self.df: pl.DataFrame = self._parse_lsj_file()
        self._levels: list[LevelComposition] = self._build_level_compositions()

    def _parse_lsj_file(self) -> pl.DataFrame:
        """解析 .lsj.lbl 文本文件为 polars DataFrame

        Returns:
            包含能级和组成信息的DataFrame
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]

        data_rows = []
        current_level = None

        for line in lines:
            # 跳过空行
            if not line.strip():
                continue

            # 尝试匹配能级行
            level_match = self.LEVEL_LINE_PATTERN.match(line)
            if level_match:
                # 保存前一个能级的数据
                if current_level is not None:
                    data_rows.append(current_level)

                # 创建新能级
                current_level = {
                    "Pos": int(level_match.group(1)),
                    "J": level_match.group(2),
                    "Parity": level_match.group(3),
                    "EnergyTotal": float(level_match.group(4)),
                    "CompOfAsf": level_match.group(5),
                    "compositions": [],
                }
                continue

            # 尝试匹配组成行
            comp_match = self.COMPOSITION_LINE_PATTERN.match(line)
            if comp_match and current_level is not None:
                current_level["compositions"].append(
                    {
                        "coefficient": float(comp_match.group(1)),
                        "weight": float(comp_match.group(2)),
                        "configuration": comp_match.group(3).strip(),
                    }
                )

        # 保存最后一个能级
        if current_level is not None:
            data_rows.append(current_level)

        # 构建 DataFrame
        df_data = []
        for row in data_rows:
            df_data.append(
                {
                    "Pos": row["Pos"],
                    "J": row["J"],
                    "Parity": row["Parity"],
                    "EnergyTotal": row["EnergyTotal"],
                    "CompOfAsf": row["CompOfAsf"],
                    "compositions": row["compositions"],
                }
            )

        return pl.DataFrame(df_data)

    def _build_level_compositions(self) -> list[LevelComposition]:
        """从DataFrame构建LevelComposition对象列表

        Returns:
            LevelComposition对象列表
        """
        levels = []
        for row in self.df.iter_rows(named=True):
            compositions = [
                CompositionUnit(
                    coefficient=comp["coefficient"],
                    weight=comp["weight"],
                    configuration=comp["configuration"],
                )
                for comp in row["compositions"]
            ]

            levels.append(
                LevelComposition(
                    pos=row["Pos"],
                    j=row["J"],
                    parity=row["Parity"],
                    energy_total=row["EnergyTotal"],
                    composition_asf=row["CompOfAsf"],
                    compositions=compositions,
                )
            )

        return levels

    def load(self) -> pl.DataFrame:
        """加载数据（实现抽象方法）

        Returns:
            包含所有能级和组成信息的DataFrame
        """
        return self.df

    def get_levels(self) -> list[LevelComposition]:
        """获取能级组成对象列表

        Returns:
            LevelComposition对象列表
        """
        return self._levels

    def get_level_by_position(
        self, pos: int, j: str | None = None, parity: str | None = None
    ) -> LevelComposition | None:
        """根据位置获取能级

        Args:
            pos: 能级位置
            j: J值（可选，用于更精确的匹配）
            parity: 宇称（可选，用于更精确的匹配）

        Returns:
            匹配的LevelComposition对象，未找到返回None
        """
        for level in self._levels:
            if level.pos == pos:
                if j is not None and level.j != j:
                    continue
                if parity is not None and level.parity != parity:
                    continue
                return level
        return None

    def to_dataframe_with_formatted_composition(
        self, min_comp: float = 0.01, show_comp_num: int = 0
    ) -> pl.DataFrame:
        """返回包含格式化组成字符串的DataFrame

        Args:
            min_comp: 最小权重阈值，默认 0.01
            show_comp_num: 显示的组成数量，0表示按权重过滤，默认 0

        Returns:
            包含格式化组成列的DataFrame
        """
        formatted_compositions = [
            level.format_composition(min_comp, show_comp_num) for level in self._levels
        ]

        return self.df.with_columns(
            pl.Series("formatted_composition", formatted_compositions)
        )
