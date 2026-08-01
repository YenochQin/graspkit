# -*- encoding: utf-8 -*-
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypedDict, override

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
        parts: list[str] = []
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


class CompositionDataFrameRow(TypedDict):
    coefficient: float
    weight: float
    configuration: str


class LevelDataFrameRow(TypedDict):
    Pos: int
    J: str
    Parity: str
    EnergyTotal: float
    CompOfAsf: str
    compositions: list[CompositionDataFrameRow]


class LSJCompLoader(BaseLoader[pl.DataFrame]):
    """LSJ组成文件加载器

    用于加载 GRASP2018 生成的 LSJ 组成文件（.lsj.lbl）。
    解析能级的 LSJ 组成信息并构建为 polars DataFrame。

    Examples:
        Pos   J   Parity      Energy Total      Comp. of ASF
          1    1     -        -11257.596876908      97.449%
             0.95591698    0.91377727   5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(7)8S0_8S.5d_7D
             0.11744206    0.01379264   5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(7)6P0_6P.5d_5P
    """

    # 正则表达式模式
    LEVEL_LINE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*(\d+)\s+([\d/]+)\s+([+-])\s+(-?\d+\.\d+)\s+([\d.]+)%\s*$"
    )

    COMPOSITION_LINE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(.+?)\s*$"
    )

    def __init__(self, file_path: str | Path) -> None:
        """初始化LSJ文件加载器

        Args:
            file_path: LSJ文件路径（.lsj.lbl 文本文件）
        """
        super().__init__(file_path)
        self._levels: list[LevelComposition] = self._parse_lsj_file()
        self.df: pl.DataFrame = self._levels_to_dataframe(self._levels)

    def _parse_lsj_file(self) -> list[LevelComposition]:
        """解析 .lsj.lbl 文本文件为类型化的能级组成列表。

        Returns:
            类型化的能级组成列表。
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]

        levels: list[LevelComposition] = []
        current_level: LevelComposition | None = None

        for line in lines:
            # 跳过空行
            if not line.strip():
                continue

            # 尝试匹配能级行
            level_match = self.LEVEL_LINE_PATTERN.match(line)
            if level_match:
                # 保存前一个能级的数据
                if current_level is not None:
                    levels.append(current_level)

                # 创建新能级
                current_level = LevelComposition(
                    pos=int(level_match.group(1)),
                    j=level_match.group(2),
                    parity=level_match.group(3),
                    energy_total=float(level_match.group(4)),
                    composition_asf=level_match.group(5),
                    compositions=[],
                )
                continue

            # 尝试匹配组成行
            comp_match = self.COMPOSITION_LINE_PATTERN.match(line)
            if comp_match and current_level is not None:
                current_level.compositions.append(
                    CompositionUnit(
                        coefficient=float(comp_match.group(1)),
                        weight=float(comp_match.group(2)),
                        configuration=comp_match.group(3).strip(),
                    )
                )

        # 保存最后一个能级
        if current_level is not None:
            levels.append(current_level)

        return levels

    @staticmethod
    def _levels_to_dataframe(levels: list[LevelComposition]) -> pl.DataFrame:
        """将类型化的能级组成列表转换为 Polars DataFrame。

        Returns:
            包含能级和组成信息的 DataFrame。
        """
        rows: list[LevelDataFrameRow] = [
            {
                "Pos": level.pos,
                "J": level.j,
                "Parity": level.parity,
                "EnergyTotal": level.energy_total,
                "CompOfAsf": level.composition_asf,
                "compositions": [
                    {
                        "coefficient": composition.coefficient,
                        "weight": composition.weight,
                        "configuration": composition.configuration,
                    }
                    for composition in level.compositions
                ],
            }
            for level in levels
        ]
        return pl.DataFrame(rows)

    @override
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
