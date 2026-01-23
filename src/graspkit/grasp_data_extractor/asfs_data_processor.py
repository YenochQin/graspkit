# -*- encoding: utf-8 -*-
"""
@Id :asfs_data_processor.py
@date :2023/04/28 11:08:58
@author :YenochQin (秦毅)

@version 2.0: Refactored to use polars instead of pandas
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ..data_IO.loaders.energy_file_loader import EnergyFileLoader
from ..data_IO.loaders.lsj_comp_loader import LSJCompLoader
from ..utils.tool_function import LS_shell_full_charged

#######################################################################
# Configuration formatting utilities
#######################################################################


@dataclass(frozen=True)
class IntraCoupled_LS:
    """原子组内耦合LS量子数
    multiplicity: 自旋多重度 2S+1
    L: 轨道角动量对应的字母 S, P, D, F,...
    intra_J: 中间耦合J值，可选
    """

    multiplicity: int  # 2S+1 自旋多重度
    L: str  # 轨道角动量字母：S,P,D,F,...
    intra_J: int | None = None  # 中间耦合J值，可选


@dataclass(frozen=True)
class InterCoupled_LS:
    """原子组间耦合LS量子数
    multiplicity: 自旋多重度 2S+1
    L: 轨道角动量对应的字母
    """

    multiplicity: int  # 2S+1 自旋多重度
    L: str  # 轨道角动量字母：S,P,D,F,...


@dataclass(frozen=True)
class ShellInfo:
    """原子轨道信息结构体
    包含主量子数、轨道类型、电子数以及LS耦合信息
    """

    n: int  # 主量子数
    shell: str  # 轨道类型：s/p/d/f
    electrons: int | None  # 轨道中的电子数，来自(e)格式，可能缺省
    intra_ls: IntraCoupled_LS | None  # 组内LS耦合信息，可能缺省
    inter_ls: InterCoupled_LS | None  # 组间LS耦合信息，可能缺省


class ShellFormatter:
    """原子轨道格式化器类
    用于解析和格式化原子轨道配置字符串
    支持格式：4f(7)3S0_7P
    其中：
    - 4: 主量子数n
    - f: 轨道类型
    - (7): 轨道中的电子数（可选）
    - 3S0: 组内LS耦合（multiplicity=32, L=S, Parity=0）
    - _7P: 组间LS耦合（multiplicity=7, L=P）
    """

    SUBSHELL_RE = re.compile(
        r"^(?P<n>\d*)"
        r"(?P<shell>[spdfghi])"
        r"(?:\((?P<ele>\d+)\))?"
        r"(?(ele)(?:(?P<intra_coupling>(?P<intra_S>\d+)(?P<intra_L>[SPDFGHIKLMNO])(?P<intra_J>\d+)?))?)"
        r"(?P<inter_coupling>_(?P<inter_S>\d+)(?P<inter_L>[SPDFGHIKLMNO]))?"
        r"$"
    )

    @classmethod
    def parse_subshell(cls, temp_configuration: str) -> ShellInfo:
        """解析原子轨道配置字符串

        Args:
            temp_configuration: 原子轨道配置字符串，如"4f(7)3S0_7P"

        Returns:
            ShellInfo: 包含解析后的原子轨道信息的结构体

        Raises:
            ValueError: 当输入字符串格式不正确时抛出异常
        """
        shell_match = cls.SUBSHELL_RE.fullmatch(temp_configuration)

        if not shell_match:
            raise ValueError(f"Invalid subshell: {temp_configuration}")

        n = int(shell_match["n"])
        shell = shell_match["shell"]
        ele = int(shell_match["ele"]) if shell_match["ele"] else None

        if shell_match["intra_coupling"]:
            intra_ls = IntraCoupled_LS(
                multiplicity=int(shell_match["intra_S"]),
                L=shell_match["intra_L"],
                intra_J=int(shell_match["intra_J"]) if shell_match["intra_J"] else None,
            )
        else:
            intra_ls = None

        if shell_match["inter_coupling"]:
            inter_ls = InterCoupled_LS(
                multiplicity=int(shell_match["inter_S"]),
                L=shell_match["inter_L"],
            )
        else:
            inter_ls = None

        return ShellInfo(
            n=n, shell=shell, electrons=ele, intra_ls=intra_ls, inter_ls=inter_ls
        )

    @staticmethod
    def format_shell(shell_info: ShellInfo) -> str:
        """格式化原子轨道核心部分

        Args:
            shell_info: 原子轨道信息结构体

        Returns:
            str: 格式化后的原子轨道字符串，如"4f^7"或"4f"
        """
        if shell_info.electrons is not None:
            return f"{shell_info.n}{shell_info.shell}^{{{shell_info.electrons}}}"
        return f"{shell_info.n}{shell_info.shell}"

    @staticmethod
    def format_intra_ls(
        intra_ls: IntraCoupled_LS | None, format_to_word_document: bool = False
    ) -> str:
        """格式化组内LS耦合信息为LaTeX格式

        Args:
            intra_ls: 组内LS耦合信息，可能为None
            format_to_word_document: 是否格式化为Word文档兼容的LaTeX

        Returns:
            str: LaTeX格式的LS耦合字符串
        """
        if not intra_ls:
            return r""

        if format_to_word_document:
            if intra_ls.intra_J is not None:
                return rf"( ^{intra_ls.multiplicity}_{intra_ls.intra_J}{intra_ls.L} )"
            return rf"( ^{intra_ls.multiplicity}{intra_ls.L} )"

        if intra_ls.intra_J is not None:
            return (
                rf"(^{intra_ls.multiplicity}_{intra_ls.intra_J}\mathrm{{{intra_ls.L}}})"
            )
        return rf"(^{intra_ls.multiplicity}\mathrm{{{intra_ls.L}}})"

    @staticmethod
    def format_inter_ls(
        inter_ls: InterCoupled_LS, format_to_word_document: bool = False
    ) -> str:
        """格式化组间LS耦合信息为LaTeX格式

        Args:
            inter_ls: 组间LS耦合信息
            format_to_word_document: 是否格式化为Word文档兼容的LaTeX

        Returns:
            str: LaTeX格式的LS耦合字符串
        """
        if format_to_word_document:
            return rf"^{{{inter_ls.multiplicity}}}{inter_ls.L}"

        return rf"^{{{inter_ls.multiplicity}}}\mathrm{{{inter_ls.L}}}"


def format_configuration(
    temp_configuration: str,
    show_full_charged_subshell: bool = False,
    format_to_word_document: bool = False,
) -> tuple[str, str]:
    """格式化GRASP原子配置字符串

    使用ShellFormatter来解析和格式化原子配置字符串
    支持处理多个子轨道的组合配置，如"4f(7)3S0_7P.5d(3)2F_5G"

    Args:
        temp_configuration: 原子配置字符串，可包含换行符
        show_full_charged_subshell: 是否显示满电子子轨道，默认False
        format_to_word_document: 是否格式化为Word文档兼容的LaTeX，默认False

    Returns:
        Tuple[str, str]: (格式化的配置字符串, 组间LS耦合字符串)
    """
    temp_configuration = re.sub(r"\n", "", temp_configuration)
    temp_conf_list = temp_configuration.split(".")

    formatted_conf = r""
    format_LS_compling = r""

    list_length = len(temp_conf_list)
    for index, shell in enumerate(temp_conf_list):
        formated_shell = ShellFormatter.parse_subshell(shell)
        is_last = index == list_length - 1
        temp_shell = formated_shell.shell

        if (formated_shell.electrons is not None) and (not is_last):
            temp_electrons = formated_shell.electrons

            if (
                LS_shell_full_charged(temp_shell, temp_electrons)
                and not show_full_charged_subshell
            ):
                continue

        formatted_conf = (
            formatted_conf
            + ShellFormatter.format_shell(formated_shell)
            + r"\,"
            + (
                ShellFormatter.format_intra_ls(
                    formated_shell.intra_ls, format_to_word_document
                )
                or ""
            )
            + r"\;"
        )

        if is_last and formated_shell.inter_ls is not None:
            format_LS_compling = ShellFormatter.format_inter_ls(
                formated_shell.inter_ls, format_to_word_document
            )
    formatted_conf = formatted_conf.replace(r"\,\;", r"\;")
    return formatted_conf, format_LS_compling


#######################################################################
# Energy data processing functions (polars)
#######################################################################


def format_energy_configurations(
    energy_df: pl.DataFrame,
    show_full_charged_subshell: bool = False,
    format_to_word_document: bool = False,
) -> pl.DataFrame:
    """格式化能级数据中的配置列

    将 configuration_raw 列格式化为 configuration 和 LSJ 列

    Args:
        energy_df: 能级DataFrame，必须包含 configuration_raw 列
        show_full_charged_subshell: 是否显示满电子子轨道
        format_to_word_document: 是否格式化为Word文档兼容的LaTeX

    Returns:
        添加了 configuration 和 LSJ 列的DataFrame
    """
    if "configuration_raw" not in energy_df.columns:
        raise ValueError(
            f"configuration_raw column not found in DataFrame. "
            f"Available columns: {energy_df.columns}"
        )

    # 应用 format_configuration 函数
    def _format_conf(config_str: str) -> tuple[str, str]:
        if config_str is None or config_str == "":
            return "", ""
        return format_configuration(
            config_str, show_full_charged_subshell, format_to_word_document
        )

    # 使用 map_elements 应用格式化函数
    formatted_result = energy_df["configuration_raw"].map_elements(
        _format_conf, return_dtype=pl.List(pl.Utf8)
    )

    # 分离结果为两列
    energy_df = energy_df.with_columns(
        [
            formatted_result.list.get(0).alias("configuration"),
            formatted_result.list.get(1).alias("LSJ"),
        ]
    )

    return energy_df


#######################################################################
# Composition formatting functions (polars)
#######################################################################


def format_compositions(
    energy_df: pl.DataFrame,
    show_full_charged_subshell: bool = False,
    format_to_word_document: bool = False,
) -> pl.DataFrame:
    """格式化能级数据中的组成列

    将所有 compositions_raw_* 列合并为一个 compositions_raw 列，
    然后格式化为 Comp_of_asf 列中的LaTeX字符串

    Args:
        energy_df: 能级DataFrame，包含 compositions_raw_* 列
        show_full_charged_subshell: 是否显示满电子子轨道
        format_to_word_document: 是否格式化为Word文档兼容的LaTeX

    Returns:
        添加了 compositions_raw 和 Comp_of_asf 列的DataFrame

    Raises:
        ValueError: 如果同一行有多个 compositions_raw_* 列包含数据
    """
    # 查找所有 compositions_raw_* 列
    comp_columns = [
        col for col in energy_df.columns if col.startswith("compositions_raw_")
    ]

    if not comp_columns:
        # 如果没有组成列，返回原DataFrame
        return energy_df

    def _merge_compositions(row: dict, row_idx: int) -> list[dict[str, Any]]:
        """合并多个组成列表为一个

        Args:
            row: DataFrame 的行数据（字典形式）
            row_idx: 行索引（用于错误信息）

        Returns:
            合并后的组成数据列表

        Raises:
            ValueError: 如果同一行有多个 compositions_raw_* 列包含数据
        """
        non_empty_cols = []
        merged = []

        for col in comp_columns:
            comp_data = row.get(col)
            if comp_data is not None and len(comp_data) > 0:
                non_empty_cols.append(col)
                merged.extend(comp_data)

        # 检查是否有冲突
        if len(non_empty_cols) > 1:
            raise ValueError(
                f"第 {row_idx} 行存在数据冲突：多个列包含组成数据 {non_empty_cols}。"
                f"同一能级只能有一个组态的组成数据。"
            )

        return merged

    def _format_single_composition(
        comp_data: list[dict[str, Any]] | None,
    ) -> str:
        """格式化单个能级的组成数据

        Args:
            comp_data: 组成数据列表，每个元素包含 ci_coeff, weight, configuration

        Returns:
            格式化的LaTeX字符串
        """
        if comp_data is None or len(comp_data) == 0:
            return ""

        parts = []
        for comp in comp_data:
            weight = comp.get("weight", 0.0)
            configuration = comp.get("configuration", "")

            # 格式化配置
            formatted_conf, formatted_ls = format_configuration(
                configuration, show_full_charged_subshell, format_to_word_document
            )

            # 构建LaTeX字符串
            if formatted_conf and formatted_ls:
                part = rf"${weight:.3f}\;{formatted_conf}\,{formatted_ls}$"
            elif formatted_conf:
                part = rf"${weight:.3f}\;{formatted_conf}$"
            elif formatted_ls:
                part = rf"${weight:.3f}\;{formatted_ls}$"
            else:
                continue

            parts.append(part)

        return " + ".join(parts)

    # 1. 合并所有 compositions_raw_* 列为一个新的 compositions_raw 列
    # 先收集所有行的合并数据
    merged_data = []
    for idx, row in enumerate(energy_df.iter_rows(named=True)):
        merged_data.append(_merge_compositions(row, idx))

    # 添加合并后的 compositions_raw 列
    energy_df = energy_df.with_columns(pl.Series("compositions_raw", merged_data))

    # 2. 格式化合并后的 compositions_raw 列为 Comp_of_asf
    formatted_comp = energy_df["compositions_raw"].map_elements(
        _format_single_composition, return_dtype=pl.Utf8, skip_nulls=True
    )

    energy_df = energy_df.with_columns(formatted_comp.alias("Comp_of_asf"))

    return energy_df


#######################################################################
# Legacy functions for backward compatibility
#######################################################################


def mcdhf_energy_data_collection(
    data_file_info: dict, a_s_list: list[int], show_full_charged_subshell: bool = False
) -> pl.DataFrame:
    """合并不同AS循环的能级数据

    Args:
        data_file_info: 文件信息字典
        a_s_list: AS循环列表
        show_full_charged_subshell: 是否显示满电子子轨道

    Returns:
        合并后的DataFrame
    """
    file_dir = data_file_info.get("file_dir", "")
    atom = data_file_info.get("atom", "")
    level_parameter = data_file_info.get("level_parameter", "")

    # 加载第一个AS数据
    file_path = f"{file_dir}/{atom}{level_parameter}{a_s_list[0]}"
    energy_data = EnergyFileLoader(file_path).load()
    energy_data = format_energy_configurations(energy_data, show_full_charged_subshell)

    # 重命名列
    energy_data = energy_data.rename(
        {
            "EnergyTotal": f"Energy_Total_{level_parameter}{a_s_list[0]}",
            "EnergyLevel": f"E_as{a_s_list[0]}",
            "splitting": f"Splitting_{a_s_list[0]}",
        }
    )

    # 合并其他AS数据
    for a_s in a_s_list[1:]:
        file_path = f"{file_dir}/{atom}{level_parameter}{a_s}"
        temp_df = EnergyFileLoader(file_path).load()
        temp_df = format_energy_configurations(temp_df, show_full_charged_subshell)

        temp_df = temp_df.rename(
            {
                "EnergyTotal": f"Energy_Total_{level_parameter}{a_s}",
                "EnergyLevel": f"E_as{a_s}",
                "splitting": f"Splitting_{a_s}",
            }
        )

        # 合并
        energy_data = energy_data.join(
            temp_df, on=["Pos", "J", "Parity"], how="outer", coalesce=True
        )

        # 计算能量差
        energy_data = energy_data.with_columns(
            (pl.col(f"E_as{a_s}") - pl.col(f"E_as{a_s - 1}")).alias(f"dE{a_s}")
        )

    return energy_data.fill_null(0)


def ci_energy_data_collection(
    energy_data: pl.DataFrame | None,
    data_file_info: dict,
    show_full_charged_subshell: bool = False,
) -> pl.DataFrame:
    """收集单个能级数据或合并CI计算的能级数据

    Args:
        energy_data: 现有能级数据，如果为None则创建新的
        data_file_info: 文件信息字典
        show_full_charged_subshell: 是否显示满电子子轨道

    Returns:
        合并后的DataFrame
    """
    file_dir = data_file_info.get("file_dir", "")
    atom = data_file_info.get("atom", "")
    level_parameter = data_file_info.get("level_parameter", "")
    this_as = data_file_info.get("this_as", 0)

    file_path = f"{file_dir}/{atom}{level_parameter}{this_as}"
    temp_df = EnergyFileLoader(file_path).load()
    temp_df = format_energy_configurations(temp_df, show_full_charged_subshell)

    if energy_data is None:
        return temp_df

    # 合并
    result = energy_data.join(
        temp_df, on=["Pos", "J", "Parity"], how="outer", coalesce=True
    )

    return result.fill_null(0).sort("No")


def level_energy_collector(
    data_file_path: str | Path,
    show_full_charged_subshell: bool = False,
) -> pl.DataFrame:
    """从文件路径收集能级数据

    Args:
        data_file_path: 能级文件路径
        show_full_charged_subshell: 是否显示满电子子轨道

    Returns:
        格式化后的能级DataFrame
    """
    energy_loader = EnergyFileLoader(data_file_path)
    level_data = energy_loader.load()
    level_data = format_energy_configurations(level_data, show_full_charged_subshell)

    return level_data


#######################################################################
# LSJ composition merging functions
#######################################################################


def merge_lsj_compositions(
    energy_df: pl.DataFrame,
    lsj_file_paths: list[str],
    min_comp: float = 0.01,
    show_comp_num: int = 0,
    show_full_charged_subshell: bool = False,
    format_to_word_document: bool = False,
) -> pl.DataFrame:
    """将LSJ组成文件合并到能级DataFrame

    读取所有LSJ文件，合并为一个DataFrame，然后按 "Pos", "J", "Parity" 匹配并 join 到能级 DataFrame

    Args:
        energy_df: 能级DataFrame，必须包含 Pos, J, Parity 列
        lsj_file_paths: LSJ组成文件路径列表
        min_comp: 最小权重阈值（当show_comp_num=0时使用，默认 0.0001）
        show_comp_num: 显示的组成数量（0表示按权重过滤，默认 0）
        show_full_charged_subshell: 是否显示满电子子轨道
        format_to_word_document: 是否格式化为Word文档兼容的LaTeX

    Returns:
        合并后的能级DataFrame，包含 CompOfAsf 列
    """
    # 收集所有LSJ数据到一个列表
    all_lsj_data: list[dict[str, str]] = []

    for lsj_path in lsj_file_paths:
        # 加载LSJ文件（只负责读取）
        lsj_loader = LSJCompLoader(lsj_path)

        # 获取格式化后的组成字符串（在这里做过滤和格式化）
        for level in lsj_loader.get_levels():
            # 过滤组成
            level_comps = []
            count = 0
            for comp in level.compositions:
                # 过滤逻辑
                if show_comp_num == 0:
                    if comp.weight <= min_comp:
                        continue
                else:
                    if count >= show_comp_num:
                        continue
                    count += 1
                level_comps.append(comp)

            # 格式化为LaTeX字符串
            if not level_comps:
                comp_str = ""
            else:
                parts = []
                for comp in level_comps:
                    weight = comp.weight
                    configuration = comp.configuration

                    # 格式化配置
                    formatted_conf, formatted_ls = format_configuration(
                        configuration,
                        show_full_charged_subshell=show_full_charged_subshell,
                        format_to_word_document=format_to_word_document,
                    )

                    # 构建LaTeX字符串
                    if formatted_conf and formatted_ls:
                        part = rf"${weight:.3f}\;{formatted_conf}\,{formatted_ls}$"
                    elif formatted_conf:
                        part = rf"${weight:.3f}\;{formatted_conf}$"
                    elif formatted_ls:
                        part = rf"${weight:.3f}\;{formatted_ls}$"
                    else:
                        continue

                    parts.append(part)

                comp_str = " + ".join(parts)

            # 添加到总列表
            all_lsj_data.append(
                {
                    "Pos": str(level.pos),
                    "J": level.j,
                    "Parity": level.parity,
                    "CompOfAsf": comp_str,
                }
            )

    # 创建合并后的LSJ DataFrame
    if all_lsj_data:
        lsj_df = pl.DataFrame(all_lsj_data)
    else:
        # 如果没有数据，返回空DataFrame（只有列名）
        lsj_df = pl.DataFrame(
            schema={
                "Pos": pl.Utf8,
                "J": pl.Utf8,
                "Parity": pl.Utf8,
                "CompOfAsf": pl.Utf8,
            }
        )

    # 确保类型匹配
    lsj_df = lsj_df.with_columns(
        pl.col("Pos").cast(pl.Utf8),
        pl.col("J").cast(pl.Utf8),
        pl.col("Parity").cast(pl.Utf8),
    )

    # 一次性 join 到能级 DataFrame
    result_df = energy_df.join(lsj_df, on=["Pos", "J", "Parity"], how="left")

    return result_df
