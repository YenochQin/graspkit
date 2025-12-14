# -*- encoding: utf-8 -*-
"""
@Id :level_data_collection.py
@date :2023/04/28 11:08:58
@author :YenochQin (秦毅)

@version 1.0: Object Oriented Programming modified from level_data_collection.py
"""

import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from ..data_IO.grasp_data_loader import GraspFileLoad, EnergyFile2csv
from ..utils.tool_function import LS_shell_full_charged


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
    intra_ls: IntraCoupled_LS | None # 组内LS耦合信息，可能缺省
    inter_ls: InterCoupled_LS | None # 组间LS耦合信息，可能缺省


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
        # 使用正则表达式完全匹配输入字符串
        shell_match = cls.SUBSHELL_RE.fullmatch(temp_configuration)

        # 如果匹配失败，抛出异常
        if not shell_match:
            raise ValueError(f"Invalid subshell: {temp_configuration}")

        # 提取基本轨道信息
        n = int(shell_match["n"])  # 主量子数
        shell = shell_match["shell"]  # 轨道类型
        ele = (
            int(shell_match["ele"]) if shell_match["ele"] else None
        )  # 电子数，如果存在则转换为整数

        # 解析组内LS耦合信息（如果存在）
        if shell_match["intra_coupling"]:
            intra_ls = IntraCoupled_LS(
                multiplicity=int(shell_match["intra_S"]),  # 自旋多重度
                L=shell_match["intra_L"],  # 轨道角动量字母
                intra_J=int(shell_match["intra_J"])
                if shell_match["intra_J"]
                else None,  # 中间耦合J值，如果存在则转换
            )
        else:
            intra_ls = None

        # 解析组间LS耦合信息（如果存在）
        if shell_match["inter_coupling"]:
            inter_ls = InterCoupled_LS(
                multiplicity=int(shell_match["inter_S"]),  # 自旋多重度
                L=shell_match["inter_L"],  # 轨道角动量字母
            )
        else:
            inter_ls = None

        # 返回包含所有解析信息的ShellInfo结构体
        return ShellInfo(
            n=n, shell=shell, electrons=ele, intra_ls=intra_ls, inter_ls=inter_ls
        )

    @staticmethod
    def format_shell(
                    shell_info: ShellInfo
                    ) -> str:
        """格式化原子轨道核心部分

        Args:
            shell_info: 原子轨道信息结构体

        Returns:
            str: 格式化后的原子轨道字符串，如"4f^7"或"4f"
        """
        # 如果存在电子数，使用上标格式，否则只返回基本轨道信息
        if shell_info.electrons is not None:
            return f"{shell_info.n}{shell_info.shell}^{{{shell_info.electrons}}}"
        return f"{shell_info.n}{shell_info.shell}"

    @staticmethod
    def format_intra_ls(
                        intra_ls: IntraCoupled_LS | None,
                        format_to_word_document: bool = False
                    ) -> str:
        """格式化组内LS耦合信息为LaTeX格式

        Args:
            intra_ls: 组内LS耦合信息，可能为None

        Returns:
            str: LaTeX格式的LS耦合字符串，如"(^32_0\\text{S})"
            如果format_to_word_document为真则取消了\\text{},并在特定地方加入了空格，
            可以直接在word公式latex形式从"线性"转为"专业"
        """
        # 如果不存在LS耦合信息，返回空字符串
        if not intra_ls:
            return ""

        # 添加对office word中的latex形式公式的支持，此形式应该可以直接从"线性"转为"专业"
        if format_to_word_document:
            if intra_ls.intra_J is not None:
                return f"( ^{intra_ls.multiplicity}_{intra_ls.intra_J}{intra_ls.L} )"
            # 否则只格式化多重度角动量
            return f"( ^{intra_ls.multiplicity}{intra_ls.L} )"
        
        # 如果存在中间耦合J值，包含在格式中
        if intra_ls.intra_J is not None:
            return f"(^{intra_ls.multiplicity}_{intra_ls.intra_J}\\text{{{intra_ls.L}}})"
        # 否则只格式化多重度角动量
        return f"(^{intra_ls.multiplicity}\\text{{{intra_ls.L}}})"

    @staticmethod
    def format_inter_ls(
                        inter_ls: InterCoupled_LS,
                        format_to_word_document: bool = False
                        ) -> str:
        """格式化组间LS耦合信息为LaTeX格式

        Args:
            inter_ls: 组间LS耦合信息

        Returns:
            str: LaTeX格式的LS耦合字符串，如"^7\\text{P}"
        """
        if format_to_word_document:
            return f"^{{{inter_ls.multiplicity}}}{inter_ls.L}"

        return f"^{{{inter_ls.multiplicity}}}\\text{{{inter_ls.L}}}"


class ConfigurationFormatter:
    """
    GRASP能级格式化器类

    使用ShellFormatter来解析和格式化原子配置字符串
    支持处理多个子轨道的组合配置，如"4f(7)3S0_7P.5d(3)2F_5G"

    Attributes:
        temp_configuration: 清理后的配置字符串（去除换行符）
        show_full_charged_subshell: 是否显示满电子子轨道的布尔值
        temp_conf_list: 按.分割的子轨道配置列表
    """

    def __init__(
                self, 
                temp_configuration: str, 
                show_full_charged_subshell: bool = False,
                format_to_word_document: bool = False
                ):
        """
        初始化ConfigurationFormatter

        Args:
            temp_configuration: 原子配置字符串，可包含换行符
            show_full_charged_subshell: 是否显示满电子子轨道，默认False
        """
        self.temp_configuration = re.sub(r"\n", "", temp_configuration)
        self.temp_conf_list = self.temp_configuration.split(".")

        self.show_full_charged_subshell = show_full_charged_subshell
        self.format_to_word_document = format_to_word_document

    def conf_format(self) -> Tuple[str, str]:
        """格式化整个原子配置

        处理所有子轨道，过滤满电子子轨道（如果需要），并格式化为LaTeX字符串

        Returns:
            Tuple[str, str]: (格式化的配置字符串, 原始配置字符串)
                           例如: ("4f^{7}\\,(^3_0\\text{S})\\;5d^{3}\\,(^4_5\\text{F})\\;",
                                 "4f(7)3S0_7P.5d(3)4F5_5G")
        """
        formatted_conf = ""
        format_LS_compling = ""

        list_length = len(self.temp_conf_list)
        for index, shell in enumerate(self.temp_conf_list):
            # 检查是否是满电子子轨道
            formated_shell = ShellFormatter.parse_subshell(shell)
            is_last = index == list_length - 1
            temp_shell = formated_shell.shell

            if (formated_shell.electrons is not None) and (not is_last):
                temp_electrons = formated_shell.electrons

                if LS_shell_full_charged(temp_shell, temp_electrons) and not self.show_full_charged_subshell:
                    continue

            formatted_conf = (
                formatted_conf
                + ShellFormatter.format_shell(formated_shell)
                + "\\,"
                + (ShellFormatter.format_intra_ls(formated_shell.intra_ls, self.format_to_word_document) or "")
                + "\\;"
            )

            if is_last and formated_shell.inter_ls is not None:

                format_LS_compling = ShellFormatter.format_inter_ls(
                    formated_shell.inter_ls,
                    self.format_to_word_document
                )
        formatted_conf = formatted_conf.replace(r"\,\;", r"\;")
        return formatted_conf, format_LS_compling


#######################################################################


class LevelsEnergyData:
    """
    This class is used to read the energy data from the grasp output file and format the data.
    """

    @classmethod
    def from_filepath(
        cls,
        filepath,
        store_csv_path: str = "",
        show_full_charged_subshell: bool = False,
        format_to_word_document: bool = False
    ):
        """从文件路径直接创建实例的类方法"""
        file_dir = str(Path(filepath).parent)
        file_name = Path(filepath).name
        config = {
            "atom": "",
            "file_dir": file_dir,
            "file_name": file_name,
            "level_parameter": "",
            "this_as": 0,
            "file_type": "ENERGY",
            "store_csv_path": store_csv_path,
            "show_full_charged_subshell": show_full_charged_subshell,
            "format_to_word_document": format_to_word_document,
        }
        return cls(config)

    def __init__(self, data_file_info):
        self.data_file_info = data_file_info
        # self.f_type = "energy"
        self.data_file_info["f_type"] = "energy"
        self.level_parameter = data_file_info.get("level_parameter")
        self.atom = data_file_info.get("atom")
        self.this_as = data_file_info.get("this_as")
        self.show_full_charged_subshell = data_file_info.get(
            "show_full_charged_subshell", False
        )
        self.format_to_word_document = data_file_info.get(
            "format_to_word_document", False
        )
        self.level_read_df = pd.DataFrame(
            columns=[
                "No",
                "Pos",
                "J",
                "Parity",
                f"Energy_Total_{self.level_parameter}{self.this_as}",
                f"E_as{self.this_as}",
                "Splitting",
                f"Configuration_{self.level_parameter}{self.this_as}raw",
            ]
        )

        self.file_dir = data_file_info.get("file_dir")

        self.file_name = f"{self.atom}{self.level_parameter}{self.this_as}"
        self.raw_data2csv = EnergyFile2csv(self.data_file_info)

    def energy_file2dataframe(self):
        self.saved_csv_file_path = self.raw_data2csv.energy_file2csv()

        # self.level_read_df = pd.read_csv(f"{self.saved_csv_file_path}", sep=r'\s+', names=['No', 'Pos', 'J', 'Parity', f'Energy_Total_{self.level_parameter}{self.this_as}', f'E_as{self.this_as}', 'Splitting', f'Configuration_{self.level_parameter}{self.this_as}raw'], dtype=str)
        self.level_read_df = pd.read_csv(
            f"{self.saved_csv_file_path}",
            header=0,
            names=[
                "No",
                "Pos",
                "J",
                "Parity",
                f"Energy_Total_{self.level_parameter}{self.this_as}",
                f"E_as{self.this_as}",
                "Splitting",
                f"Configuration_{self.level_parameter}{self.this_as}raw",
            ],
            dtype=str,
        )

        return self.level_read_df

    def energy_level_2_pd(self):
        self.saved_csv_file_path = self.raw_data2csv.energy_file2csv()

        # self.level_read_df = pd.read_csv(f"{self.saved_csv_file_path}", sep=r'\s+', names=['No', 'Pos', 'J', 'Parity', f'Energy_Total_{self.level_parameter}{self.this_as}', f'E_as{self.this_as}', 'Splitting', f'Configuration_{self.level_parameter}{self.this_as}raw'], dtype=str)
        self.level_read_df = pd.read_csv(f"{self.saved_csv_file_path}", header=0)

        return self.level_read_df

    def energy_data_formate(self):
        self.energy_file2dataframe()
        if (
            not self.level_read_df[
                f"Configuration_{self.level_parameter}{self.this_as}raw"
            ]
            .isnull()
            .all()
        ):
            self.level_read_df[
                f"Configuration_{self.level_parameter}{self.this_as}"
            ] = self.level_read_df[
                f"Configuration_{self.level_parameter}{self.this_as}raw"
            ].apply(
                lambda x: ConfigurationFormatter(
                    x, self.show_full_charged_subshell, self.format_to_word_document
                ).conf_format()[0]
            )

            self.level_read_df[
                f"Configuration_LSJ_{self.level_parameter}_as{self.this_as}"
            ] = (
                self.level_read_df[
                    f"Configuration_{self.level_parameter}{self.this_as}raw"
                ].apply(
                    lambda x: ConfigurationFormatter(
                        x, self.show_full_charged_subshell, self.format_to_word_document
                    ).conf_format()[1]
                + "_{"
                + self.level_read_df["J"]
                + "}"
            ))

            self.level_read_df[f"ASF_LSJ_as{self.this_as}"] = (
                "$"
                + self.level_read_df[
                    f"Configuration_{self.level_parameter}{self.this_as}"
                ]
                + self.level_read_df[
                    f"Configuration_LSJ_{self.level_parameter}_as{self.this_as}"
                ]
                + "$"
            )

        # self.level_read_df[[f'E_as{self.this_as}', 'Splitting']].fillna(0, inplace=True)
        self.level_read_df[
            [
                "No",
                f"Energy_Total_{self.level_parameter}{self.this_as}",
                f"E_as{self.this_as}",
            ]
        ] = self.level_read_df[
            [
                "No",
                f"Energy_Total_{self.level_parameter}{self.this_as}",
                f"E_as{self.this_as}",
            ]
        ].apply(pd.to_numeric)

        return self.level_read_df


#######################################################################


def mcdhf_energy_data_collection(
    data_file_info, a_s_list, show_full_charged_subshell: bool = False
):
    """
    This function is used to merge the energy data from different a_s.
    """
    data_file_info["this_as"] = a_s_list[0]
    data_file_info["show_full_charged_subshell"] = show_full_charged_subshell
    energy_data = LevelsEnergyData(data_file_info).energy_data_formate()
    for a_s in a_s_list[1:]:
        data_file_info["file"] = (
            f"{data_file_info['atom']}{data_file_info['level_parameter']}{a_s}"
        )
        data_file_info["this_as"] = a_s
        temp_level_as = LevelsEnergyData(data_file_info).energy_data_formate()
        energy_data = pd.merge(
            energy_data,
            temp_level_as,
            how="outer",
            on=["Pos", "J", "Parity"],
            suffixes=("", str(a_s)),
        )
        energy_data[f"dE{a_s}"] = (
            energy_data[f"E_as{a_s}"] - energy_data[f"E_as{a_s - 1}"]
        )
        energy_data[f"dE{a_s}per"] = (
            energy_data[f"dE{a_s}"] / energy_data[f"E_as{a_s - 1}"]
        )
    energy_data = energy_data.dropna(how="all", axis=1)
    energy_data = energy_data.fillna(0)
    return energy_data


def ci_energy_data_collection(
    energy_data: pd.DataFrame | None,
    data_file_info: dict,
    show_full_charged_subshell: bool = False,
):
    """
    This function is used to collect single energy levels data or merge energy levels data from ci calculation.
    """
    data_file_info["show_full_charged_subshell"] = show_full_charged_subshell
    if energy_data is None:
        energy_data = LevelsEnergyData(data_file_info).energy_data_formate()
    else:
        temp_ci_level_as = LevelsEnergyData(data_file_info).energy_data_formate()
        energy_data = pd.merge(
            energy_data,
            temp_ci_level_as,
            how="outer",
            on=["Pos", "J", "Parity"],
            suffixes=(
                "",
                str(data_file_info["level_parameter"]) + str(data_file_info["this_as"]),
            ),
        )
        energy_data = energy_data.dropna(how="all", axis=1)
        energy_data = energy_data.fillna(0.0)
        energy_data = energy_data.sort_values(
            by=f"No{data_file_info['level_parameter']}{data_file_info['this_as']}",
            ascending=True,
        )
    return energy_data


def level_energy_collector(
    data_file_path: str | Path,
    store_csv_path: str = "",
    show_full_charged_subshell: bool = False,
):
    """
    This function is used to collect single energy levels data or merge energy levels data from ci calculation.
    """

    temp_level_load = LevelsEnergyData.from_filepath(
        data_file_path, store_csv_path, show_full_charged_subshell
    )
    level_data = temp_level_load.energy_data_formate()

    return level_data


#######################################################################

# Add level's composition of ASF


class LevelsASFComposition:
    def __init__(
        self,
        energy_data_df: pd.DataFrame,
        data_file_info: dict,
        min_comp: float = 0.03,
        show_comp_num: int = 0,
        show_full_charged_subshell: bool = False,
        format_to_word_document: bool = False,
    ):
        self.energy_data_df = energy_data_df
        self.data_file_info = data_file_info
        # TODO 这里有bug
        self.show_full_charged_subshell = show_full_charged_subshell
        self.format_to_word_document = format_to_word_document
        # self.show_full_charged_subshell = data_file_info.get(
        #     "show_full_charged_subshell", False
        # )
        # self.format_to_word_document = data_file_info.get(
        #     "format_to_word_document", False
        # )
        self.data_file_info["file_type"] = "LSJ"
        self.data_file_load = GraspFileLoad(self.data_file_info)
        result = self.data_file_load.data_file_process()
        if isinstance(result, tuple) and len(result) == 2:
            self.lsj_lbl_data, self.level_loc_lbl = result
        else:
            raise ValueError(
                "Expected tuple of (lsj_lbl_data, level_loc_lbl) from data_file_process"
            )

        self.min_comp = min_comp
        self.show_comp_num = show_comp_num

    def level_composition_unit_format(self):
        temp_lsj_unit_info_list = self.temp_lsj_unit_information
        temp_lsj_unit_coefficient = np.float64(temp_lsj_unit_info_list[0])
        temp_lsj_unit_w = np.float64(temp_lsj_unit_info_list[1]).round(3)
        temp_lsj_unit_conf = temp_lsj_unit_info_list[2]
        temp_lsj_unit_format = ConfigurationFormatter(
            temp_lsj_unit_conf, self.show_full_charged_subshell, self.format_to_word_document
        )
        temp_lsj_unit_format_conf = temp_lsj_unit_format.conf_format()[0]
        temp_lsj_unit_format_conf_ls = temp_lsj_unit_format.conf_format()[1]

        if temp_lsj_unit_format_conf != "" and temp_lsj_unit_format_conf_ls != "":
            temp_comp_unit_format = f"${str(temp_lsj_unit_w)}\\;{temp_lsj_unit_format_conf}\\,{temp_lsj_unit_format_conf_ls}$ +"
        elif temp_lsj_unit_format_conf != "" and temp_lsj_unit_format_conf_ls == "":
            temp_comp_unit_format = (
                f"${str(temp_lsj_unit_w)}\\;{temp_lsj_unit_format_conf}$ +"
            )
        else:
            raise ValueError(f"{temp_lsj_unit_format_conf=}为空字符")

        return temp_comp_unit_format, temp_lsj_unit_coefficient, temp_lsj_unit_w

    # def level_composition_format(self, self.temp_lsj_information):
    def level_composition_format(self):
        self.temp_level_asf_comp = ""
        component_count = 0

        for self.temp_lsj_unit in self.temp_lsj_information:
            self.temp_lsj_unit_information = self.temp_lsj_unit.split()
            if len(self.temp_lsj_unit_information) == 3:
                # Get weight for filtering
                temp_lsj_unit_w = np.float64(self.temp_lsj_unit_information[1]).round(3)

                # Apply filter logic: if show_comp_num is 0, filter by weight; otherwise, filter by count
                if self.show_comp_num == 0:
                    # Filter by weight: only include components with weight > min_comp
                    if temp_lsj_unit_w <= self.min_comp:
                        continue
                else:
                    # Filter by count: only include first show_comp_num components
                    if component_count >= self.show_comp_num:
                        continue
                    component_count += 1

                self.temp_comp_unit_format = (
                    LevelsASFComposition.level_composition_unit_format(self)[0]
                )
                self.temp_level_asf_comp = (
                    self.temp_level_asf_comp + self.temp_comp_unit_format
                )

            else:
                continue
        self.temp_level_asf_comp = self.temp_level_asf_comp.strip(" +")
        self.temp_level_asf_comp = re.sub(r"\$ \+\$", " + ", self.temp_level_asf_comp)

        return self.temp_level_asf_comp

    def asf_comp_locate(self):
        self.temp_level_asf_comp_loc = self.temp_level_asf.split()
        self.temp_level_asf_dataframe_loc = self.energy_data_df.loc[
            (self.energy_data_df["Pos"] == self.temp_level_asf_comp_loc[0])
            & (self.energy_data_df["J"] == self.temp_level_asf_comp_loc[1])
            & (self.energy_data_df["Parity"] == self.temp_level_asf_comp_loc[2])
        ].index[0]

        return self.temp_level_asf_dataframe_loc

    def level_comp_of_asf(self):
        for self.temp_level_loc in self.level_loc_lbl:
            if self.temp_level_loc != self.level_loc_lbl[-1]:
                self.temp_level_lsj_info = self.lsj_lbl_data[
                    self.temp_level_loc + 1 : self.level_loc_lbl[
                        self.level_loc_lbl.index(self.temp_level_loc) + 1
                    ]
                ]
            else:
                self.temp_level_lsj_info = self.lsj_lbl_data[self.temp_level_loc + 1 :]
            self.temp_level_asf = self.lsj_lbl_data[self.temp_level_loc]
            self.temp_level_df_loc = LevelsASFComposition.asf_comp_locate(self)
            self.temp_lsj_information = self.temp_level_lsj_info
            self.temp_level_asf_comp = LevelsASFComposition.level_composition_format(
                self
            )
            self.energy_data_df.loc[
                self.temp_level_df_loc,
                f"Comp_of_asf_{self.data_file_info['level_parameter']}{self.data_file_info['this_as']}",
            ] = self.temp_level_asf_comp

        return self.energy_data_df


#######################################################################


def asf_radial_wavefunction_collection(data_file_info: dict) -> pd.DataFrame:
    """
    read asf radial wavefunction data,
    no matter the format of the data file is binary file or plot file,
    return the data as a DataFrame.
    """
    data_file_load = GraspFileLoad(data_file_info)
    result = data_file_load.data_file_process()
    if isinstance(result, pd.DataFrame):
        return result
    else:
        raise ValueError("Expected DataFrame from data_file_process")


#######################################################################


class RadialElectrondensityFunction:
    def __init__(self, data_file_info: dict):
        self.data_file_info = data_file_info
        self.data_file_load = GraspFileLoad(self.data_file_info)
        result = self.data_file_load.data_file_process()
        if isinstance(result, list):
            self.radial_electron_density_data = result
        else:
            raise ValueError("Expected list from data_file_process")
        self.radial_electron_density_data_df = pd.DataFrame(
            columns=["r", "D(r)", "rho(r)"]
        )

    def density_data_collection(self):
        block_index = []
        for i in range(len(self.radial_electron_density_data)):
            if re.match(
                r"(\d)+\s+([0-9,/])+\s+([+,-])", self.radial_electron_density_data[i]
            ):
                block_index.append(i)
        block_index.append(len(self.radial_electron_density_data))

        for i in range(len(block_index) - 1):
            block_group = self.radial_electron_density_data[block_index[i]]
            temp_block = self.radial_electron_density_data[
                block_index[i] + 1 : block_index[i + 1]
            ]
            temp_block = [i.replace("D", "e").split() for i in temp_block]
            temp_block_pd = pd.DataFrame(temp_block, columns=["r", "D(r)", "rho(r)"])
            temp_block_pd["Group"] = block_group
            self.radial_electron_density_data_df = pd.concat(
                [self.radial_electron_density_data_df, temp_block_pd], ignore_index=True
            )

        return self.radial_electron_density_data_df
