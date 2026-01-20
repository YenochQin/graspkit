# -*- encoding: utf-8 -*-
"""
@Id :transition_loader.py
@date :2026/01/19
@author :YenochQin (秦毅)
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .base_loader import BaseLoader


class TransitionLoader(BaseLoader[list[str]]):
    """GRASP2018 跃迁数据加载器

    用于加载 GRASP2018 生成的跃迁数据文件（.t 文本格式）。
    支持标准格式和 LSJ 耦合格式。

    默认返回原始行列表（与旧的 GraspFileLoad.data_file_process() 兼容），
    也提供解析为结构化数据的方法。
    """

    def load(self) -> list[str]:
        """加载跃迁数据（原始行列表）

        Returns:
            跃迁数据的行列表，每个元素是一行文本
            末尾会添加一个空字符串（与旧代码兼容）

        Raises:
            ValueError: 文件格式不正确
            IOError: 读取错误
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]

        # 添加空字符串作为结束标记（与旧代码兼容）
        lines.append("")

        return lines

    def is_lsj_format(self) -> bool:
        """检查是否为 LSJ 耦合格式

        Returns:
            True 如果是 LSJ 耦合格式（文件名以 .lsj 结尾），False 如果是标准格式
        """
        filename = self.get_file_name()
        return filename.endswith(".lsj")

    def get_transition_type(self) -> str:
        """获取跃迁类型

        解析文件中的跃迁类型行，如 "Electric 2**(1)-pole transitions"

        Returns:
            跃迁类型字符串，如 "E1", "M1", "E2" 等

        Raises:
            ValueError: 无法找到跃迁类型行
        """
        lines = self.load()

        for line in lines:
            match = re.match(r"([A-Za-z]*) 2\*\*\( ([0-9])\)-pole transitions", line)
            if match:
                # 返回第一个字母（E/M）+ 极数（1/2/...）
                return match.group(1)[0] + match.group(2)

        raise ValueError("Transition type line not found in file")

    def find_transition_blocks(self) -> tuple[list[int], list[int]]:
        """查找跃迁数据块的位置

        跃迁文件可能包含多个跃迁类型块（如 E1, M1, E2 等）。
        此方法找到每个块的起始和结束行索引。

        Returns:
            (trans_data_line_index, transition_type_line_index) 元组
            - trans_data_line_index: 数据块的起始和结束行索引列表
            - transition_type_line_index: 跃迁类型行的行索引列表

        Raises:
            ValueError: 文件格式不正确
        """
        lines = self.load()

        # 找到所有跃迁类型行
        transition_type_line_index = [
            i
            for i, line in enumerate(lines)
            if re.match(r"([A-Za-z]*) 2\*\*\( ([0-9])\)-pole transitions", line)
        ]

        if not transition_type_line_index:
            raise ValueError("No transition type lines found in file")

        # 找到包含实际数据的跃迁类型行（通过检查是否有 M/C/B 列）
        transition_data_type_line_index = [
            line
            for line in transition_type_line_index
            if line + 5 < len(lines) and re.match(r"f.*f.*[M,C,B]", lines[line + 5])
        ]

        trans_data_line_index = []

        # 确定每个数据块的起始和结束位置
        for i, type_line in enumerate(transition_data_type_line_index):
            # 数据块从类型行后第5行开始
            start_idx = type_line + 5

            # 确定结束位置
            if i == len(transition_data_type_line_index) - 1:
                # 最后一个块，直到文件末尾
                end_idx = len(lines) - 2
            else:
                # 下一个类型行前2行结束
                next_type_line = transition_type_line_index[
                    transition_type_line_index.index(type_line) + 1
                ]
                end_idx = next_type_line - 2

            trans_data_line_index.extend([start_idx, end_idx])

        return trans_data_line_index, transition_type_line_index

    def get_transition_type_from_line(self, line: str) -> str:
        """从跃迁类型行提取类型

        Args:
            line: 跃迁类型行，如 "Electric 2**(1)-pole transitions"

        Returns:
            跃迁类型字符串，如 "E1", "M1", "E2" 等
        """
        match = re.match(r"([A-Za-z]*) 2\*\*\( ([0-9])\)-pole transitions", line)
        if match:
            return match.group(1)[0] + match.group(2)
        return ""

    def to_dataframe(self) -> pd.DataFrame:
        """将跃迁数据转换为 DataFrame

        Returns:
            包含跃迁数据的 DataFrame

        Raises:
            ValueError: 数据格式不正确
        """
        lines = self.load()
        trans_data_line_index, _ = self.find_transition_blocks()

        columns = [
            "Upper_file",
            "Upper_loc",
            "Upper_J",
            "Upper_parity",
            "Lower_file",
            "Lower_loc",
            "Lower_J",
            "transition_type",
            "Lower_parity",
            "energy_level_difference",
            "wavelength_vac",
            "transition_rate_C",
            "oscillator_strength_C",
            "line_strength_C",
            "transition_rate_B",
            "oscillator_strength_B",
            "line_strength_B",
            "transition_rate_M",
            "oscillator_strength_M",
            "line_strength_M",
            "transition_dT",
        ]

        all_data = []

        # 处理每个数据块
        for i in range(0, len(trans_data_line_index), 2):
            start_idx = trans_data_line_index[i]
            end_idx = trans_data_line_index[i + 1]

            # 找到跃迁类型行（在数据块前5行）
            type_line_idx = start_idx - 5
            transition_type = self.get_transition_type_from_line(lines[type_line_idx])

            # 处理数据行
            for line_idx in range(start_idx, end_idx + 1):
                line = lines[line_idx]
                if not line.strip():
                    continue

                parts = line.split()
                if len(parts) < 9:
                    continue

                try:
                    # 解析数据行
                    upper_file = int(parts[0])
                    upper_loc = int(parts[1])
                    upper_j = parts[2]
                    upper_parity = parts[3]

                    lower_file = int(parts[4])
                    lower_loc = int(parts[5])
                    lower_j = parts[6]
                    lower_parity = parts[7]

                    energy_diff = float(parts[8])
                    wavelength = float(parts[9])

                    # C方法数据 (列 10-12)
                    rate_c = float(parts[10]) if len(parts) > 10 else 0.0
                    strength_c = float(parts[11]) if len(parts) > 11 else 0.0
                    line_s_c = float(parts[12]) if len(parts) > 12 else 0.0

                    # B方法数据 (列 13-15)
                    rate_b = float(parts[13]) if len(parts) > 13 else 0.0
                    strength_b = float(parts[14]) if len(parts) > 14 else 0.0
                    line_s_b = float(parts[15]) if len(parts) > 15 else 0.0

                    # M方法数据 (列 16-18)
                    rate_m = float(parts[16]) if len(parts) > 16 else 0.0
                    strength_m = float(parts[17]) if len(parts) > 17 else 0.0
                    line_s_m = float(parts[18]) if len(parts) > 18 else 0.0

                    # 计算跃迁几率差异
                    rate_max = max(rate_b, rate_c)
                    d_t = abs(rate_b - rate_c) / rate_max if rate_max > 0 else 0.0

                    row_data = [
                        upper_file,
                        upper_loc,
                        upper_j,
                        upper_parity,
                        lower_file,
                        lower_loc,
                        lower_j,
                        transition_type,
                        lower_parity,
                        energy_diff,
                        wavelength,
                        rate_c,
                        strength_c,
                        line_s_c,
                        rate_b,
                        strength_b,
                        line_s_b,
                        rate_m,
                        strength_m,
                        line_s_m,
                        d_t,
                    ]

                    all_data.append(row_data)

                except (ValueError, IndexError):
                    # 跳过格式错误的行
                    continue

        return pd.DataFrame(all_data, columns=columns)
