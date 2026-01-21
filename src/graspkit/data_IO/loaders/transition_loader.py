# -*- encoding: utf-8 -*-
"""
@Id :transition_loader.py
@date :2026/01/19
@author :YenochQin (秦毅)

@version 2.0: Refactored with file scanning mechanism and multiple file support
"""

import re
from pathlib import Path
from typing import Any

import polars as pl

from ...utils.tool_function import doubleJ_to_J
from .base_loader import BaseLoader


class TransitionFileScan:
    """跃迁文件扫描器

    用于扫描目录中的跃迁数据文件
    """

    @staticmethod
    def scan_directory(
        directory: str | Path,
        calculation_type: str = "rci",
        load_ct_lsj: bool = False,
    ) -> list[Path]:
        """扫描目录中的跃迁数据文件

        Args:
            directory: 目录路径
            calculation_type: 计算类型，"rmcdhf" 使用 .t 文件，"rci" 使用 .ct 文件
            load_ct_lsj: 是否加载 .[c]t.lsj 文件

        Returns:
            找到的跃迁文件路径列表，按文件名排序

        Raises:
            ValueError: 目录不存在或计算类型无效
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # 确定文件后缀
        if calculation_type == "rmcdhf":
            base_suffix = ".t"
        elif calculation_type == "rci":
            base_suffix = ".ct"
        else:
            raise ValueError(
                f"Invalid calculation_type: {calculation_type}. "
                "Must be 'rmcdhf' or 'rci'"
            )

        # 确定要查找的后缀
        if load_ct_lsj:
            suffix = base_suffix + ".lsj"
        else:
            suffix = base_suffix

        # 扫描文件
        files = sorted(dir_path.glob(f"*{suffix}"))

        if not files:
            # 如果没找到 .lsj 文件，尝试查找基础文件
            if load_ct_lsj:
                files = sorted(dir_path.glob(f"*{base_suffix}"))
                if not files:
                    raise ValueError(
                        f"No transition files found with suffix {suffix} or {base_suffix} "
                        f"in directory: {directory}"
                    )
            else:
                raise ValueError(
                    f"No transition files found with suffix {suffix} "
                    f"in directory: {directory}"
                )

        return files

    @staticmethod
    def get_file_type(file_path: Path) -> str:
        """获取跃迁文件类型

        Args:
            file_path: 文件路径

        Returns:
            文件类型: "t", "ct", "t.lsj", 或 "ct.lsj"
        """
        filename = file_path.name

        if filename.endswith(".t.lsj"):
            return "t.lsj"
        elif filename.endswith(".ct.lsj"):
            return "ct.lsj"
        elif filename.endswith(".t"):
            return "t"
        elif filename.endswith(".ct"):
            return "ct"
        else:
            return "unknown"


#######################################################################
# Transition data parsing utilities
#######################################################################


def _convert_fortran_float(value: str) -> float:
    """转换 Fortran 双精度浮点数格式为 Python float

    Args:
        value: Fortran 格式的浮点数字符串，如 "9.35167D-13", "2.50540D-05"

    Returns:
        转换后的浮点数

    Examples:
        >>> _convert_fortran_float("9.35167D-13")
        9.35167e-13
        >>> _convert_fortran_float("6.12752D+01")
        612.752
    """
    if not value or value.strip() == "":
        return 0.0
    return float(value.replace("D", "E"))


def _parse_transition_first_line(line: str) -> dict[str, Any]:
    """解析跃迁数据的第一行（磁性跃迁和电性跃迁通用）

    格式: f1/f2 上能级Pos 上能级J 上能级宇称  f1/f2 下能级Pos 下能级J 下能级宇称  能量 规范 A_C gf_C S_C

    例如：
    f1  1    2 +  f2  1    2 +        2368.69 C  0.00000D+00  0.00000D+00  0.00000D+00

    Args:
        line: 第一行数据

    Returns:
        包含跃迁基础数据的字典（不包含B规范数据）
    """
    parts = line.split()

    # 解析上能级信息 (前4个字段: f1/f2, Pos, J, 宇称)
    upper_file_str = parts[0] if len(parts) > 0 else ""
    upper_file = int(upper_file_str[1]) if upper_file_str in ("f1", "f2") else 1
    upper_pos = parts[1] if len(parts) > 1 else ""
    upper_j = parts[2] if len(parts) > 2 else ""
    upper_parity = parts[3] if len(parts) > 3 else ""

    # 解析下能级信息 (接下来4个字段: f1/f2, Pos, J, 宇称)
    lower_file_str = parts[4] if len(parts) > 4 else ""
    lower_file = int(lower_file_str[1]) if lower_file_str in ("f1", "f2") else 2
    lower_pos = parts[5] if len(parts) > 5 else ""
    lower_j = parts[6] if len(parts) > 6 else ""
    lower_parity = parts[7] if len(parts) > 7 else ""

    # 解析跃迁数据 (能量, 规范, A_C, gf_C, S_C)
    energy = float(parts[8]) if len(parts) > 8 else 0.0
    transition_rate = _convert_fortran_float(parts[10]) if len(parts) > 10 else 0.0
    gf = _convert_fortran_float(parts[11]) if len(parts) > 11 else 0.0
    line_strength = _convert_fortran_float(parts[12]) if len(parts) > 12 else 0.0

    return {
        "upper_file": upper_file,
        "upper_pos": upper_pos,
        "upper_j": upper_j,
        "upper_parity": upper_parity,
        "lower_file": lower_file,
        "lower_pos": lower_pos,
        "lower_j": lower_j,
        "lower_parity": lower_parity,
        "energy": energy,
        "transition_rate_C": transition_rate,
        "gf_C": gf,
        "line_strength_C": line_strength,
    }


def _parse_transition_second_line(line: str) -> dict[str, float]:
    """解析电性跃迁数据的第二行（B规范数据）

    格式: [空格] B A_B gf_B S_B

    例如：
            B  0.00000D+00  0.00000D+00  0.00000D+00

    Args:
        line: 第二行数据（B规范）

    Returns:
        包含B规范数据的字典 {"transition_rate_B": ..., "gf_B": ..., "line_strength_B": ...}

    Raises:
        ValueError: 如果第二行数据格式不正确或不包含B规范数据
    """
    parts = line.split()

    if len(parts) < 4:
        raise ValueError(
            f"Invalid second line format for electric transition: expected at least 4 fields, got {len(parts)}"
        )

    if parts[0] not in ("C", "B", "M"):
        raise ValueError(
            f"Invalid gauge identifier in second line: expected 'C', 'B', or 'M', got '{parts[0]}'"
        )

    return {
        "transition_rate_B": _convert_fortran_float(parts[1]),
        "gf_B": _convert_fortran_float(parts[2]),
        "line_strength_B": _convert_fortran_float(parts[3]),
    }


def parse_electric_transition_line(line1: str, line2: str) -> dict[str, Any]:
    """解析电性跃迁数据行

    电性跃迁有两行数据：
    - 第一行：f1/f2 上能级Pos 上能级J 上能级宇称  f1/f2 下能级Pos 下能级J 下能级宇称  能量 规范 A_C gf_C S_C
    - 第二行：B规范数据

    例如：
    f2  1    2 +  f1  1    1 -        6489.57 C  6.06225D+00  1.07902D-06  5.47381D-05
                                              B  9.08381D-04  1.61683D-10  8.20207D-09

    Args:
        line1: 第一行数据（必需）
        line2: 第二行数据（必需）

    Returns:
        包含跃迁数据的字典

    Raises:
        ValueError: 如果第二行数据为空或格式不正确
    """
    # 解析第一行
    result = _parse_transition_first_line(line1)

    # 解析第二行（B规范数据）
    if not line2 or not line2.strip():
        raise ValueError(
            "Second line is required for electric transition but got empty line"
        )

    b_data = _parse_transition_second_line(line2)
    result.update(b_data)

    return result


def parse_magnetic_transition_line(line: str) -> dict[str, Any]:
    """解析磁性跃迁数据行

    磁性跃迁只有一行数据，格式与电性跃迁的第一行相同，但只有M规范（对应C规范）

    例如：
    f1  1    2 +  f2  1    2 +        2368.69 M  0.00000D+00  0.00000D+00  0.00000D+00

    Args:
        line: 跃迁数据行

    Returns:
        包含跃迁数据的字典（B规范字段为None）
    """
    result = _parse_transition_first_line(line)
    # 磁性跃迁只有C规范（M规范），没有B规范
    result["transition_rate_B"] = None
    result["gf_B"] = None
    result["line_strength_B"] = None
    return result


#######################################################################
# LSJ format parsing utilities
#######################################################################


def _parse_level_line(line: str) -> dict[str, Any]:
    """解析 .lsj 格式的能级行

    格式: "  2-11257.5967049  5s(2).4d(10).5p(6).6s(2).4f(7)8S.5d_7D"

    Args:
        line: .lsj 格式的能级行

    Returns:
        {"double_J": 双倍J值, "energy": 能量, "configuration": 配置字符串}
    """
    parts = line.split()
    # 第一部分: "double_J-能量"
    double_j_energy = parts[0]
    if "-" in double_j_energy:
        double_j_str, energy_str = double_j_energy.split("-", 1)
    else:
        double_j_str = parts[0]
        energy_str = parts[1] if len(parts) > 1 else "0"

    double_j = double_j_str.strip()
    energy = float(energy_str) if energy_str else 0.0

    # 配置信息（剩余部分）
    config = " ".join(parts[1:]) if len(parts) > 1 else ""

    return {"double_J": double_j, "energy": energy, "configuration": config}


def _parse_lsj_magnetic_transition(
    upper_line: str, lower_line: str, energy_line: str, data_line: str
) -> dict[str, Any]:
    """解析 .lsj 格式的磁性跃迁数据（4行）

    Args:
        upper_line: 上能级行
        lower_line: 下能级行
        energy_line: 能量和波长行
        data_line: 跃迁数据行

    Returns:
        包含跃迁数据的字典
    """
    upper = _parse_level_line(upper_line)
    lower = _parse_level_line(lower_line)

    # 解析能量行: "  12462.51 CM-1      8024.06 ANGS(VAC)      8023.23 ANGS(AIR)"
    energy_parts = energy_line.split()
    energy_diff = float(energy_parts[0]) if energy_parts else 0.0

    # 解析数据行: " M2  S =  1.65556D-01   GF =  7.16288D-16   AKI =  1.48413D-08"
    data_parts = data_line.split()

    # 提取跃迁类型 (M1, M2, M3...)
    transition_type = data_parts[0] if data_parts else ""

    # 提取 S (线强度)
    s_idx = data_parts.index("S") + 2 if "S" in data_parts else -1
    line_strength = (
        _convert_fortran_float(data_parts[s_idx])
        if s_idx >= 0 and s_idx < len(data_parts)
        else 0.0
    )

    # 提取 GF (振子强度)
    gf_idx = data_parts.index("GF") + 2 if "GF" in data_parts else -1
    gf = (
        _convert_fortran_float(data_parts[gf_idx])
        if gf_idx >= 0 and gf_idx < len(data_parts)
        else 0.0
    )

    # 提取 AKI (跃迁几率)
    aki_idx = data_parts.index("AKI") + 2 if "AKI" in data_parts else -1
    transition_rate = (
        _convert_fortran_float(data_parts[aki_idx])
        if aki_idx >= 0 and aki_idx < len(data_parts)
        else 0.0
    )

    # 转换 double_J 为 J 字符串
    upper_j = (
        doubleJ_to_J(int(upper["double_J"]))
        if upper["double_J"].isdigit()
        else upper["double_J"]
    )
    lower_j = (
        doubleJ_to_J(int(lower["double_J"]))
        if lower["double_J"].isdigit()
        else lower["double_J"]
    )

    return {
        "upper_j": upper_j,
        "upper_energy": upper["energy"],
        "upper_configuration": upper["configuration"],
        "lower_j": lower_j,
        "lower_energy": lower["energy"],
        "lower_configuration": lower["configuration"],
        "energy": energy_diff,
        "transition_rate_C": transition_rate,
        "gf_C": gf,
        "line_strength_C": line_strength,
        "transition_rate_B": None,  # 磁性跃迁没有B规范
        "gf_B": None,
        "line_strength_B": None,
        "transition_type": transition_type,
    }


def _parse_lsj_electric_transition(
    upper_line: str,
    lower_line: str,
    energy_line: str,
    data_line: str,
    extra_line: str = "",
) -> dict[str, Any]:
    """解析 .lsj 格式的电性跃迁数据（4-5行）

    Args:
        upper_line: 上能级行
        lower_line: 下能级行
        energy_line: 能量和波长行
        data_line: 跃迁数据行（C规范）
        extra_line: 额外数据行（B规范，可选）

    Returns:
        包含跃迁数据的字典
    """
    upper = _parse_level_line(upper_line)
    lower = _parse_level_line(lower_line)

    # 解析能量行
    energy_parts = energy_line.split()
    energy_diff = float(energy_parts[0]) if energy_parts else 0.0

    # 解析数据行（C规范）: " E1  S =  8.20207D-09   GF =  1.61683D-10   AKI =  9.08381D-04   dT =  0.99985"
    data_parts = data_line.split()

    # 提取跃迁类型 (E1, E2, E3...)
    transition_type = data_parts[0] if data_parts else ""

    # 提取 S (线强度)
    s_idx = data_parts.index("S") + 2 if "S" in data_parts else -1
    line_strength_c = (
        _convert_fortran_float(data_parts[s_idx])
        if s_idx >= 0 and s_idx < len(data_parts)
        else 0.0
    )

    # 提取 GF (振子强度)
    gf_idx = data_parts.index("GF") + 2 if "GF" in data_parts else -1
    gf_c = (
        _convert_fortran_float(data_parts[gf_idx])
        if gf_idx >= 0 and gf_idx < len(data_parts)
        else 0.0
    )

    # 提取 AKI (跃迁几率)
    aki_idx = data_parts.index("AKI") + 2 if "AKI" in data_parts else -1
    transition_rate_c = (
        _convert_fortran_float(data_parts[aki_idx])
        if aki_idx >= 0 and aki_idx < len(data_parts)
        else 0.0
    )

    # 解析B规范数据（如果有）
    line_strength_b = 0.0
    gf_b = 0.0
    transition_rate_b = 0.0

    if extra_line and extra_line.strip():
        extra_parts = extra_line.split()
        if len(extra_parts) >= 3:
            # 格式: "          5.47381D-05         1.07902D-06          6.06225D+00"
            # 三个值分别是: S, GF, AKI
            line_strength_b = (
                _convert_fortran_float(extra_parts[0]) if extra_parts[0] else 0.0
            )
            gf_b = (
                _convert_fortran_float(extra_parts[1]) if len(extra_parts) > 1 else 0.0
            )
            transition_rate_b = (
                _convert_fortran_float(extra_parts[2]) if len(extra_parts) > 2 else 0.0
            )

    # 转换 double_J 为 J 字符串
    upper_j = (
        doubleJ_to_J(int(upper["double_J"]))
        if upper["double_J"].isdigit()
        else upper["double_J"]
    )
    lower_j = (
        doubleJ_to_J(int(lower["double_J"]))
        if lower["double_J"].isdigit()
        else lower["double_J"]
    )

    return {
        "upper_j": upper_j,
        "upper_energy": upper["energy"],
        "upper_configuration": upper["configuration"],
        "lower_j": lower_j,
        "lower_energy": lower["energy"],
        "lower_configuration": lower["configuration"],
        "energy": energy_diff,
        "transition_rate_C": transition_rate_c,
        "gf_C": gf_c,
        "line_strength_C": line_strength_c,
        "transition_rate_B": transition_rate_b,
        "gf_B": gf_b,
        "line_strength_B": line_strength_b,
        "transition_type": transition_type,
    }


#######################################################################
# Transition Loader
#######################################################################


class TransitionLoader(BaseLoader):
    """GRASP2018 跃迁数据加载器

    用于加载 GRASP2018 生成的跃迁数据文件。
    支持以下文件类型：
    - .t 文件 (rmcdhf 计算结果)
    - .ct 文件 (rci 计算结果)
    - .t.lsj 文件 (rmcdhf LSJ 耦合格式)
    - .ct.lsj 文件 (rci LSJ 耦合格式)

    支持两种加载模式：
    1. 单文件模式：直接加载指定的跃迁文件
    2. 目录扫描模式：扫描目录中的所有跃迁文件并合并

    返回值：
        list[str] - 文件行列表（单文件或合并后的多文件）
    """

    def __init__(
        self,
        file_path: str | Path,
        calculation_type: str = "rci",
        load_ct_lsj: bool = False,
    ):
        """初始化跃迁文件加载器

        Args:
            file_path: 文件或目录路径
            calculation_type: 计算类型（目录扫描时使用）
                "rmcdhf": 使用 .t 文件
                "rci": 使用 .ct 文件
            load_ct_lsj: 是否加载 .lsj 格式文件
                True: 优先加载 .t.lsj 或 .ct.lsj 文件
                False: 加载 .t 或 .ct 文件

        Raises:
            ValueError: 路径不存在或参数无效
        """
        path_obj = Path(file_path)

        if not path_obj.exists():
            raise ValueError(f"Path does not exist: {file_path}")

        # 始终保存用户输入的原始路径
        self.file_path = path_obj

        # 判断是文件还是目录
        self._is_directory = path_obj.is_dir()

        if self._is_directory:
            # 目录模式：扫描文件
            self.file_paths = TransitionFileScan.scan_directory(
                file_path, calculation_type, load_ct_lsj
            )
        else:
            # 单文件模式
            self.file_paths = [path_obj]

        self.calculation_type = calculation_type
        self.load_ct_lsj = load_ct_lsj

    def load(self) -> list[str]:
        """加载跃迁数据

        Returns:
            文件行列表（单文件或合并后的多文件）

        Raises:
            ValueError: 文件格式不正确
            IOError: 读取错误
        """
        if self._is_directory:
            # 目录模式：合并所有文件
            return self._load_multiple_files()
        else:
            # 单文件模式
            return self._load_single_file(self.file_path)

    def _load_single_file(self, file_path: Path) -> list[str]:
        """加载单个跃迁文件

        Args:
            file_path: 文件路径

        Returns:
            文件行列表（原始文本数据）
        """
        return self._load_transition_file(file_path)

    def _load_multiple_files(self) -> list[str]:
        """加载多个跃迁文件并合并

        Returns:
            合并后的文件行列表（文件间用两个空字符串分隔）

        Raises:
            ValueError: 没有有效文件可加载
        """
        all_lines = []

        for i, file_path in enumerate(self.file_paths):
            try:
                lines = self._load_single_file(file_path)
                # 移除末尾的空字符串（避免多次添加）
                if lines and lines[-1] == "":
                    lines = lines[:-1]

                # 添加当前文件的数据
                all_lines.extend(lines)

                # 如果不是最后一个文件，添加两个空字符串作为分隔符
                if i < len(self.file_paths) - 1:
                    all_lines.extend(["", ""])

            except Exception as e:
                # 记录错误但继续处理其他文件
                print(f"Warning: Failed to load {file_path}: {e}")
                continue

        if not all_lines:
            raise ValueError("No valid transition data files could be loaded")

        # 添加最终的空字符串作为结束标记
        all_lines.append("")

        return all_lines

    def _load_transition_file(self, file_path: Path) -> list[str]:
        """加载跃迁文件（支持所有文本格式：.t, .ct, .t.lsj, .ct.lsj）

        Args:
            file_path: 文件路径

        Returns:
            文件行列表（原始文本数据）
        """
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]

        # 添加空字符串作为结束标记（与旧代码兼容）
        lines.append("")

        return lines

    def _classify_lsj_transitions(
        self, lines: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """归类 .lsj 格式的跃迁数据

        将原始行列表按跃迁种类（E1, M1, E2, M2...）分类并解析

        Args:
            lines: 文件行列表

        Returns:
            字典 {跃迁种类标记: 解析后的跃迁数据列表}
            例如: {"E1": [{...}, {...}], "M1": [{...}, {...}]}
        """
        classified = {}

        # 找到所有 "Transition between files:" 的位置
        transition_indices = []
        for idx, line in enumerate(lines):
            if "Transition between files:" in line:
                transition_indices.append(idx)

        # 处理每个跃迁块
        for idx in transition_indices:
            # 检查是否有数据（+3行是否是下一个 "Transition between files:"）
            if idx + 3 < len(lines) and "Transition between files:" in lines[idx + 3]:
                continue  # 没有数据，跳过

            # 数据从 idx + 5 开始（跳过 "Transition between files:", 文件名1, 文件名2, 两个空行）
            data_start_idx = idx + 5
            if data_start_idx >= len(lines):
                continue

            i = data_start_idx
            while i < len(lines):
                # 跳过空行
                while i < len(lines) and (not lines[i] or lines[i].strip() == ""):
                    i += 1
                    # 检查是否到达下一个跃迁块
                    if i < len(lines) and "Transition between files:" in lines[i]:
                        break

                # 检查是否到达文件末尾或下一个跃迁块
                if i >= len(lines) or (
                    i < len(lines) and "Transition between files:" in lines[i]
                ):
                    break

                # 每个跃迁数据块至少需要4行：上能级、下能级、能量行、跃迁数据行
                if i + 3 >= len(lines):
                    break

                # 获取跃迁数据行（第4行）
                data_line = lines[i + 3] if i + 3 < len(lines) else ""

                # 判断跃迁类型（从跃迁数据行判断）
                stripped_data = data_line.strip()
                if not stripped_data:
                    # 空行，移动到下一行
                    i += 1
                    continue

                # 判断是电性(E)还是磁性(M)
                if stripped_data[0] == "E":
                    # 电性跃迁：5行（上能级、下能级、能量、C规范数据、B规范数据）
                    if i + 4 < len(lines):
                        upper_line = lines[i]
                        lower_line = lines[i + 1]
                        energy_line = lines[i + 2]
                        data_line = lines[i + 3]
                        extra_line = lines[i + 4] if i + 4 < len(lines) else ""

                        try:
                            parsed = _parse_lsj_electric_transition(
                                upper_line,
                                lower_line,
                                energy_line,
                                data_line,
                                extra_line,
                            )
                            transition_type = parsed["transition_type"]
                            if transition_type not in classified:
                                classified[transition_type] = []
                            classified[transition_type].append(parsed)
                        except (ValueError, IndexError):
                            pass
                        i += 5
                    else:
                        break

                elif stripped_data[0] == "M":
                    # 磁性跃迁：4行（上能级、下能级、能量、M规范数据）
                    upper_line = lines[i]
                    lower_line = lines[i + 1]
                    energy_line = lines[i + 2]
                    data_line = lines[i + 3]

                    try:
                        parsed = _parse_lsj_magnetic_transition(
                            upper_line, lower_line, energy_line, data_line
                        )
                        transition_type = parsed["transition_type"]
                        if transition_type not in classified:
                            classified[transition_type] = []
                        classified[transition_type].append(parsed)
                    except (ValueError, IndexError):
                        pass
                    i += 4
                else:
                    # 未知的跃迁类型，跳过
                    i += 1

        return classified

    def _classify_ct_transitions(self, lines: list[str]) -> dict[str, list[str]]:
        """归类跃迁数据

        将原始行列表按跃迁种类（E1, M1, E2, M2...）分类

        Args:
            lines: 文件行列表

        Returns:
            字典 {跃迁种类标记: 数据行列表}
            例如: {"E1": [...], "M1": [...], "E2": [...]}
        """
        # 1. 定位所有包含 "-pole transitions" 的行索引
        transition_indices = []
        for idx, line in enumerate(lines):
            if "-pole transitions" in line:
                transition_indices.append(idx)

        # 2. 处理每个跃迁种类
        classified = {}

        for idx in transition_indices:
            # 获取跃迁类型行
            type_line = lines[idx]

            # 判断是电性还是磁性，提取 pole 数值
            if "Electric" in type_line:
                em_type = "E"
            elif "Magnetic" in type_line:
                em_type = "M"
            else:
                continue  # 未知类型，跳过

            # 提取 pole 数值: "2**( 1)-pole" -> "1"
            # 匹配格式: "2**( 1)-pole" 或 "2**(1)-pole"
            pole_match = re.search(r"2\*\*\(\s*(\d+)\s*\)-pole", type_line)
            if not pole_match:
                continue
            pole = pole_match.group(1)

            # 形成跃迁种类标记
            transition_type = f"{em_type}{pole}"

            # 数据从索引+5开始（跳过4行header + 1行空行）
            data_start_idx = idx + 5

            # 检查是否有数据
            if data_start_idx >= len(lines):
                continue

            first_data_line = lines[data_start_idx]
            if first_data_line == "" or first_data_line.isspace():
                continue  # 没有数据，跳过

            # 收集数据行
            data_lines = []
            i = data_start_idx
            empty_count = 0  # 连续空字符串计数
            while i < len(lines):
                line = lines[i]

                # 检查是否为空字符串
                if line == "" or line.isspace():
                    empty_count += 1
                    # 遇到两个连续空字符串，表示该跃迁种类数据结束
                    if empty_count >= 2:
                        break
                    i += 1
                    continue

                # 重置计数器，添加非空行
                empty_count = 0
                data_lines.append(line)
                i += 1

            # 归类数据
            if transition_type not in classified:
                classified[transition_type] = []
            classified[transition_type].extend(data_lines)

        return classified

    # ========== 便捷方法 ==========

    def get_file_type(self) -> str:
        """获取当前加载的文件类型

        Returns:
            文件类型: "t", "ct", "t.lsj", 或 "ct.lsj"
        """
        return TransitionFileScan.get_file_type(Path(self.file_path))

    def is_lsj_format(self) -> bool:
        """检查是否为 LSJ 耦合格式

        Returns:
            True 如果是 LSJ 耦合格式
        """
        return self.get_file_type().endswith(".lsj")

    def is_directory_mode(self) -> bool:
        """检查是否为目录扫描模式

        Returns:
            True 如果是目录扫描模式
        """
        return self._is_directory

    def get_loaded_files(self) -> list[str]:
        """获取已加载的文件列表

        Returns:
            文件路径列表
        """
        return [str(f) for f in self.file_paths]

    def to_dataframe(self) -> pl.DataFrame:
        """将跃迁数据转换为 polars DataFrame

        直接从归类后的原始数据构建 DataFrame，
        无需中间解析步骤。

        Returns:
            包含所有跃迁数据的 DataFrame，包含 transition_type 列
        """
        all_data = []

        # 检查是否是 .lsj 格式
        if self.load_ct_lsj or (
            self.file_paths
            and TransitionFileScan.get_file_type(self.file_paths[0]).endswith(".lsj")
        ):
            # .lsj 格式：数据已经解析好了
            lines = self.load()
            classified = self._classify_lsj_transitions(lines)

            for transition_type, transitions in classified.items():
                all_data.extend(transitions)

            # 构建 DataFrame（.lsj 格式包含额外的配置信息）
            return pl.DataFrame(all_data)

        else:
            # .ct/.t 格式：需要解析原始行
            lines = self.load()
            classified = self._classify_ct_transitions(lines)

            for transition_type, data_lines in classified.items():
                # 判断是电性(E)还是磁性(M)
                if transition_type.startswith("E"):
                    # 电性跃迁：每两行为一组
                    for i in range(0, len(data_lines), 2):
                        if i + 1 < len(data_lines):
                            line1 = data_lines[i]
                            line2 = data_lines[i + 1]
                            # 跳过空行和标题行（数据行以 f1 或 f2 开头）
                            if line1 and line1.strip():
                                stripped = line1.strip()
                                # 检查是否为数据行（以 f1 或 f2 开头）
                                if stripped.startswith("f1 ") or stripped.startswith(
                                    "f2 "
                                ):
                                    try:
                                        parsed_data = parse_electric_transition_line(
                                            line1, line2
                                        )
                                        parsed_data["transition_type"] = transition_type
                                        all_data.append(parsed_data)
                                    except (ValueError, IndexError):
                                        continue

                elif transition_type.startswith("M"):
                    # 磁性跃迁：每行为一组
                    for line in data_lines:
                        # 跳过空行和标题行（数据行以 f1 或 f2 开头）
                        if line and line.strip():
                            stripped = line.strip()
                            # 检查是否为数据行（以 f1 或 f2 开头）
                            if stripped.startswith("f1 ") or stripped.startswith("f2 "):
                                try:
                                    parsed_data = parse_magnetic_transition_line(line)
                                    parsed_data["transition_type"] = transition_type
                                    all_data.append(parsed_data)
                                except (ValueError, IndexError):
                                    continue

            # 构建 DataFrame
            columns = [
                "upper_file",
                "upper_pos",
                "upper_j",
                "upper_parity",
                "lower_file",
                "lower_pos",
                "lower_j",
                "lower_parity",
                "energy",
                # "gauge",
                "transition_rate_C",
                "gf_C",
                "line_strength_C",
                "transition_rate_B",
                "gf_B",
                "line_strength_B",
                "transition_type",
            ]

            return pl.DataFrame(all_data, schema=columns)
