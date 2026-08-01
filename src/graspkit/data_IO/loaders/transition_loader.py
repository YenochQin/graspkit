# -*- encoding: utf-8 -*-
import re
from pathlib import Path
from typing import override

import polars as pl

from ...utils.tool_function import doubleJ_to_J
from .base_loader import BaseLoader

# 正则表达式：匹配 S/GF/AKI = 数值格式的数据
DATA_PATTERN = re.compile(r"(S|GF|AKI)\s*=\s*([\d\.D\+\-]+)")


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
                f"Invalid calculation_type: {calculation_type}. Must be 'rmcdhf' or 'rci'"
            )

        # 确定要查找的后缀
        if load_ct_lsj:
            suffix = base_suffix + ".lsj"
        else:
            suffix = base_suffix

        # 扫描文件
        files = sorted(dir_path.glob(f"*{suffix}"))

        if not files:
            # 如果没找到文件，尝试查找备选格式
            if load_ct_lsj:
                # 没找到 .lsj 文件，尝试查找基础文件
                files = sorted(dir_path.glob(f"*{base_suffix}"))
                if not files:
                    raise ValueError(
                        f"No transition files found with suffix {suffix} \nor {base_suffix} in directory: {directory}"
                    )
            else:
                # 没找到基础文件，尝试查找 .lsj 文件
                files = sorted(dir_path.glob(f"*{base_suffix}.lsj"))
                if not files:
                    raise ValueError(
                        f"No transition files found with suffix {suffix} \nor {base_suffix}.lsj in directory: {directory}"
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


def _parse_transition_first_line(line: str) -> dict[str, str]:
    """解析跃迁数据的第一行（磁性跃迁和电性跃迁通用）

    格式: f1/f2 上能级Pos 上能级J 上能级宇称  f1/f2 下能级Pos 下能级J 下能级宇称  能量 规范 A_C gf_C S_C

    例如：
    f1  1    2 +  f2  1    2 +        2368.69 C  0.00000D+00  0.00000D+00  0.00000D+00

    Args:
        line: 第一行数据

    Returns:
        包含跃迁基础数据的字典（不包含B规范数据）
    """
    parts: list[str] = line.split()
    # 填充到固定长度 13
    p = parts + [""] * (13 - len(parts))

    # 解构赋值（index 9 是规范标识，用 _ 忽略）
    (
        upper_file, upper_pos, upper_j, upper_parity,
        lower_file, lower_pos, lower_j, lower_parity,
        energy, _, transition_rate, gf, line_strength
    ) = p[:13]

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


def _parse_transition_second_line(line: str) -> dict[str, str]:
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
    parts: list[str] = line.split()

    if len(parts) < 4:
        raise ValueError(
            f"Invalid second line format for electric transition: expected at least 4 fields, got {len(parts)}"
        )

    if parts[0] not in ("C", "B", "M"):
        raise ValueError(
            f"Invalid gauge identifier in second line: expected 'C', 'B', or 'M', got '{parts[0]}'"
        )

    return {
        "transition_rate_B": parts[1],
        "gf_B": parts[2],
        "line_strength_B": parts[3],
    }


def parse_electric_transition_line(line1: str, line2: str) -> dict[str, str]:
    """解析电性跃迁数据行

    电性跃迁有两行数据：

    - 第一行：f1/f2 上能级Pos 上能级J 上能级宇称  f1/f2 下能级Pos 下能级J 下能级宇称  能量 规范 A_C gf_C S_C
    - 第二行：B规范数据

    Example:
        第一行包含 C 规范数据，第二行包含 B 规范数据。

    Args:
        line1: 第一行数据（必需）
        line2: 第二行数据（必需）

    Returns:
        包含跃迁数据的字典

    Raises:
        ValueError: 如果第二行数据为空或格式不正确
    """
    # 解析第一行
    result: dict[str, str] = _parse_transition_first_line(line1)

    # 解析第二行（B规范数据）
    if not line2 or not line2.strip():
        raise ValueError(
            "Second line is required for electric transition but got empty line"
        )

    b_data: dict[str, str] = _parse_transition_second_line(line2)
    result.update(b_data)

    return result


def parse_magnetic_transition_line(line: str) -> dict[str, str]:
    """解析磁性跃迁数据行

    磁性跃迁只有一行数据，格式与电性跃迁的第一行相同，但只有M规范（对应C规范）

    例如：
    f1  1    2 +  f2  1    2 +        2368.69 M  0.00000D+00  0.00000D+00  0.00000D+00

    Args:
        line: 跃迁数据行

    Returns:
        包含跃迁数据的字典（B规范字段为None）
    """
    result: dict[str, str] = _parse_transition_first_line(line)
    # 磁性跃迁只有C规范（M规范），没有B规范
    result["transition_rate_B"] = ""
    result["gf_B"] = ""
    result["line_strength_B"] = ""
    return result


#######################################################################
# LSJ format parsing utilities
#######################################################################


def _parse_level_line(line: str) -> dict[str, str]:
    """解析 .lsj 格式的能级行

    格式: "  2-11257.5967049  5s(2).4d(10).5p(6).6s(2).4f(7)8S.5d_7D"
    第一部分格式: "double_J-能量"（能量一定是负数）

    Args:
        line: .lsj 格式的能级行

    Returns:
        {"double_J": 双倍J值, "energy": 能量字符串, "configuration": 配置字符串}

    Raises:
        ValueError: 如果格式不符合 "double_J-能量" 的要求
    """
    parts = line.split()
    # 第一部分: "double_J-能量"（能量一定是负数）
    double_j_energy: str = parts[0]
    if "-" not in double_j_energy:
        raise ValueError(f"Invalid level format: expected 'double_J-energy', got '{double_j_energy}'")

    double_j_str, energy_str = double_j_energy.split("-", 1)
    double_j: str = double_j_str.strip()
    energy: str = f"-{energy_str}"

    # 配置信息（剩余部分）
    config: str = " ".join(parts[1:]) if len(parts) > 1 else ""

    return {"double_J": double_j, "energy": energy, "configuration": config}


def _parse_lsj_magnetic_transition(
    upper_line: str, lower_line: str, energy_line: str, data_line: str
) -> dict[str, str]:
    """解析 .lsj 格式的磁性跃迁数据（4行）

    Args:
        upper_line: 上能级行
        lower_line: 下能级行
        energy_line: 能量和波长行
        data_line: 跃迁数据行

    Returns:
        包含跃迁数据的字典
    """
    upper: dict[str, str] = _parse_level_line(upper_line)
    lower: dict[str, str] = _parse_level_line(lower_line)

    # 解析能量行: "  12462.51 CM-1      8024.06 ANGS(VAC)      8023.23 ANGS(AIR)"
    energy_parts: list[str] = energy_line.split()
    energy_diff: str = energy_parts[0] if energy_parts else "0.0"

    # 解析数据行: " M2  S =  1.65556D-01   GF =  7.16288D-16   AKI =  1.48413D-08"
    data_parts: list[str] = data_line.split()
    transition_type: str = data_parts[0] if data_parts else ""

    # 使用正则表达式提取 S, GF, AKI
    found_data: dict[str, str] = dict(DATA_PATTERN.findall(data_line))
    line_strength: str = found_data.get("S", "0.0")
    gf: str = found_data.get("GF", "0.0")
    transition_rate: str = found_data.get("AKI", "0.0")

    # 转换 double_J 为 J 字符串
    upper_j: str = doubleJ_to_J(upper["double_J"])
    lower_j: str = doubleJ_to_J(lower["double_J"])

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
        "transition_rate_B": "",  # 磁性跃迁没有B规范
        "gf_B": "",
        "line_strength_B": "",
        "transition_type": transition_type,
    }


def _parse_lsj_electric_transition(
    upper_line: str,
    lower_line: str,
    energy_line: str,
    data_line: str,
    extra_line: str = "",
) -> dict[str, str]:
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
    upper: dict[str, str] = _parse_level_line(upper_line)
    lower: dict[str, str] = _parse_level_line(lower_line)

    # 解析能量行
    energy_parts: list[str] = energy_line.split()
    energy_diff: str = energy_parts[0] if energy_parts else "0.0"

    # 解析数据行（C规范）: " E1  S =  8.20207D-09   GF =  1.61683D-10   AKI =  9.08381D-04   dT =  0.99985"
    data_parts: list[str] = data_line.split()
    transition_type: str = data_parts[0] if data_parts else ""

    # 使用正则表达式提取 S, GF, AKI
    found_data: dict[str, str] = dict(DATA_PATTERN.findall(data_line))
    line_strength_c: str = found_data.get("S", "0.0")
    gf_c: str = found_data.get("GF", "0.0")
    transition_rate_c: str = found_data.get("AKI", "0.0")

    # 解析B规范数据（如果有）
    line_strength_b: str = "0.0"
    gf_b: str = "0.0"
    transition_rate_b: str = "0.0"

    if extra_line and extra_line.strip():
        extra_parts: list[str] = extra_line.split()
        if len(extra_parts) >= 3:
            # 格式: "          5.47381D-05         1.07902D-06          6.06225D+00"
            # 三个值分别是: S, GF, AKI
            line_strength_b = extra_parts[0] if extra_parts[0] else "0.0"
            gf_b = extra_parts[1] if len(extra_parts) > 1 else "0.0"
            transition_rate_b = extra_parts[2] if len(extra_parts) > 2 else "0.0"

    # 转换 double_J 为 J 字符串
    upper_j: str = doubleJ_to_J(upper["double_J"])
    lower_j: str = doubleJ_to_J(lower["double_J"])

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


class TransitionLoader(BaseLoader[list[str]]):
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

    Returns:
        File lines from one transition file or from merged directory results.
    """

    def __init__(
        self,
        file_path: str | Path,
        calculation_type: str = "rci",
        load_ct_lsj: bool = False,
    ) -> None:
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
        try:
            super().__init__(file_path)
        except FileNotFoundError as exc:
            raise ValueError(f"Path does not exist: {file_path}") from exc

        path_obj = self.file_path

        # 判断是文件还是目录
        self._is_directory: bool = path_obj.is_dir()

        if self._is_directory:
            # 目录模式：扫描文件
            self.file_paths: list[Path] = TransitionFileScan.scan_directory(
                file_path, calculation_type, load_ct_lsj
            )
        else:
            # 单文件模式
            self.file_paths = [path_obj]

        self.calculation_type: str = calculation_type
        self.load_ct_lsj: bool = load_ct_lsj

    @override
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
            return self._load_transition_file(self.file_path)

    def _load_transition_file(self, file_path: Path) -> list[str]:
        """加载跃迁文件（支持所有文本格式：.t, .ct, .t.lsj, .ct.lsj）

        Args:
            file_path: 文件路径

        Returns:
            文件行列表（原始文本数据）
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data_lines: list[str] = [line.rstrip("\n") for line in f.readlines()]

        # 添加空字符串作为结束标记（与旧代码兼容）
        data_lines.append("")

        return data_lines

    def _load_multiple_files(self) -> list[str]:
        """加载多个跃迁文件并合并

        Returns:
            合并后的文件行列表（文件间用两个空字符串分隔）

        Raises:
            ValueError: 没有有效文件可加载
        """
        all_data_lines: list[str] = []

        for i, file_path in enumerate(self.file_paths):
            try:
                data_lines: list[str] = self._load_transition_file(file_path)
                # 移除末尾的空字符串（避免多次添加）
                if data_lines and data_lines[-1] == "":
                    data_lines = data_lines[:-1]

                # 添加当前文件的数据
                all_data_lines.extend(data_lines)

                # 如果不是最后一个文件，添加两个空字符串作为分隔符
                if i < len(self.file_paths) - 1:
                    all_data_lines.extend(["", ""])

            except Exception as e:
                # 记录错误但继续处理其他文件
                print(f"Warning: Failed to load {file_path}: {e}")
                continue

        if not all_data_lines:
            raise ValueError("No valid transition data files could be loaded")

        # 添加最终的空字符串作为结束标记
        all_data_lines.append("")

        return all_data_lines

    def _parse_lsj_transitions(
        self, data_lines: list[str]
    ) -> list[dict[str, str]]:
        """解析 .lsj 格式的跃迁数据

        将原始行列表解析为跃迁数据列表

        Args:
            data_lines: 数据文件行列表

        Returns:
            跃迁数据列表
        """
        transitions: list[dict[str, str]] = []

        # 找到所有 "Transition between files:" 的位置
        transition_indices: list[int] = [
            idx for idx, line in enumerate(data_lines) if "Transition between files:" in line
        ]

        # 处理每个跃迁块
        for idx in transition_indices:
            # 检查是否有数据
            if idx + 3 < len(data_lines) and "Transition between files:" in data_lines[idx + 3]:
                continue

            # 数据从 idx + 5 开始
            i = idx + 5
            if i >= len(data_lines):
                continue

            while i < len(data_lines):
                # 跳过空行
                while i < len(data_lines) and not data_lines[i].strip():
                    i += 1
                    if i < len(data_lines) and "Transition between files:" in data_lines[i]:
                        break

                if i >= len(data_lines) or "Transition between files:" in data_lines[i]:
                    break

                # 需要至少4行
                if i + 3 >= len(data_lines):
                    break

                data_line = data_lines[i + 3]
                stripped_data = data_line.strip()
                if not stripped_data:
                    i += 1
                    continue

                # 判断跃迁类型并解析
                try:
                    if stripped_data[0] == "E":
                        # 电性跃迁：5行
                        extra_line = data_lines[i + 4] if i + 4 < len(data_lines) else ""
                        parsed = _parse_lsj_electric_transition(
                            data_lines[i + 1], data_lines[i], data_lines[i + 2], data_line, extra_line
                        )
                        transitions.append(parsed)
                        i += 5
                    elif stripped_data[0] == "M":
                        # 磁性跃迁：4行
                        parsed = _parse_lsj_magnetic_transition(
                            data_lines[i + 1], data_lines[i], data_lines[i + 2], data_line
                        )
                        transitions.append(parsed)
                        i += 4
                    else:
                        i += 1
                except (ValueError, IndexError):
                    i += 1

        return transitions

    def _parse_ct_transitions(
        self,
        data_lines_raw: list[str]
    ) -> list[dict[str, str]]:
        """解析 .ct/.t 格式的跃迁数据

        将原始行列表解析为跃迁数据列表

        Args:
            data_lines: 文件行列表

        Returns:
            跃迁数据列表
        """
        transitions: list[dict[str, str]] = []

        # 定位所有包含 "-pole transitions" 的行索引
        transition_indices: list[int] = [
            idx for idx, line in enumerate(data_lines_raw) if "-pole transitions" in line
        ]

        for idx in transition_indices:
            # 获取跃迁类型行，判断类型
            type_line = data_lines_raw[idx]
            is_electric = "Electric" in type_line
            is_magnetic = "Magnetic" in type_line

            # 数据从索引+5开始
            data_start_idx = idx + 5
            if data_start_idx >= len(data_lines_raw):
                continue

            if not data_lines_raw[data_start_idx] or data_lines_raw[data_start_idx].isspace():
                continue

            # 收集数据行
            data_lines: list[str] = []
            i = data_start_idx
            empty_count = 0
            while i < len(data_lines_raw):
                line = data_lines_raw[i]
                if line == "" or line.isspace():
                    empty_count += 1
                    if empty_count >= 2:
                        break
                    i += 1
                    continue
                empty_count = 0
                data_lines.append(line)
                i += 1

            # 解析数据行
            for line in data_lines:
                stripped = line.strip()
                if not (stripped.startswith("f1 ") or stripped.startswith("f2 ")):
                    continue

                try:
                    if is_electric:
                        # 电性跃迁：需要找下一行
                        line_idx = data_lines.index(line)
                        if line_idx + 1 < len(data_lines):
                            line2 = data_lines[line_idx + 1]
                            parsed = parse_electric_transition_line(line, line2)
                            # 从 type_line 提取 pole 数值
                            pole_match = re.search(r"2\*\*\(\s*(\d+)\s*\)-pole", type_line)
                            pole = pole_match.group(1) if pole_match else ""
                            parsed["transition_type"] = f"E{pole}"
                            transitions.append(parsed)
                    elif is_magnetic:
                        parsed = parse_magnetic_transition_line(line)
                        pole_match = re.search(r"2\*\*\(\s*(\d+)\s*\)-pole", type_line)
                        pole = pole_match.group(1) if pole_match else ""
                        parsed["transition_type"] = f"M{pole}"
                        transitions.append(parsed)
                except (ValueError, IndexError):
                    continue

        return transitions

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
        all_data: list[dict[str, str]] = []

        # 检查是否是 .lsj 格式
        if self.load_ct_lsj or (
            self.file_paths
            and TransitionFileScan.get_file_type(self.file_paths[0]).endswith(".lsj")
        ):
            # .lsj 格式：数据已经解析好了
            data_lines = self.load()
            transitions = self._parse_lsj_transitions(data_lines)
            all_data.extend(transitions)

        else:
            # .ct/.t 格式：需要解析原始行
            data_lines = self.load()
            transitions = self._parse_ct_transitions(data_lines)
            all_data.extend(transitions)

        return self._convert_data_types(pl.DataFrame(all_data))

    def _convert_data_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """将 DataFrame 中的字符串列转换为适当的数值类型

        Args:
            df: 原始 DataFrame（字符串类型）

        Returns:
            转换类型后的 DataFrame
        """
        # 定义所有需要转换的列
        c_cols = ["transition_rate_C", "gf_C", "line_strength_C"]
        b_cols = ["transition_rate_B", "gf_B", "line_strength_B"]

        # 将 energy 列转为 float
        df = df.with_columns(
            pl.col("energy")
            .str.strip_chars()
            .cast(pl.Float64)
        )
        # 将 B 规范列的空字符串替换为 null
        df = df.with_columns(
            pl.col(b_cols).replace("", None)
        )

        # C 规范列直接转换（没有空字符串）
        df = df.with_columns(
            pl.col(c_cols)
            .str.replace("D", "E")
            .str.strip_chars()
            .cast(pl.Float64)
        )

        # B 规范列条件转换
        for col in b_cols:
            df = df.with_columns(
                pl.when(pl.col("transition_type").str.starts_with("E"))
                .then(
                    pl.col(col)
                    .str.replace("D", "E")
                    .str.strip_chars()
                    .cast(pl.Float64)
                )
                .otherwise(None)
                .alias(col)
            )

        return df
