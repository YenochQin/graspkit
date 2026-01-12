#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Id :tool_function.py
@date :2024/05/07 11:11:09
@author :YenochQin (秦毅)
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

######################################################################


def level_print_title(Rydberg=109737.31568508):
    """
    print energy levels function
    """
    print(
        f"""
    Energy levels for ...
Rydberg constant is  {Rydberg}

---------------------------------------------
 No Pos  J  Parity   Energy Total    Levels
                      (a.u.)         (cm^-1)
---------------------------------------------
"""
    )


def level_J_value(j_idx: int) -> str:
    j_value_list = [
        "0",
        "1/2",
        "1",
        "3/2",
        "2",
        "5/2",
        "3",
        "7/2",
        "4",
        "9/2",
        "5",
        "11/2",
        "6",
        "13/2",
        "7",
        "15/2",
        "8",
        "17/2",
        "9",
        "19/2",
        "10",
        "21/2",
        "11",
        "23/2",
        "12",
        "25/2",
        "13",
        "27/2",
        "14",
        "29/2",
        "15",
        "31/2",
        "16",
        "33/2",
        "17",
        "35/2",
        "18",
        "37/2",
        "19",
        "39/2",
        "20",
        "41/2",
        "21",
        "43/2",
        "22",
    ]

    return j_value_list[j_idx - 1]


def level_parity(parity_idx: int) -> str:
    parity_list = ["+", "-"]

    return parity_list[parity_idx - 1]


def energy_au_cm(energy_au: float, Rydberg=109737.31568508) -> float:
    return energy_au * Rydberg * 2


######################################################################


def align_2d_list_columns(two_dimensional_list):
    """
    根据最长列的长度，对二维列表中的所有列进行补0对齐。

    参数:
    two_dimensional_list: 一个二维列表（列表的列表），其中每列的长度可能不同。

    返回:
    一个新的二维列表，其中所有列的长度都与最长列对齐。
    """
    # 获取最长列的长度
    max_length = max(len(column) for column in two_dimensional_list)

    # 对每列进行处理，确保长度与最长列对齐
    aligned_list = []
    for column in two_dimensional_list:
        # 计算当前列需要补充的0的数量
        fill_count = max_length - len(column)
        # 进行补0操作并添加到结果列表中
        aligned_column = np.pad(
            column, (0, fill_count), mode="constant", constant_values=0
        )
        aligned_list.append(aligned_column)

    return aligned_list


######################################################################


def int_nl_2_str_nl(n: int, kappa: int) -> str:
    r"""
    $j = l + 1/2, \kappa = -(l+1)$
    $j = l - 1/2, \kappa = +l $
    """
    l_list = ["s", "p", "d", "f", "g", "h", "i"]
    str_nl = ""
    if kappa > 0:
        l = kappa
        str_nl = f"{n}{l_list[l]}-"
    elif kappa < 0:
        l = -kappa - 1
        str_nl = f"{n}{l_list[l]} "
    else:
        print("error: kappa should not be zero")

    return str_nl


def str_subshell_2_kappa(str_subshell: str) -> int:
    r"""
    $j = l + 1/2, \kappa = -(l+1)$
    $j = l - 1/2, \kappa = +l $
    """
    kappa_value = {
        "s ": -1,
        "p-": 1,
        "p ": -2,
        "d-": 2,
        "d ": -3,
        "f-": 3,
        "f ": -4,
        "g-": 4,
        "g ": -5,
        "h-": 5,
        "h ": -6,
        "i-": 6,
        "i ": -7,
    }

    return kappa_value.get(str_subshell, 0)


######################################################################


def doubleJ_to_J(doubleJ):
    if doubleJ % 2 == 0:
        return f"{int(doubleJ / 2)}"
    else:
        return f"{doubleJ}/2"


######################################################################


# Function to read Fortran-style binary records (assume 4-byte record marker)
def read_fortran_record(file, dtype, count=1):
    # Read the record length (4 bytes before the data)
    record_len_before = np.fromfile(file, dtype=np.int32, count=1)[0]

    # Print for debugging
    # print(f"Record length before: {record_len_before}")

    # Read the actual data
    data = np.fromfile(file, dtype=dtype, count=count)

    # Read the record length (4 bytes after the data)
    record_len_after = np.fromfile(file, dtype=np.int32, count=1)[0]

    # Print for debugging
    # print(f"Record length after: {record_len_after}")

    # Verify that the record lengths match
    if record_len_before != record_len_after:
        raise ValueError(
            f"Record length mismatch: {record_len_before} != {record_len_after}"
        )

    return data


######################################################################


def chunk_string(s: str, n: int) -> list[str]:
    """将字符串分割成固定长度的块"""
    return [s[i : i + n] for i in range(0, len(s), n)]


######################################################################


def level_data_compare(levels_file_1: list, levels_file_2: list):
    level_data_1 = []
    level_data_2 = []
    skip_line = 0
    for i, line in enumerate(levels_file_1):  # 使用enumerate获取行号
        if "No Pos  J" in line:
            skip_line = i + 3
            break
    level_data_1 = levels_file_1[skip_line:]
    for i, line in enumerate(levels_file_2):  # 使用enumerate获取行号
        if "No Pos  J" in line:
            skip_line = i + 3
            break
    level_data_2 = levels_file_2[skip_line:]

    if len(level_data_1) != len(level_data_2):
        raise ValueError("The number of levels is not equal!")

    for i, (line1, line2) in enumerate(zip(level_data_1, level_data_2)):
        if not line1 or not line2:  # 跳过空行
            continue
        if line1.split()[-1] != line2.split()[-1]:
            raise ValueError(f"Configuration state functions differ at line {i + 1}")

    return True


######################################################################


def LS_shell_full_charged(shell_name: str, shell_charged_num: int) -> bool:
    full_charged = {"s": 2, "p": 6, "d": 10, "f": 14, "g": 18, "h": 22, "i": 26}
    return full_charged.get(shell_name, 0) == shell_charged_num


######################################################################
