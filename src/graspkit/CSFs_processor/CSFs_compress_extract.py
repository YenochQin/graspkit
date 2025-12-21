# -*- encoding: utf-8 -*-
"""
@Id :CSFs_compress_extract.py
@date :2025/03/10 16:02:06
@author :YenochQin (秦毅)
"""

import random
import re
import numpy as np
import polars as pl
from ..utils.progress_manager import wrap_iterator

from ..utils.tool_function import str_subshell_2_kappa, chunk_string, str_subshell_2_kappa
from ..utils.data_modules import CSFs

#######################################################################
# CSFs source data compress to a simplified form
#######################################################################


def subshell_charged_state(subshell_CSF: str) -> dict[str, str]:
    """
    解析轨道电荷状态，返回包含主量子数、轨道名称和电荷数的字典。
    """
    temp_subshell_state = re.findall(
        r"([0-9]*)([s,p,d,f,g][\s,-])\( (\d+)\)", subshell_CSF
    )[0]
    main_quantum_num = temp_subshell_state[0]
    subshell_name = temp_subshell_state[1]
    subshell_charged_num = temp_subshell_state[2]  # 保持为字符串
    return {
        "subshell_main_quantum_num": main_quantum_num,
        "subshell_name": subshell_name,
        "subshell_charged_num": subshell_charged_num,
    }


def if_subshell_full_charged(subshell_name: str, subshell_charged_num: int) -> bool:
    full_charged = {
        "s ": 2,
        "p-": 2,
        "p ": 4,
        "d-": 4,
        "d ": 6,
        "f-": 6,
        "f ": 8,
        "g-": 8,
        "g ": 10,
    }
    return full_charged.get(subshell_name, 0) == subshell_charged_num

def CSF_subshell_split(CSFs_configuration_raw: str) -> list:
    # CSFs_configuration_raw need drop '\n' first !!!

    subshells_charged = [
        CSFs_configuration_raw[i : i + 9]
        for i in range(0, len(CSFs_configuration_raw), 9)
    ]

    return subshells_charged

def get_CSFs_peel_subshells(CSFs_file_data: CSFs) -> list:
    """获取CSFs文件中的peel subshells列表

    Args:
        CSFs_file_data: CSFs文件数据对象

    Returns:
        list: 清理后的peel subshells列表，每个元素都已去除多余空格
    """
    # 获取原始字符串并去除前后的空白字符(包括换行符)
    peel_subshells = CSFs_file_data.subshell_info_raw[-1].strip()

    # 分割字符串并过滤掉空字符串，同时对每个子串去除前后空格
    return [s.strip() for s in peel_subshells.split() if s.strip()]

def CSF_subshell_transform(
    subshells_charged: str, CSFs_file_Peel_subshells: list
) -> list[int]:
    ## 暂时用不了
    subshells_charged_list = CSF_subshell_split(subshells_charged)

    filled_dict = {}

    for item in subshells_charged_list:
        subshell, e_charges = re.findall(r"([0-9]*[s,p,d,f,g][\s,-])\( (\d+)\)", item)[
            0
        ]
        filled_dict[subshell] = int(e_charges)

    transform_subshells_charged = [
        filled_dict.get(subshell, 0) for subshell in CSFs_file_Peel_subshells
    ]

    return transform_subshells_charged

#######################################################################

def CSF_subshell_compress(CSF_configuration_raw: str):
    """
    compress CSF subshell from
      5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)  4f-( 1)  4f ( 6)  5d ( 1)
    to
    '5|-1|2;4|2|4;4|-3|6;5|1|2;5|-2|4;6|-1|2;4|3|1;4|-4|6;5|-3|1;'
    """
    subshells_charged = re.split(r"(\d*\w[\s|-]\(\s\d*\))", CSF_configuration_raw)
    subshells_charged = [item for item in subshells_charged if item.strip()]

    print(subshells_charged)
    compressed_CSF = ""

    for subshell in subshells_charged:
        temp_subshell_charged_state = subshell_charged_state(subshell)
        temp_subshell_kappa = str_subshell_2_kappa(
            temp_subshell_charged_state["subshell_name"]
        )
        compressed_CSF += f"{temp_subshell_charged_state['subshell_main_quantum_num']}|{temp_subshell_kappa}|{temp_subshell_charged_state['subshell_charged_num']};"

    return compressed_CSF

def CSF_compress(CSF_raw: list) -> str:
    """
    compress CSF
    """
    if len(CSF_raw) != 3:
        raise ValueError("CSF_raw need to be 3 line")

    return CSF_subshell_compress(CSF_raw[0]) + "".join(CSF_raw[1:])

def CSF_subshell_extract(simplified_str):
    """
    extract CSF subshell from
    '5|-1|2;4|2|4;4|-3|6;5|1|2;5|-2|4;6|-1|2;4|3|1;4|-4|6;5|-3|1;'
    to
      5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)  4f-( 1)  4f ( 6)  5d ( 1)
    """
    reverse_kappa = {
        -1: "s ",
        1: "p-",
        -2: "p ",
        2: "d-",
        -3: "d ",
        3: "f-",
        -4: "f ",
        4: "g-",
        -5: "g ",
        5: "h-",
        -6: "h ",
        6: "i-",
        -7: "i ",
    }

    items = simplified_str.split(";")
    restored = []

    for item in items:
        try:
            n, kappa, value = item.split("|")
            orbital_key = reverse_kappa.get(int(kappa), "")
            restored.append(f"{n}{orbital_key}({value})")
        except:
            continue

    return "  ".join(restored)

#######################################################################

def csf_J(csf_3rd_line: str):
    """
    extract J from CSF 3rd line
    """
    # 按空格分割字符串
    parts = csf_3rd_line.split()

    # 最后一个部分包含J值和宇称
    j_parity_part = parts[-1]

    # 分离J值和宇称
    j_str = j_parity_part[:-1]  # 去掉最后一个字符（宇称符号）
    parity = j_parity_part[-1]  # 最后一个字符就是宇称符号

    # 返回J字符串和宇称符号
    return j_str, parity

def J_to_doubleJ(J_str: str) -> int:
    """
    将J字符串转换为二倍值(2J)
    示例:
    '3/2' -> 3
    '2' -> 4
    '5/2' -> 5
    """
    J_str = J_str.strip()
    if "/" in J_str:
        # 处理半整数情况
        numerator, _ = map(int, J_str.split("/"))
        return numerator
    else:
        # 处理整数情况
        return int(J_str) * 2

def CSF_info_2_dict(CSF_item_list: list[str]) -> dict:
    # 解析 subshell 信息
    CSF_info_dict = {}  # 初始化为字典而不是调用CSF_subshell_split
    CSF_info_dict["subshells"] = CSF_subshell_split(CSF_item_list[0])

    # 添加 temp_coupled_j 和 final_coupled_j_parity
    CSF_info_dict.update(
        {
            "temp_coupled_j": CSF_item_list[1],
            "final_coupled_j_parity": CSF_item_list[2],
        }
    )

    # 解析 final_coupled_j_parity 中的 J 和 parity
    j_p = CSF_item_list[2].split()[-1]  # 提取 J 和 parity 部分
    CSF_info_dict["parity"] = j_p[-1]  # parity 是最后一个字符
    CSF_info_dict["J"] = j_p[:-1]  # J 是 parity 之前的部分

    return CSF_info_dict

def CSF_item_2_dict(CSF_item_list: list[str]) -> dict:
    CSF_item_dict = {}

    CSF_item_dict.update(
        {
            "subshell_raw": CSF_item_list[0],
            "temp_coupled_j": CSF_item_list[1],
            "final_coupled_j_parity": CSF_item_list[2],
        }
    )

    j_p = CSF_item_list[2].split()[-1]  # 提取 J 和 parity 部分
    CSF_item_dict["parity"] = j_p[-1]  # parity 是最后一个字符
    CSF_item_dict["J"] = j_p[:-1]  # J 是 parity 之前的部分

    return CSF_item_dict

def get_CSFs_file_info(csfs_file_data: list) -> dict:
    """
    Process CSF file data and extract structured information.

    Args:
        csfs_file_data: Raw CSF data list containing subshell info and CSFs entries

    Returns:
        dictionary containing:
        - subshell_info_raw: Original header lines
        - parsed subshell parameters (n, orbitals, etc.)
        - star_idxs: Positions of CSF separators
        - CSFs_j_value: Collected J-values from CSFs
    """
    # Extract first 4 lines containing subshell information
    subshell_info = csfs_file_data[0:4]

    CSFs_file_info = {}
    CSFs_file_info["subshell_info_raw"] = subshell_info

    # Process subshell info pairs (parameter name + values)
    for i in range(0, len(subshell_info), 2):
        key = subshell_info[i].rstrip(":")  # Remove colon from key
        value = subshell_info[i + 1].split()  # Split values into list
        CSFs_file_info[key] = value

    # Find all CSF separators ('*') in the data
    star_idxs = []
    for idx, value in enumerate(csfs_file_data):
        if "*" in value:
            star_idxs.append(idx)
    CSFs_file_info["star_idxs"] = star_idxs

    # Collect J-values preceding each separator and the final value
    CSFs_j_value = []
    CSFs_block_parity = []
    prev_idx = 5
    CSFs_file_info["CSFs_block_data"] = []  # 初始化 CSFs_block_data 列表

    for idx in star_idxs:
        temp_j_value, temp_parity = csf_J(csfs_file_data[idx - 1])
        CSFs_j_value.append(temp_j_value)
        CSFs_block_parity.append(temp_parity)
        # 处理每个块的数据，而不是一次性存储所有块
        block_data = csfs_file_data[prev_idx:idx]
        if len(block_data) % 3 != 0:
            raise ValueError("CSFs_list length must be a multiple of 3")
        CSFs_file_info["CSFs_block_data"].append(block_data)  # 添加当前块的数据
        prev_idx = idx + 1

    temp_j_value, temp_parity = csf_J(csfs_file_data[-1])
    CSFs_j_value.append(temp_j_value)
    CSFs_block_parity.append(temp_parity)
    CSFs_file_info["CSFs_j_value"] = CSFs_j_value

    CSFs_parity = set(CSFs_block_parity)
    if len(CSFs_parity) == 1:
        CSFs_file_info["parity"] = list(CSFs_parity)[0]

    # 处理最后一个块的数据
    last_block_data = csfs_file_data[prev_idx:]
    if len(last_block_data) % 3 != 0:
        raise ValueError("CSFs_list length must be a multiple of 3")
    CSFs_file_info["CSFs_block_data"].append(last_block_data)  # 添加最后一个块的数据

    return CSFs_file_info

#######################################################################

def split_by_asterisk(lines):
    """
    将列表按单独一个星号行分割为二维列表

    :param lines: readlines 读取的列表
    :param keep_empty: 是否保留空块（默认过滤）
    :return: 二维列表，例如 [[块1行], [块2行], ...]
    """
    result = []
    current_chunk = []

    for line in lines:
        # 严格匹配：仅当行内容为单个星号（含换行符）
        if line.strip() == "*":
            result.append(current_chunk)
            current_chunk = []
        else:
            current_chunk.append(line)  # 可选：去除换行符

    # 添加最后一个块
    result.append(current_chunk)

    return result

def shuffle_three_line_groups(lst):
    """
    将列表按每三行一组随机打乱顺序
    示例输入格式：
    ['行1', '行2', '行3', '行4', '行5', '行6', ...]
    输出格式：
    ['行4', '行5', '行6', '行1', '行2', '行3', ...]
    """
    # 检查是否能被3整除
    if len(lst) % 3 != 0:
        raise ValueError("列表长度必须是3的倍数")

    # 将列表分成三元组
    groups = [lst[i : i + 3] for i in range(0, len(lst), 3)]

    # 打乱组顺序
    random.shuffle(groups)

    # 重新展开为平铺列表
    shuffled = [line for group in groups for line in group]

    return shuffled


#######################################################################
################### CSFs descriptor      #############################
#######################################################################

def parse_csf_2_descriptor(
    peel_subshells_list: list[str], csf: list[str]
) -> np.ndarray:
    """
    将CSF（Configuration State Function）数据解析为描述符数组

    Args:
        peel_subshells_list (list[str]): 剥离子壳层名称列表，如 ['5s', '4d-', '4d', ...]
        csf (list[str]): CSF数据的三行字符串列表
            - 第一行：子壳层和电子数信息
            - 第二行：中间J耦合值
            - 第三行：最终耦合和总J值

    Returns:
        np.ndarray: 长度为 3*len(peel_subshells_list) 的浮点数组
            每个子壳层对应3个数值：[电子数, 中间J值, 耦合J值]

    Example:
        >>> peel_subshells = ['5s', '4d-', '4d']
        >>> csf_data = [
        ...     '  5s ( 2)  4d-( 4)  4d ( 6)',
        ...     '                   3/2      ',
        ...     '                        4-  '
        ... ]
        >>> result = parse_csf_2_descriptor(peel_subshells, csf_data)
        >>> # 返回 [2.0, 0.0, 8.0, 4.0, 3.0, 8.0, 6.0, 0.0, 8.0] 的数组
    """

    # 第一步：预处理CSF的三行数据，去除末尾换行符并统一长度
    subshells_line, middle_line_raw, coupling_line_raw = [line.rstrip() for line in csf]
    line_length = len(subshells_line)  # 以第一行长度为标准
    middle_line = middle_line_raw.ljust(line_length)  # 左对齐并填充到指定长度
    coupling_line = coupling_line_raw[4:-5].ljust(
        line_length
    )  # 去除前4位和后5位，然后左对齐

    # 第二步：提取最终J值（从第三行的后5位中提取）
    final_J = coupling_line_raw[-5:-1]  # 例如：'4-' 或 '3/2'
    final_double_J = J_to_doubleJ(final_J)  # 转换为2J的整数表示

    # 第三步：将三行数据按每9个字符分块处理
    subshell_list = chunk_string(subshells_line, 9)  # 子壳层信息块
    middle_line_list = chunk_string(middle_line, 9)  # 中间耦合信息块
    coupling_line_list = chunk_string(coupling_line, 9)  # 耦合信息块

    # 第四步：初始化描述符数组和已占用轨道索引列表
    csf_descriptor = np.zeros(3 * len(peel_subshells_list), dtype=np.float32)
    orbs_occupied_idxs = []  # 记录哪些轨道被占用

    # 第五步：遍历每个子壳层块，提取和处理信息
    for i, (subshell_charges, middle_line_item, coupling_line_item) in enumerate(
        zip(subshell_list, middle_line_list, coupling_line_list)
    ):
        # 提取子壳层名称和电子数
        subshell = subshell_charges[:5].strip()  # 前5位是子壳层名称，如 '5s'
        subshell_electron_num = int(subshell_charges[6:8])  # 第6-8位是电子数
        is_last = i == len(subshell_list) - 1  # 判断是否为最后一个子壳层

        # 处理第二行数据（中间J耦合值）
        temp_middle_item = 0
        if not middle_line_item.isspace():  # 如果不是空白
            # 如果有分号分隔的多个值，取最后一个
            temp_middle_item = middle_line_item.split(";")[-1].strip()
            temp_middle_item = J_to_doubleJ(temp_middle_item)  # 转换为2J值

        # 处理第三行数据（耦合J值）
        temp_coupling_item = 0
        if not coupling_line_item.isspace():  # 如果第三行有值
            temp_coupling_item = coupling_line_item.strip()
            temp_coupling_item = J_to_doubleJ(temp_coupling_item)
        elif not middle_line_item.isspace():  # 如果第三行没值但第二行有值
            temp_coupling_item = temp_middle_item  # 使用第二行的值

        # 特殊处理：如果是最后一个子壳层，使用最终J值
        if is_last:
            temp_coupling_item = final_double_J

        # 第六步：在轨道列表中查找当前子壳层的索引
        try:
            orbs_idx = peel_subshells_list.index(subshell)
            descriptor_idx = orbs_idx * 3  # 每个轨道占用3个位置
        except ValueError:
            print(f"Warning: {subshell} not found in orbs list")
            continue

        # 第七步：记录已占用轨道并填充描述符数组
        orbs_occupied_idxs.append(orbs_idx)
        csf_descriptor[descriptor_idx : descriptor_idx + 3] = [
            subshell_electron_num,  # 电子数
            temp_middle_item,  # 中间J值
            temp_coupling_item,  # 耦合J值
        ]

    # 第八步：处理未占用的轨道（使用集合运算找到差集）
    all_orbs_idxs = set(range(len(peel_subshells_list)))  # 所有轨道索引
    occupied_orbs_idxs = set(orbs_occupied_idxs)  # 已占用轨道索引
    remaining_orbs_idxs = list(
        all_orbs_idxs - occupied_orbs_idxs
    )  # 未占用轨道索引

    # 第九步：为未占用轨道填充最终J值
    for idx in remaining_orbs_idxs:
        csf_descriptor[idx * 3 + 2] = final_double_J  # 只设置耦合J值位置

    return csf_descriptor

def parse_csf_2_descriptor_with_subshell(
    peel_subshells_list: list[str], csf: list[str]
) -> np.ndarray:
    """
    包含子壳层信息的CSF描述符解析函数

    描述符格式（每个轨道5个数值）：
    [主量子数, kappa值, 电子数, 中间J值, 耦合J值]

    Args:
        peel_subshells_list: 剥离子壳层列表，如 ['5s', '4d-', '4d', ...]
        csf: CSF的三行数据

    Returns:
        np.ndarray: 长度为 5*len(peel_subshells_list) 的描述符数组
    """

    # 预处理CSF数据
    subshells_line, middle_line_raw, coupling_line_raw = [line.rstrip() for line in csf]
    line_length = len(subshells_line)
    middle_line = middle_line_raw.ljust(line_length)
    coupling_line = coupling_line_raw[4:-5].ljust(line_length)

    # 提取最终J值
    final_J = coupling_line_raw[-5:-1]
    final_double_J = J_to_doubleJ(final_J)

    # 分块处理
    subshell_list = chunk_string(subshells_line, 9)
    middle_line_list = chunk_string(middle_line, 9)
    coupling_line_list = chunk_string(coupling_line, 9)

    # 初始化描述符数组（每个轨道5个数值）
    csf_descriptor = np.zeros(5 * len(peel_subshells_list), dtype=np.float32)
    orbs_occupied_idxs = []

    # 首先为所有轨道填充子壳层信息（主量子数和kappa值）
    for idx, subshell in enumerate(peel_subshells_list):
        # 直接解析子壳层名称
        # 提取主量子数（数字部分）
        main_quantum_num = int("".join(filter(str.isdigit, subshell)))

        # 提取轨道类型（字母部分，包括可能的'-'）
        orbital_part = "".join(filter(lambda x: not x.isdigit(), subshell))
        # 确保格式正确（如 's ', 'd-', 'f '等）
        if not orbital_part.endswith(" ") and not orbital_part.endswith("-"):
            orbital_part += " "

        kappa_value = str_subshell_2_kappa(orbital_part)

        descriptor_idx = idx * 5
        csf_descriptor[descriptor_idx] = main_quantum_num  # 第1位：主量子数
        csf_descriptor[descriptor_idx + 1] = kappa_value  # 第2位：kappa值

    # 处理每个子壳层的电子数和J值信息
    for i, (subshell_charges, middle_line_item, coupling_line_item) in enumerate(
        zip(subshell_list, middle_line_list, coupling_line_list)
    ):
        subshell = subshell_charges[:5].strip()
        subshell_electron_num = int(subshell_charges[6:8])
        is_last = i == len(subshell_list) - 1

        # 判断轨道是否填满
        is_full = if_subshell_full_charged(subshell, subshell_electron_num)

        # 处理中间J值
        temp_middle_item = 0
        if not middle_line_item.isspace():
            temp_middle_item = middle_line_item.split(";")[-1].strip()
            temp_middle_item = J_to_doubleJ(temp_middle_item)
            # 未填满轨道J值乘以2增强特征
            if not is_full:
                temp_middle_item *= 2

        # 处理耦合J值
        temp_coupling_item = 0
        if not coupling_line_item.isspace():
            temp_coupling_item = coupling_line_item.strip()
            temp_coupling_item = J_to_doubleJ(temp_coupling_item)
            if not is_full:
                temp_coupling_item *= 2
        elif not middle_line_item.isspace():
            temp_coupling_item = temp_middle_item

        # 最后一个子壳层使用最终J值
        if is_last:
            temp_coupling_item = final_double_J * (2 if not is_full else 1)

        # 查找轨道索引
        try:
            orbs_idx = peel_subshells_list.index(subshell)
            descriptor_idx = orbs_idx * 5
        except ValueError:
            print(f"Warning: {subshell} not found in orbs list")
            continue

        orbs_occupied_idxs.append(orbs_idx)

        # 填满轨道J值设为0
        if is_full:
            temp_middle_item = 0
            temp_coupling_item = 0

        # 填充描述符的第3、4、5位
        csf_descriptor[descriptor_idx + 2] = subshell_electron_num  # 第3位：电子数
        csf_descriptor[descriptor_idx + 3] = temp_middle_item  # 第4位：中间J值
        csf_descriptor[descriptor_idx + 4] = temp_coupling_item  # 第5位：耦合J值

    # 处理未占用的轨道（第5位填最终J值的二倍）
    all_orbs_idxs = set(range(len(peel_subshells_list)))
    occupied_orbs_idxs = set(orbs_occupied_idxs)
    remaining_orbs_idxs = list(all_orbs_idxs - occupied_orbs_idxs)

    for idx in remaining_orbs_idxs:
        csf_descriptor[idx * 5 + 4] = final_double_J * 2  # 第5位：最终J值的二倍

    return csf_descriptor

#######################################################################

def batch_process_csfs_to_descriptors(
                                    CSFs_file_data: CSFs
                                ) -> np.ndarray:
    """
    批量处理CSFs文件中的所有CSF数据，转换为描述符数组

    Args:
        CSFs_file_data (CSFs): CSFs文件数据对象
        progress_bar (bool): 是否显示进度条

    Returns:
        np.ndarray: 形状为 (总CSF数量, 3*轨道数量) 的描述符数组

    Example:
        >>> # 基本使用
        >>> descriptors = batch_process_csfs_to_descriptors(csfs_data)
        >>> # 然后选择性保存
        >>> save_descriptors(descriptors, 'output/csf_descriptors', 'csv')
    """
    # 获取剥离子壳层列表
    peel_subshells_list = get_CSFs_peel_subshells(CSFs_file_data)

    all_descriptors = []

    for block_idx, block in enumerate(CSFs_file_data.CSFs_block_data):
        # 遍历块中的每个CSF项
        for csf_idx, csf_item in enumerate(
            wrap_iterator(block, desc=f"处理CSF块 {block_idx + 1}")
        ):
            try:
                # 检查CSF项是否包含3行
                if len(csf_item) != 3:
                    print(
                        f"Warning: CSF item in block {block_idx}, idx {csf_idx} has {len(csf_item)} lines instead of 3. Skipping..."
                    )
                    continue
                descriptor = parse_csf_2_descriptor_with_subshell(
                        peel_subshells_list, csf_item
                    )
                all_descriptors.append(descriptor)

            except Exception as e:
                print(f"Error processing CSF in block {block_idx}, item {csf_idx}: {e}")
                print(f"CSF data: {csf_item}")
                continue

    if not all_descriptors:
        raise ValueError("No valid CSF data processed!")

    # 转换为numpy数组
    descriptors_array = np.stack(all_descriptors)

    print(f"Successfully processed {len(descriptors_array)} CSFs")
    print(f"Descriptor array shape: {descriptors_array.shape}")
    print(f"Number of orbitals: {len(peel_subshells_list)}")

    return descriptors_array

def batch_process_csfs_parquet_to_descriptors(
                                    CSFs_file_header: dict,
                                    CSFs_file_data: pl.DataFrame
                                ) -> np.ndarray:
    """
    批量处理CSFs文件中的所有CSF数据，转换为描述符数组
    """
    # 获取剥离子壳层列表
    peel_subshells = CSFs_file_header["header_info"]["header_lines"][3]
    peel_subshells_list = [s.strip() for s in peel_subshells.split() if s.strip()]

    # 【修改点】：将 with_columns 改为 select
    # 这样返回的 descriptors_df 将只包含 "descriptor" 这一列
    descriptors_df = CSFs_file_data.select(
        descriptor = pl.concat_list(["line1", "line2", "line3"])
            .map_elements(
                lambda x: parse_csf_2_descriptor_with_subshell(peel_subshells_list, x).tolist(),
                return_dtype=pl.List(pl.Float64) 
            )
    )

    print(f"Successfully processed {CSFs_file_data.shape[0]} CSFs")
    # 这里的 shape 列数应该是 1
    print(f"Descriptor df shape: {descriptors_df.shape}") 
    print(f"Number of orbitals: {len(peel_subshells_list)}")

    numpy_matrix = np.array(descriptors_df["descriptor"].to_list())
    return numpy_matrix

#######################################################################
