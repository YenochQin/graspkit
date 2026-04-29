# -*- encoding: utf-8 -*-
import re

import numpy as np
from numpy.typing import NDArray
import polars as pl

from ..utils.data_modules import CSFs
from ..utils.tool_function import chunk_string

#######################################################################
# CSFs source data compress to a simplified form
#######################################################################


def subshell_charged_state(subshell_CSF: str) -> dict[str, str]:
    """Parse the charge state encoded in a subshell CSF fragment.

    Args:
        subshell_CSF: Fixed-width subshell fragment such as ``"5s ( 2)"``.

    Returns:
        Dictionary with the main quantum number, subshell label, and charge
        count stored as strings.
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
    """Check whether a relativistic subshell is fully occupied.

    Args:
        subshell_name: Relativistic subshell label, for example ``"p-"`` or
            ``"d "``.
        subshell_charged_num: Number of electrons occupying the subshell.

    Returns:
        True if the electron count equals the configured full occupation for
        the subshell.
    """
    full_charged: dict[str, int] = {
        "s ": 2,
        "p-": 2,
        "p ": 4,
        "d-": 4,
        "d ": 6,
        "f-": 6,
        "f ": 8,
        "g-": 8,
        "g ": 10,
        "h-": 10,
        "h ": 12,
        "i-": 12,
        "i ": 14,
    }
    return full_charged.get(subshell_name, 0) == subshell_charged_num


def CSF_subshell_split(CSFs_configuration_raw: str) -> list[str]:
    """Split a raw CSF configuration line into fixed-width subshell fields.

    Args:
        CSFs_configuration_raw: CSF configuration line with trailing newline
            already removed.

    Returns:
        List of 9-character subshell fields.
    """
    # CSFs_configuration_raw need drop '\n' first !!!

    subshells_charged: list[str] = [
        CSFs_configuration_raw[i : i + 9]
        for i in range(0, len(CSFs_configuration_raw), 9)
    ]

    return subshells_charged


def get_CSFs_peel_subshells(CSFs_file_data: CSFs) -> list[str]:
    """获取CSFs文件中的peel subshells列表

    Args:
        CSFs_file_data: CSFs文件数据对象

    Returns:
        list: 清理后的peel subshells列表，每个元素都已去除多余空格
    """
    # 获取原始字符串并去除前后的空白字符(包括换行符)
    peel_subshells: str = CSFs_file_data.subshell_info_raw[-1].strip()

    # 分割字符串并过滤掉空字符串，同时对每个子串去除前后空格
    return [s.strip() for s in peel_subshells.split() if s.strip()]


#######################################################################


def csf_J(csf_3rd_line: str) -> tuple[str, str]:
    """Extract the total angular momentum and parity from a CSF third line.

    Args:
        csf_3rd_line: Third line of a GRASP CSF record.

    Returns:
        Tuple containing the raw J string and parity symbol.
    """
    # 按空格分割字符串
    parts: list[str] = csf_3rd_line.split()

    # 最后一个部分包含J值和宇称
    j_parity_part: str = parts[-1]

    # 分离J值和宇称
    j_str: str = j_parity_part[:-1]  # 去掉最后一个字符（宇称符号）
    parity: str = j_parity_part[-1]  # 最后一个字符就是宇称符号

    # 返回J字符串和宇称符号
    return j_str, parity


def J_to_doubleJ(J_str: str) -> int:
    """Convert a J quantum-number string to its doubled integer value.

    Args:
        J_str: Angular momentum value such as ``"3/2"``, ``"2"``, or
            ``"5/2"``.

    Returns:
        Integer value of ``2J``.
    """
    J_str = J_str.strip()
    if "/" in J_str:
        # 处理半整数情况
        numerator, _ = map(int, J_str.split("/"))
        return numerator
    else:
        # 处理整数情况
        return int(J_str) * 2



def CSF_item_2_dict(CSF_item_list: list[str]) -> dict[str, str]:
    """Convert the three raw lines of a CSF item into a small metadata dict.

    Args:
        CSF_item_list: Three-line CSF record containing the subshell line,
            intermediate coupling line, and final coupling line.

    Returns:
        Dictionary with raw CSF fields plus parsed final parity and J value.
    """
    CSF_item_dict: dict[str, str] = {}

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
    orbs_occupied_idxs: list[int] = []  # 记录哪些轨道被占用

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
            temp_middle_str = middle_line_item.split(";")[-1].strip()
            temp_middle_item = J_to_doubleJ(temp_middle_str)  # 转换为2J值

        # 处理第三行数据（耦合J值）
        temp_coupling_item = 0
        if not coupling_line_item.isspace():  # 如果第三行有值
            temp_coupling_str = coupling_line_item.strip()
            temp_coupling_item = J_to_doubleJ(temp_coupling_str)
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
    remaining_orbs_idxs = list(all_orbs_idxs - occupied_orbs_idxs)  # 未占用轨道索引

    # 第九步：为未占用轨道填充最终J值
    for idx in remaining_orbs_idxs:
        csf_descriptor[idx * 3 + 2] = final_double_J  # 只设置耦合J值位置

    return csf_descriptor


#######################################################################


def batch_process_csfs_to_descriptors(CSFs_file_data: CSFs) -> np.ndarray:
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
    peel_subshells_list: list[str] = get_CSFs_peel_subshells(CSFs_file_data)

    all_descriptors: list[NDArray[np.float64]] = []

    for block_idx, block in enumerate(CSFs_file_data.CSFs_block_data):
        # 遍历块中的每个CSF项
        for csf_idx, csf_item in enumerate(block):
            try:
                # 检查CSF项是否包含3行
                if len(csf_item) != 3:
                    print(
                        f"Warning: CSF item in block {block_idx}, idx {csf_idx} has {len(csf_item)} lines instead of 3. Skipping..."
                    )
                    continue
                descriptor: NDArray[np.float64] = parse_csf_2_descriptor(peel_subshells_list, csf_item)
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
    CSFs_file_header: dict[str, dict[str, str]], CSFs_file_data: pl.DataFrame
) -> np.ndarray:
    """Convert parquet-backed CSF rows into descriptor vectors.

    Args:
        CSFs_file_header: Parsed CSF header containing peel subshell metadata.
        CSFs_file_data: DataFrame with ``line1``, ``line2``, and ``line3``
            columns for each CSF record.

    Returns:
        NumPy matrix with one descriptor row per CSF.
    """
    # 获取剥离子壳层列表
    peel_subshells = CSFs_file_header["header_info"]["header_lines"][3]
    peel_subshells_list = [s.strip() for s in peel_subshells.split() if s.strip()]

    # 【修改点】：将 with_columns 改为 select
    # 这样返回的 descriptors_df 将只包含 "descriptor" 这一列
    descriptors_df = CSFs_file_data.select(
        descriptor=pl.concat_list(["line1", "line2", "line3"]).map_elements(
            lambda x: parse_csf_2_descriptor(peel_subshells_list, x).tolist(),
            return_dtype=pl.List(pl.Float64),
        )
    )

    print(f"Successfully processed {CSFs_file_data.shape[0]} CSFs")
    # 这里的 shape 列数应该是 1
    print(f"Descriptor df shape: {descriptors_df.shape}")
    print(f"Number of orbitals: {len(peel_subshells_list)}")

    numpy_matrix = np.array(descriptors_df["descriptor"].to_list())
    return numpy_matrix


#######################################################################
