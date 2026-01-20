# -*- encoding: utf-8 -*-
"""
@Id :csf_loader.py
@date :2026/01/19
@author :YenochQin (秦毅)
"""

import re
from .base_loader import BaseLoader
from ...utils.data_modules import CSFs


class CSFLoader(BaseLoader[CSFs]):
    """CSF配置状态函数文件加载器 (.c文件)

    用于加载 GRASP2018 生成的 CSF 配置文件。
    解析子轨道信息和块结构。
    """

    def load(self) -> CSFs:
        """加载CSF数据

        Returns:
            CSFs对象，包含：
                - subshell_info_raw: 子轨道信息列表
                - CSFs_block_j_value: 块J值列表
                - parity: 宇称（奇/偶）
                - CSFs_block_data: 块数据列表（每块3行一组CSF）
                - CSFs_block_length: 每块的CSF数量
                - block_num: 块数量
        """
        # 读取文本文件
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip() for line in f.readlines()]

        subshell_info_raw = lines[:4]

        # 查找所有包含星号的行
        star_idxs = [idx for idx, line in enumerate(lines) if "*" in line]

        CSFs_block_j_value = []
        CSFs_block_parity = []
        CSFs_block_data = []
        CSFs_block_length = []

        # 处理每个块
        for i, idx in enumerate(star_idxs):
            # 获取J值和宇称（在星号前一行的第3行）
            line_idx = idx - 1
            j_parity_line = lines[line_idx].strip()

            # 解析J值和宇称
            match = re.match(r"(\d+/)?\s+([+-])\s+$", j_parity_line)
            if not match:
                raise ValueError(f"Invalid J/parity format: {j_parity_line}")

            j_str = match.group(1)
            parity = match.group(2)

            CSFs_block_j_value.append(j_str)
            CSFs_block_parity.append(parity)

            # 提取块数据（从上一个星号后到当前星号）
            prev_idx = star_idxs[i - 1] if i > 0 else 0
            next_idx = star_idxs[i] if i < len(star_idxs) else len(lines)

            block_lines = lines[prev_idx + 1 : next_idx]

            # 检查CSF块长度是否为3的倍数
            if len(block_lines) % 3 != 0:
                raise ValueError(
                    f"CSF block length must be a multiple of 3, got {len(block_lines)}"
                )

            # 将CSF块分成每3行一组
            block_csfs = [block_lines[i : i + 3] for i in range(0, len(block_lines), 3)]

            CSFs_block_data.append(block_csfs)
            CSFs_block_length.append(len(block_csfs))

        # 处理最后一个块（最后一个星号后到文件结尾）
        last_star_idx = star_idxs[-1] if star_idxs else 0
        last_block_lines = lines[last_star_idx + 1 :]

        if len(last_block_lines) % 3 != 0:
            raise ValueError(
                f"Last CSF block length must be a multiple of 3, got {len(last_block_lines)}"
            )

        last_block_csfs = [
            last_block_lines[i : i + 3] for i in range(0, len(last_block_lines), 3)
        ]

        CSFs_block_data.append(last_block_csfs)
        CSFs_block_length.append(len(last_block_csfs))

        # 确定宇称
        parity_set = set(CSFs_block_parity)
        parity = list(parity_set)[0] if len(parity_set) == 1 else ""

        block_num = len(CSFs_block_data)

        return CSFs(
            subshell_info_raw=subshell_info_raw,
            CSFs_block_j_value=CSFs_block_j_value,
            parity=parity,
            CSFs_block_data=CSFs_block_data,
            CSFs_block_length=CSFs_block_length,
            block_num=block_num,
        )

    def get_block_data(self, block_idx: int) -> list[list[str]]:
        """获取指定块的数据

        Args:
            block_idx: 块索引（0-based）

        Returns:
            CSF块数据列表

        Raises:
            IndexError: 块索引超出范围
        """
        csfs = self.load()

        if block_idx < 0 or block_idx >= csfs.block_num:
            raise IndexError(
                f"Block index {block_idx} out of range [0, {csfs.block_num})"
            )

        return csfs.CSFs_block_data[block_idx]

    def get_total_csfs_count(self) -> int:
        """获取总CSF数量

        Returns:
            所有块中的CSF总数
        """
        csfs = self.load()
        return sum(csfs.CSFs_block_length)

    def get_blocks_count(self) -> int:
        """获取块数量

        Returns:
            CSF块总数
        """
        csfs = self.load()
        return csfs.block_num

    def get_peel_subshells(self) -> list[str]:
        """获取电子层轨道列表

        Returns:
            电子层轨道名称列表，已去除空白
        """
        csfs = self.load()
        peel_line = csfs.subshell_info_raw[-1].strip()

        return [s.strip() for s in peel_line.split() if s.strip()]
