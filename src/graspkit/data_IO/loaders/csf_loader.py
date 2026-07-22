# -*- encoding: utf-8 -*-
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

        # 查找 "CSF(s):" 行来确定CSF数据的起始位置
        csf_start_idx = 4  # 默认从第5行开始
        for idx, line in enumerate(lines):
            if "CSF(s):" in line or "CSFs:" in line:
                csf_start_idx = idx + 1
                break

        subshell_info_raw = lines[:4]

        # 查找 block 分隔行
        star_idxs = [idx for idx, line in enumerate(lines) if line.strip() == "*"]

        CSFs_block_j_value: list[str] = []
        CSFs_block_parity: list[str] = []
        CSFs_block_data: list[list[list[str]]] = []
        CSFs_block_length: list[int] = []

        block_starts = [csf_start_idx, *(idx + 1 for idx in star_idxs)]
        block_ends = [*star_idxs, len(lines)]
        for block_idx, (start_idx, end_idx) in enumerate(
            zip(block_starts, block_ends, strict=True)
        ):
            block_lines = lines[start_idx:end_idx]
            if not block_lines:
                raise ValueError(f"CSF block {block_idx} is empty")
            if len(block_lines) % 3 != 0:
                raise ValueError(
                    f"CSF block {block_idx} length must be a multiple of 3, "
                    f"got {len(block_lines)}"
                )

            j_parity_line = block_lines[-1].strip()
            match = re.search(
                r"(?P<j>\d+(?:/\d+)?)?\s*(?P<parity>[+-])$",
                j_parity_line,
            )
            if not match:
                raise ValueError(f"Invalid J/parity format: {j_parity_line}")

            block_csfs = [
                block_lines[index : index + 3]
                for index in range(0, len(block_lines), 3)
            ]
            CSFs_block_j_value.append(match.group("j") or "")
            CSFs_block_parity.append(match.group("parity"))
            CSFs_block_data.append(block_csfs)
            CSFs_block_length.append(len(block_csfs))

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
