# -*- encoding: utf-8 -*-
"""
@Id :binary_file_loader.py
@date :2026/01/19
@author :YenochQin (秦毅)
"""

import struct
import numpy as np
from typing import Any, BinaryIO, Tuple, Union
from numpy.typing import NDArray
from .base_loader import BaseLoader


class BinaryFileLoader(BaseLoader[Any]):
    """二进制文件加载器基类

    用于读取 GRASP2018 生成的二进制文件。

    Fortran 二进制文件格式说明：
    - 每个 Fortran 记录前后都有 4 字节的长度标记
    - 对于单类型数据数组，使用 read_fortran_record()
    - 对于混合类型数据（如 read (3) a, b, c），使用 read_mixed_fortran_record()
    """

    def read_fortran_record(
        self, file: BinaryIO, dtype: Any, count: int = 1
    ) -> NDArray[np.number[Any]]:
        """读取 Fortran 格式的二进制记录（单类型数组）

        适用于读取 Fortran 语句如：
        - read (3) (data(i), i=1, n)

        Args:
            file: 文件对象
            dtype: 数据类型 (np.dtype 或字符串如 'int32', 'float64')
            count: 读取数量

        Returns:
            读取的数据数组

        Raises:
            ValueError: 文件格式不正确

        Example:
            >>> # 读取 Fortran: read (3) (ival(i), i=1, 10)
            >>> data = read_fortran_record(file, 'int32', 10)
            >>> # 读取 Fortran: read (3) (dval(i), i=1, 5)
            >>> data = read_fortran_record(file, 'float64', 5)
        """
        # 读取记录长度标记（4字节）
        record_len_bytes = file.read(4)
        if len(record_len_bytes) < 4:
            raise ValueError("Unexpected end of file while reading record length")

        record_len = struct.unpack('i', record_len_bytes)[0]

        # 读取实际数据
        data = np.fromfile(file, dtype=dtype, count=count)

        # 读取记录结束标记（4字节）
        record_len_end_bytes = file.read(4)
        if len(record_len_end_bytes) < 4:
            raise ValueError("Unexpected end of file while reading record end marker")

        record_len_end = struct.unpack('i', record_len_end_bytes)[0]

        # 验证记录长度匹配
        if record_len != record_len_end:
            raise ValueError(
                f"Record length mismatch: start={record_len}, end={record_len_end}"
            )

        return data

    def read_mixed_fortran_record(
        self, file: BinaryIO, field_specs: list[Union[str, Tuple[str, int]]]
    ) -> list[Any]:
        """读取 Fortran 格式的二进制记录（混合类型）

        适用于读取 Fortran 语句如：
        - read (3) nn, laky, energy, npts
        - read (3) a0, (pg(i), i=1, n), (qg(i), i=1, n)

        每个记录前后都有 4 字节的长度标记。

        Args:
            file: 文件对象
            field_specs: 字段规格列表，每个规格可以是：
                - 'i' 或 'int32': 1 个 int32
                - 'd' 或 'float64': 1 个 float64
                - 'i[n]' 或 'int32[n]': n 个 int32
                - 'd[n]' 或 'float64[n]': n 个 float64
                - ('i', n): n 个 int32（元组形式，n 为变量）
                - ('d', n): n 个 float64（元组形式，n 为变量）

        Returns:
            读取的数据列表，按 field_specs 顺序返回

        Raises:
            ValueError: 文件格式不正确

        Example:
            >>> # 读取 Fortran: read (3) nn, laky, energy, npts
            >>> nn, laky, energy, npts = read_mixed_fortran_record(file, ['i', 'i', 'd', 'i'])
            >>> # 读取 Fortran: read (3) a0, (pg(i), i=1, npts), (qg(i), i=1, npts)
            >>> a0, pg, qg = read_mixed_fortran_record(file, [('d', 1), ('d', npts), ('d', npts)])
        """
        # 解析字段规格并计算数据总大小
        data_size = 0
        parsed_specs = []

        for spec in field_specs:
            if isinstance(spec, tuple):
                # 元组形式：('i', n) 或 ('d', n)
                fmt, count = spec
            elif '[' in spec:
                # 字符串形式：'i[n]' 或 'd[n]'
                fmt, count_str = spec.split('[')
                count = int(count_str.rstrip(']'))
            else:
                # 简单形式：'i' 或 'd'
                fmt, count = spec, 1

            # 确定每种类型的字节大小
            if fmt in ('i', 'int32'):
                bytes_per_item = 4
                parsed_fmt = 'i'
            elif fmt in ('d', 'float64'):
                bytes_per_item = 8
                parsed_fmt = 'd'
            else:
                raise ValueError(f"Unsupported format: {fmt}")

            data_size += bytes_per_item * count
            parsed_specs.append((parsed_fmt, count))

        # 读取记录长度标记（4字节）
        record_len_bytes = file.read(4)
        if len(record_len_bytes) < 4:
            raise ValueError("Unexpected end of file while reading record length")

        record_len = struct.unpack('i', record_len_bytes)[0]

        # 验证记录长度
        if record_len != data_size:
            raise ValueError(
                f"Record length mismatch: expected {data_size} bytes, got {record_len} bytes"
            )

        # 读取数据
        result = []
        for fmt, count in parsed_specs:
            if fmt == 'i':
                if count == 1:
                    data = struct.unpack('i', file.read(4))[0]
                else:
                    data = list(struct.unpack(f'{count}i', file.read(4 * count)))
            elif fmt == 'd':
                if count == 1:
                    data = struct.unpack('d', file.read(8))[0]
                else:
                    data = list(struct.unpack(f'{count}d', file.read(8 * count)))
            else:
                raise ValueError(f"Unsupported format: {fmt}")
            result.append(data)

        # 读取记录结束标记（4字节）
        record_len_end_bytes = file.read(4)
        if len(record_len_end_bytes) < 4:
            raise ValueError("Unexpected end of file while reading record end marker")

        record_len_end = struct.unpack('i', record_len_end_bytes)[0]

        # 验证记录结束标记
        if record_len_end != record_len:
            raise ValueError(
                f"Record end length mismatch: start={record_len}, end={record_len_end}"
            )

        return result

    def write_fortran_record(self, file: BinaryIO, data: NDArray[np.number[Any]]) -> None:
        """写入 Fortran 格式的二进制记录

        Args:
            file: 文件对象
            data: 要写入的数据

        Raises:
            ValueError: 数据格式不正确
        """
        # 计算数据大小（字节）
        data_size = data.nbytes

        # 写入记录长度标记（4字节）
        file.write(struct.pack('i', data_size))

        # 写入数据
        data.tofile(file)

        # 写入记录结束标记（4字节）
        file.write(struct.pack('i', data_size))
