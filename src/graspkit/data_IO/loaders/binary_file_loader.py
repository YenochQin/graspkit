# -*- encoding: utf-8 -*-
import struct
from abc import ABC
from typing import IO, TypeVar

import numpy as np
from numpy.typing import NDArray

from .base_loader import BaseLoader

_T = TypeVar("_T", bound=np.generic)
class BinaryFileLoader(BaseLoader[object], ABC):
    """二进制文件加载器基类

    用于读取 GRASP2018 生成的二进制文件。

    Fortran 二进制文件格式说明：
    - 每个 Fortran 记录前后都有 4 字节的长度标记
    - 对于单类型数据数组，使用 read_fortran_record()
    - 对于混合类型数据（如 read (3) a, b, c），使用 read_mixed_fortran_record()
    """

    def read_fortran_record(
        self,
        file: IO[bytes],
        dtype: type[_T] | np.dtype[_T] | str,
        count: int = 1
    ) -> NDArray[_T]:
        """读取 Fortran 格式的二进制记录（单类型数组）

        适用于读取 Fortran 语句如：
        - read (3) (data(i), i=1, n)

        Args:
            file: 文件对象
            dtype: 数据类型 (np.dtype 或字符串如 'int32', 'float64', 'S1')
            count: 读取数量

        Returns:
            读取的数据数组（数值类型或字节类型）

        Raises:
            ValueError: 文件格式不正确

        Example:
            >>> # 读取 Fortran: read (3) (ival(i), i=1, 10)
            >>> data = read_fortran_record(file, 'int32', 10)
            >>> # 读取 Fortran: read (3) (dval(i), i=1, 5)
            >>> data = read_fortran_record(file, 'float64', 5)
            >>> # 读取 Fortran: read (3) header（字符串）
            >>> header = read_fortran_record(file, 'S1', 6)
        """
        # 读取记录长度标记（4字节）
        record_len_bytes: bytes = file.read(4)
        if len(record_len_bytes) < 4:
            raise ValueError("Unexpected end of file while reading record length")

        record_len: int = struct.unpack("i", record_len_bytes)[0]

        # 读取实际数据
        data: NDArray[_T] = np.fromfile(file, dtype=dtype, count=count)

        # 读取记录结束标记（4字节）
        record_len_end_bytes: bytes = file.read(4)
        if len(record_len_end_bytes) < 4:
            raise ValueError("Unexpected end of file while reading record end marker")

        record_len_end: int = struct.unpack("i", record_len_end_bytes)[0]

        # 验证记录长度匹配
        if record_len != record_len_end:
            raise ValueError(
                f"Record length mismatch: start={record_len}, end={record_len_end}"
            )

        return data

    def read_mixed_scalars(
        self, file: IO[bytes], field_specs: list[str]
    ) -> list[int | float]:
        """读取 Fortran 格式的二进制记录（标量序列）

        适用于读取 Fortran 语句如：
        - read (3) nn, laky, energy, npts

        每个记录前后都有 4 字节的长度标记。

        Args:
            file: 文件对象
            field_specs: 字段规格列表，每个规格可以是：
                - 'i' 或 'int32': 1 个 int32
                - 'd' 或 'float64': 1 个 float64

        Returns:
            读取的标量值列表，按 field_specs 顺序返回

        Raises:
            ValueError: 文件格式不正确

        Example:
            >>> # 读取 Fortran: read (3) nn, laky, energy, npts
            >>> nn, laky, energy, npts = read_mixed_scalars(file, ['i', 'i', 'd', 'i'])
        """
        # 解析字段规格并计算数据总大小
        data_size = 0
        parsed_specs: list[str] = []

        for spec in field_specs:
            if spec in ("i", "int32"):
                bytes_per_item = 4
                parsed_specs.append("i")
            elif spec in ("d", "float64"):
                bytes_per_item = 8
                parsed_specs.append("d")
            else:
                raise ValueError(f"Unsupported format: {spec}")

            data_size += bytes_per_item

        # 读取记录长度标记（4字节）
        record_len_bytes = file.read(4)
        if len(record_len_bytes) < 4:
            raise ValueError("Unexpected end of file while reading record length")

        record_len: int = struct.unpack("i", record_len_bytes)[0]

        # 验证记录长度
        if record_len != data_size:
            raise ValueError(
                f"Record length mismatch: expected {data_size} bytes, got {record_len} bytes"
            )

        # 读取标量数据
        result: list[int | float] = []
        for fmt in parsed_specs:
            if fmt == "i":
                data = struct.unpack("i", file.read(4))[0]
            else:  # fmt == "d"
                data = struct.unpack("d", file.read(8))[0]
            result.append(data)

        # 读取记录结束标记（4字节）
        record_len_end_bytes = file.read(4)
        if len(record_len_end_bytes) < 4:
            raise ValueError("Unexpected end of file while reading record end marker")

        record_len_end = struct.unpack("i", record_len_end_bytes)[0]

        # 验证记录结束标记
        if record_len_end != record_len:
            raise ValueError(
                f"Record end length mismatch: start={record_len}, end={record_len_end}"
            )

        return result

    def read_mixed_arrays(
        self, file: IO[bytes], field_specs: list[tuple[str, int]]
    ) -> list[list[int] | list[float]]:
        """读取 Fortran 格式的二进制记录（数组序列）

        适用于读取 Fortran 语句如：
        - read (3) a0, (pg(i), i=1, n), (qg(i), i=1, n)

        每个记录前后都有 4 字节的长度标记。

        Args:
            file: 文件对象
            field_specs: 字段规格列表，每个规格是元组 (format, count)：
                - ('i', n): n 个 int32
                - ('d', n): n 个 float64

        Returns:
            读取的数组列表，按 field_specs 顺序返回

        Raises:
            ValueError: 文件格式不正确

        Example:
            >>> # 读取 Fortran: read (3) a0, (pg(i), i=1, npts), (qg(i), i=1, npts)
            >>> a0, pg, qg = read_mixed_arrays(file, [('d', 1), ('d', npts), ('d', npts)])
        """
        # 解析字段规格并计算数据总大小
        data_size = 0
        parsed_specs: list[tuple[str, int]] = []

        for fmt, count in field_specs:
            if fmt in ("i", "int32"):
                bytes_per_item = 4
                parsed_fmt = "i"
            elif fmt in ("d", "float64"):
                bytes_per_item = 8
                parsed_fmt = "d"
            else:
                raise ValueError(f"Unsupported format: {fmt}")

            data_size += bytes_per_item * count
            parsed_specs.append((parsed_fmt, count))

        # 读取记录长度标记（4字节）
        record_len_bytes = file.read(4)
        if len(record_len_bytes) < 4:
            raise ValueError("Unexpected end of file while reading record length")

        record_len: int = struct.unpack("i", record_len_bytes)[0]

        # 验证记录长度
        if record_len != data_size:
            raise ValueError(
                f"Record length mismatch: expected {data_size} bytes, got {record_len} bytes"
            )

        # 读取数组数据
        result: list[list[int] | list[float]] = []
        for fmt, count in parsed_specs:
            if fmt == "i":
                data = list(struct.unpack(f"{count}i", file.read(4 * count)))
            elif fmt == "d":
                data = list(struct.unpack(f"{count}d", file.read(8 * count)))
            else:
                raise ValueError(f"Unsupported format: {fmt}")
            result.append(data)

        # 读取记录结束标记（4字节）
        record_len_end_bytes: bytes = file.read(4)
        if len(record_len_end_bytes) < 4:
            raise ValueError("Unexpected end of file while reading record end marker")

        record_len_end = struct.unpack("i", record_len_end_bytes)[0]

        # 验证记录结束标记
        if record_len_end != record_len:
            raise ValueError(
                f"Record end length mismatch: start={record_len}, end={record_len_end}"
            )

        return result

    def write_mixed_scalars(
        self, file: IO[bytes], values: list[int | float]
    ) -> None:
        """写入 Fortran 格式的二进制记录（标量序列）

        适用于写入类似 Fortran 语句：
        - write (3) nn, laky, energy, npts

        每个记录前后都有 4 字节的长度标记。

        Args:
            file: 文件对象
            values: 要写入的标量值列表

        Example:
            >>> # 写入 Fortran: write (3) nn, laky, energy, npts
            >>> write_mixed_scalars(file, [nn, laky, energy, npts])
        """
        # 计算数据总大小（字节）
        data_size = 0
        for value in values:
            if isinstance(value, int):
                data_size += 4
            else:  # float
                data_size += 8

        # 写入记录长度标记（4字节）
        _ = file.write(struct.pack("i", data_size))

        # 写入标量数据
        for value in values:
            if isinstance(value, int):
                _ = file.write(struct.pack("i", value))
            else:  # float
                _ = file.write(struct.pack("d", value))

        # 写入记录结束标记（4字节）
        _ = file.write(struct.pack("i", data_size))

    def write_mixed_arrays(
        self, file: IO[bytes], arrays: list[list[int] | list[float]]
    ) -> None:
        """写入 Fortran 格式的二进制记录（数组序列）

        适用于写入类似 Fortran 语句：
        - write (3) a0, (pg(i), i=1, n), (qg(i), i=1, n)

        每个记录前后都有 4 字节的长度标记。

        Args:
            file: 文件对象
            arrays: 要写入的数组列表

        Example:
            >>> # 写入 Fortran: write (3) a0, (pg(i), i=1, npts), (qg(i), i=1, npts)
            >>> write_mixed_arrays(file, [a0, pg, qg])
        """
        # 计算数据总大小（字节）
        data_size = 0
        for array in arrays:
            if array and isinstance(array[0], int):
                data_size += 4 * len(array)
            else:  # float
                data_size += 8 * len(array)

        # 写入记录长度标记（4字节）
        _ = file.write(struct.pack("i", data_size))

        # 写入数组数据
        for array in arrays:
            if array and isinstance(array[0], int):
                for value in array:
                    _ = file.write(struct.pack("i", value))
            else:  # float
                for value in array:
                    _ = file.write(struct.pack("d", value))

        # 写入记录结束标记（4字节）
        _ = file.write(struct.pack("i", data_size))

    def write_fortran_record(
        self, file: IO[bytes], data: NDArray[np.number]
    ) -> None:
        """写入 Fortran 格式的二进制记录（单类型数组）

        适用于写入类似 Fortran 语句：
        - write (3) (data(i), i=1, n)

        每个记录前后都有 4 字节的长度标记。

        Args:
            file: 文件对象
            data: 要写入的数据

        Example:
            >>> # 写入 Fortran: write (3) (data(i), i=1, n)
            >>> write_fortran_record(file, data)
        """
        # 计算数据大小（字节）
        data_size: int = data.nbytes

        # 写入记录长度标记（4字节）
        _ = file.write(struct.pack("i", data_size))

        # 写入数据
        data.tofile(file)

        # 写入记录结束标记（4字节）
        _ = file.write(struct.pack("i", data_size))
