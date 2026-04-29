# -*- encoding: utf-8 -*-
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


class BaseLoader(ABC, Generic[T]):
    """文件加载器基类

    所有专用文件加载器的基类，提供统一的接口和错误处理。
    使用泛型 T 指定 load() 方法的返回类型。
    """

    def __init__(self, file_path: str | Path):
        """初始化加载器

        Args:
            file_path: 文件路径（绝对路径或相对于当前工作目录）

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件路径无效
        """
        self.file_path: Path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    @abstractmethod
    def load(self) -> T:
        """加载数据（抽象方法）

        Returns:
            加载的数据对象，类型由子类决定

        Raises:
            ValueError: 数据格式无效
            IOError: 读取错误
        """
        pass

    def get_file_path(self) -> Path:
        """获取文件路径

        Returns:
            文件路径
        """
        return self.file_path

    def get_file_name(self) -> str:
        """获取文件名

        Returns:
            文件名（含扩展名）
        """
        return self.file_path.name

    def get_file_stem(self) -> str:
        """获取文件名（不含扩展名）

        Returns:
            文件名（不含扩展名）
        """
        return self.file_path.stem

    def get_file_extension(self) -> str:
        """获取文件扩展名

        Returns:
            文件扩展名（含点）
        """
        return self.file_path.suffix

    def file_exists(self) -> bool:
        """检查文件是否存在

        Returns:
            文件是否存在
        """
        return self.file_path.exists()

    def file_size(self) -> int:
        """获取文件大小

        Returns:
            文件大小（字节数）
        """
        return self.file_path.stat().st_size

    def get_modification_time(self) -> float:
        """获取文件修改时间

        Returns:
            文件修改时间戳
        """
        return self.file_path.stat().st_mtime

    def validate_file(self, expected_extensions: list[str] | None = None) -> None:
        """验证文件扩展名

        Args:
            expected_extensions: 期望的文件扩展名列表

        Raises:
            ValueError: 文件扩展名不符合预期
        """
        if expected_extensions is None:
            return

        ext = self.get_file_extension().lower()
        if ext not in expected_extensions:
            raise ValueError(f"Invalid file extension: {ext}. ")
