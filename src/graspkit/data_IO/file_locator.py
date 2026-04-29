# -*- encoding: utf-8 -*-
import re
from pathlib import Path


class FileLocator:
    """统一文件定位器，处理文件路径解析和glob匹配

    该类负责根据文件模式（glob通配符）定位文件，
    支持递归搜索和清晰错误处理。
    """

    def __init__(self, base_dir: str | Path):
        """初始化文件定位器

        Args:
            base_dir: 基础目录，可以是字符串或Path对象

        Raises:
            ValueError: 目录不存在
        """
        self.base_dir: Path = Path(base_dir)

        if not self.base_dir.exists():
            raise ValueError(f"Base directory does not exist: {self.base_dir}")

        if not self.base_dir.is_dir():
            raise ValueError(f"Path is not a directory: {self.base_dir}")

    def locate_files(self, pattern: str, recursive: bool = False) -> list[Path]:
        """定位匹配的文件

        Args:
            pattern: 文件模式（支持glob通配符），如 "*.txt" 或 "*.{c,m}"
            recursive: 是否递归搜索子目录

        Returns:
            匹配的文件路径列表（按修改时间排序）

        Raises:
            FileNotFoundError: 没有找到匹配文件
        """
        if recursive:
            files = list(self.base_dir.rglob(pattern))
        else:
            files = list(self.base_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No files matching pattern '{pattern}' found in {self.base_dir}"
            )

        # 按修改时间排序（新文件在前）
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        return files

    def locate_file(self, pattern: str, recursive: bool = False) -> Path:
        """定位单个文件

        Args:
            pattern: 文件模式
            recursive: 是否递归搜索

        Returns:
            找到的文件路径

        Raises:
            FileNotFoundError: 没有找到文件或找到多个
            ValueError: 找到多个匹配文件
        """
        files = self.locate_files(pattern, recursive)

        if len(files) == 1:
            return files[0]
        elif len(files) > 1:
            raise ValueError(f"Multiple files found matching '{pattern}': {files}")
        else:
            raise FileNotFoundError(
                f"No file found matching pattern '{pattern}' in {self.base_dir}"
            )

    def locate_by_regex(self, pattern: str, recursive: bool = False) -> list[Path]:
        """使用正则表达式定位文件

        Args:
            pattern: 正则表达式模式
            recursive: 是否递归搜索

        Returns:
            匹配的文件路径列表

        Raises:
            FileNotFoundError: 没有找到匹配文件
        """
        compiled_pattern = re.compile(pattern)

        if recursive:
            all_files = [f for f in self.base_dir.rglob("*") if f.is_file()]
            files = [f for f in all_files if compiled_pattern.search(f.name)]
        else:
            all_files = [f for f in self.base_dir.glob("*") if f.is_file()]
            files = [f for f in all_files if compiled_pattern.search(f.name)]

        if not files:
            raise FileNotFoundError(
                f"No files matching regex pattern '{pattern}' found in {self.base_dir}"
            )

        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return files

    def get_directory(self) -> Path:
        """获取基础目录

        Returns:
            基础目录路径
        """
        return self.base_dir

    def exists(self) -> bool:
        """检查基础目录是否存在

        Returns:
            目录是否存在
        """
        return self.base_dir.exists()

    def is_directory(self) -> bool:
        """检查路径是否为目录

        Returns:
            是否为目录
        """
        return self.base_dir.is_dir()
