# -*- encoding: utf-8 -*-
"""
@Id :processing_data_load.py
@date :2025/06/16 16:30:36
@author :YenochQin (秦毅)
"""

from pathlib import Path

import gzip
import pickle
from typing import Any
import rtoml

import numpy as np
import pandas as pd
import polars as pl
import h5py

from ..utils.data_modules import CSFs
from .ml_cal_config_module import MLCalConfig


def load_csf_metadata(filepath: str | Path) -> dict:
    # 转换为Path对象
    filepath = Path(filepath)

    with open(filepath, "rb") as f:
        return pickle.load(f)


#######################################################################


def load_csfs_binary(filepath: str | Path) -> CSFs:
    filepath = Path(filepath)

    # 检查文件路径是否已经有正确的后缀
    if not str(filepath).endswith(".pkl.gz"):
        filepath = filepath.with_suffix(".pkl.gz")

    with gzip.open(filepath, "rb") as f:
        data = pickle.load(f)

    return CSFs(
        subshell_info_raw=data["metadata"]["subshell_info_raw"],
        CSFs_block_j_value=data["metadata"]["CSFs_block_j_value"],
        parity=data["metadata"]["parity"],
        CSFs_block_data=data["block_data"],  # 原始嵌套结构
        CSFs_block_length=data["metadata"]["CSFs_block_length"],
        block_num=data["metadata"]["block_num"],
    )


#######################################################################


def pkl_loader(load_csfs_idx_file_path) -> dict:
    """
    加载CSF索引文件（pickle格式）。

    Args:
        load_csfs_idx_file_path: 索引文件路径

    Returns:
        dict: 块索引到CSF索引列表/数组的映射
        读取的pkl文件中可能会有ci系数
    Raises:
        TypeError: 如果加载的数据不是正确格式
        FileNotFoundError: 如果文件不存在
    """

    file_path = Path(load_csfs_idx_file_path)

    # 自动添加.pkl扩展名（如果没有）
    if not file_path.suffix:
        file_path = file_path.with_suffix(".pkl")

    if not file_path.exists():
        raise FileNotFoundError(f"CSF索引文件不存在: {file_path}")

    # 直接加载pickle文件
    with open(file_path, "rb") as f:
        blocks_csfs_idx = pickle.load(f)

    # 类型检查和转换
    if not isinstance(blocks_csfs_idx, dict):
        raise TypeError(f"Expected dict, got {type(blocks_csfs_idx)}")

    return blocks_csfs_idx


#######################################################################


def load_large_hash(file_path: str | Path) -> dict[int, dict[str, int]]:
    """从文件加载预计算的哈希映射"""
    # 转换为Path对象
    file_path = Path(file_path)

    with open(file_path, "rb") as f:
        return pickle.load(f)


#######################################################################
def load_config(config_path: str | Path) -> MLCalConfig:
    """加载TOML配置文件并进行类型转换和数据处理

    使用Pydantic进行类型验证和转换，支持嵌套访问和动态属性。

    Args:
        config_path: TOML配置文件路径

    Returns:
        MLCalConfig: Pydantic配置模型，支持点号访问属性

    Raises:
        TypeError: 如果TOML顶层不是表
        ValueError: 如果配置验证失败（缺少必需字段、数值范围错误等）
    """
    # 转换为Path对象
    config_path = Path(config_path)

    # 使用 rtoml 读取TOML文件
    raw = rtoml.load(config_path)

    # 运行时校验 + 静态窄化：确保顶层是 dict
    if not isinstance(raw, dict):
        raise TypeError("Top-level TOML must be a table")

    # 使用Pydantic的model_validate进行类型转换和验证
    return MLCalConfig.model_validate(raw)


#######################################################################


def load_descriptors(
    load_path: str | Path,
    file_format: str | None = None,
    use_cpp: bool = False,
) -> np.ndarray | None:
    """
    加载描述符数组

    Args:
        load_path ( str | Path): 加载路径（可含或不含扩展名）
        file_format (str | None): 文件格式，如果为None则自动推断
        use_cpp (bool): 是否使用C++生成的HDF5文件

    Returns:
        np.ndarray | None: 描述符数组，加载失败返回None

    Example:
        >>> descriptors = load_descriptors('output/csf_descriptors.npy')
        >>> descriptors = load_descriptors(Path('output/csf_descriptors.npy'))
        >>> descriptors = load_descriptors('output/csf_descriptors', use_cpp=True)
    """

    # 转换为Path对象
    load_path = Path(load_path)

    # 如果使用C++ HDF5文件
    if use_cpp:
        hdf5_path = load_path.with_suffix(".h5")
        if hdf5_path.exists():
            try:
                with h5py.File(hdf5_path, "r") as f:
                    if "descriptors" in f:
                        # 检查descriptors是否是一个数据集，而不是数据类型
                        descriptors_obj = f["descriptors"]
                        if isinstance(descriptors_obj, h5py.Dataset):
                            descriptors = descriptors_obj[:]
                            print(f"Descriptors loaded from HDF5: {hdf5_path}")
                            return descriptors
                        else:
                            print(f"Error: 'descriptors' is not a dataset but a {type(descriptors_obj)}")
                    else:
                        print(f"Error: 'descriptors' dataset not found in {hdf5_path}")
            except Exception as e:
                print(f"Error loading HDF5 file {hdf5_path}: {str(e)}")
        else:
            print(f"Error: HDF5 file not found: {hdf5_path}")

    # 原有的文件格式支持
    if file_format is None:
        if load_path.suffix == ".npy":
            file_format = "npy"
            load_path = load_path.with_suffix("")  # 移除扩展名
        elif load_path.suffix == ".csv":
            file_format = "csv"
            load_path = load_path.with_suffix("")
        elif load_path.suffix == ".pkl":
            file_format = "pkl"
            load_path = load_path.with_suffix("")
        else:
            # 尝试自动检测（使用新的文件名格式）
            if (load_path.parent / f"{load_path.name}_descriptors.npy").exists():
                file_format = "npy"
            elif (load_path.parent / f"{load_path.name}_descriptors.csv").exists():
                file_format = "csv"
            elif (load_path.parent / f"{load_path.name}_descriptors.pkl").exists():
                file_format = "pkl"
            else:
                print(f"Error: Cannot find file with path: {load_path}")
                return None

    try:
        if file_format.lower() == "npy":
            file_path = load_path.parent / f"{load_path.name}_descriptors.npy"
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return None
            descriptors = np.load(file_path)
            print(f"Descriptors loaded from: {file_path}")
            return descriptors

        elif file_format.lower() == "csv":
            file_path = load_path.parent / f"{load_path.name}_descriptors.csv"
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return None
            df = pd.read_csv(file_path)
            descriptors = df.values
            print(f"Descriptors loaded from: {file_path}")
            return descriptors

        elif file_format.lower() == "pkl":
            file_path = load_path.parent / f"{load_path.name}_descriptors.pkl"
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return None
            with open(file_path, "rb") as f:
                descriptors = pickle.load(f)
            print(f"Descriptors loaded from: {file_path}")
            return descriptors

        else:
            print(f"Error: Unsupported file format: {file_format}")
            return None

    except Exception as e:
        print(f"Error loading descriptors: {str(e)}")
        return None


def load_descriptors_with_multi_block(
    load_path: str | Path,
    file_format: str | None = None,
    use_cpp: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    加载带标签的描述符数组

    Args:
        load_path (str | Path): 加载路径（不含扩展名）
        file_format (str | None): 文件格式，如果为None则自动推断
        use_cpp (bool): 是否使用C++生成的HDF5文件

    Returns:
        tuple[np.ndarray, np.ndarray] | None: (描述符数组, 标签数组)，加载失败返回None

    Example:
        >>> descriptors, labels = load_descriptors_with_block_idxs('ml_data/features')
        >>> descriptors, labels = load_descriptors_with_block_idxs(Path('ml_data/features'))
        >>> descriptors, labels = load_descriptors_with_block_idxs('ml_data/features', use_cpp=True)
    """

    # 转换为Path对象
    load_path = Path(load_path)

    # 如果使用C++ HDF5文件
    if use_cpp:
        hdf5_path = load_path.with_suffix(".h5")
        if hdf5_path.exists():
            try:
                with h5py.File(hdf5_path, "r") as f:
                    descriptors = np.array([])
                    labels = np.array([])

                    # 检查并加载descriptors
                    if "descriptors" in f:
                        desc_obj = f["descriptors"]
                        if isinstance(desc_obj, h5py.Dataset):
                            descriptors = desc_obj[:]
                        else:
                            print(f"Warning: 'descriptors' is not a dataset but a {type(desc_obj)}")

                    # 检查并加载labels
                    if "labels" in f:
                        labels_obj = f["labels"]
                        if isinstance(labels_obj, h5py.Dataset):
                            labels = labels_obj[:]
                        else:
                            print(f"Warning: 'labels' is not a dataset but a {type(labels_obj)}")

                    if descriptors is not None:
                        print(f"Descriptors loaded from HDF5: {hdf5_path}")
                        if labels is not None:
                            print(f"Labels loaded from HDF5: {hdf5_path}")
                        return descriptors, labels
                    else:
                        print(f"Error: 'descriptors' dataset not found or invalid in {hdf5_path}")
                        return None
            except Exception as e:
                print(f"Error loading HDF5 file {hdf5_path}: {str(e)}")
                return None
        else:
            print(f"Error: HDF5 file not found: {hdf5_path}")

    # 原有的文件格式支持
    if file_format is None:
        if (
            load_path.parent / f"{load_path.name}_descriptors_block_idxs.csv"
        ).exists():
            file_format = "csv"
        elif (load_path.parent / f"{load_path.name}_descriptors.npy").exists() and (
            load_path.parent / f"{load_path.name}_descriptors_block_idxs.npy"
        ).exists():
            file_format = "npy"
        elif (
            load_path.parent / f"{load_path.name}_descriptors_block_idxs.pkl"
        ).exists():
            file_format = "pkl"
        else:
            print(f"Error: Cannot find files with path: {load_path}")
            return None

    try:
        if file_format.lower() == "csv":
            file_path = (
                load_path.parent / f"{load_path.name}_descriptors_block_idxs.csv"
            )
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return None

            df = pd.read_csv(file_path)
            # 最后一列是标签，其余是描述符
            descriptors = df.iloc[:, :-1].to_numpy()
            labels = df.iloc[:, -1].to_numpy()
            print(f"Descriptors and labels loaded from: {file_path}")
            return descriptors, labels

        elif file_format.lower() == "npy":
            data_path = load_path.parent / f"{load_path.name}_descriptors.npy"
            labels_path = (
                load_path.parent / f"{load_path.name}_descriptors_block_idxs.npy"
            )

            if not data_path.exists():
                print(f"Error: Data file not found: {data_path}")
                return None
            if not labels_path.exists():
                print(f"Error: Labels file not found: {labels_path}")
                return None

            descriptors = np.load(data_path)
            labels = np.load(labels_path)
            print(f"Descriptors loaded from: {data_path}")
            print(f"Labels loaded from: {labels_path}")
            return descriptors, labels

        elif file_format.lower() == "pkl":
            file_path = (
                load_path.parent / f"{load_path.name}_descriptors_block_idxs.pkl"
            )
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return None

            with open(file_path, "rb") as f:
                data_dict = pickle.load(f)

            if "descriptors" not in data_dict or "labels" not in data_dict:
                print(f"Error: Invalid data format in {file_path}")
                return None

            descriptors = data_dict["descriptors"]
            labels = data_dict["labels"]
            print(f"Descriptors and labels loaded from: {file_path}")
            return descriptors, labels

        else:
            print(f"Error: Unsupported file format: {file_format}")
            return None

    except Exception as e:
        print(f"Error loading descriptors with labels: {str(e)}")
        return None


#######################################################################
# Rust Parquet 描述符加载器 - 惰性加载版本（使用 scan + 按索引提取）
#######################################################################

def load_descriptors_by_scan(
    scan_parquet: pl.LazyFrame,
    idxs: np.ndarray,
    batch_size: int = 3_000_000,
) -> np.ndarray:
    """
    使用 Polars scan 模式按索引批量加载描述符（惰性加载，仅加载指定索引的数据）

    适用于大规模描述符文件（上亿级），通过 scan + slice 只提取需要的索引段，
    避免一次性加载全部数据到内存。

    Args:
        parquet_path: parquet 文件路径
        idxs: 需要加载的 CSF 索引数组
        batch_size: 每批处理的索引数量（默认3000万，适合训练数据规模）

    Returns:
        np.ndarray: 加载的描述符数组，形状为 (len(idxs), n_features)

    Example:
        >>> # 只加载索引为 [0, 100, 1000, ..., 100000] 的描述符
        >>> idxs = np.array([0, 100, 1000, 100000])
        >>> descriptors = load_descriptors_by_scan("data.parquet", idxs)
        >>> print(descriptors.shape)  # (4, n_features)
    """

    if len(idxs) == 0:
        raise ValueError("索引数组不能为空")

    # 对索引排序，便于使用连续的 slice 读取
    sorted_idxs = np.sort(idxs)

    # 记录原始索引用于恢复顺序
    original_order = np.argsort(np.argsort(idxs))

    result = []
    total_batches = (len(sorted_idxs) + batch_size - 1) // batch_size

    for i in range(0, len(sorted_idxs), batch_size):
        batch_idxs = sorted_idxs[i:i + batch_size]

        # scan + slice：只读取需要的数据段
        start = batch_idxs[0]
        end = batch_idxs[-1]
        length = end - start + 1

        # 惰性扫描 + 切片收集
        batch_df = scan_parquet.slice(start, length).collect()
        batch_array = batch_df.to_numpy()

        # 从 batch_array 中提取实际需要的索引
        # batch_idxs 是相对于整个文件的索引，需要转换为相对于 start 的索引
        relative_idxs = batch_idxs - start
        extracted = batch_array[relative_idxs]

        result.append(extracted)

        if (i // batch_size + 1) % 10 == 0 or (i // batch_size + 1) == total_batches:
            print(f"已加载 {i // batch_size + 1}/{total_batches} 批")

    # 合并所有批次并恢复原始顺序
    merged = np.vstack(result)[original_order]

    print(f"加载完成: {len(idxs)} 个描述符, 形状: {merged.shape}")
    return merged


def scan_descriptors_polars(
    parquet_path: str | Path
) -> tuple[pl.LazyFrame, dict]:
    """
    使用 Polars 加载 parquet 描述符文件

    Args:
        parquet_path: parquet 文件路径

    Returns:
        pl.LazyFrame : 完整的描述符数组
        dict: parquet 文件元数据

    Example:
        >>> descriptors, meta = load_descriptors_polars("descriptors.parquet")
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pl.scan_parquet(parquet_path)

    meta: dict[Any, Any] = _get_parquet_metadata(df)

    return df, meta


def _get_parquet_metadata(scan_parquet: pl.LazyFrame) -> dict:
    """
    获取 parquet 文件的元数据（无需加载全部数据）

    Args:
        scan_parquet: polars scan_parquet 对象

    Returns:
        dict: 包含 n_rows, n_columns, columns, dtypes 等信息

    """

    # 获取总行数
    n_rows = scan_parquet.select(pl.len()).collect().item()

    # 获取列信息（不读取数据）
    schema = scan_parquet.collect_schema()

    return {
        "n_rows": n_rows,
        "n_columns": len(schema),
        "columns": list(schema.names()),
        "dtypes": {name: str(dtype) for name, dtype in schema.items()},
    }

