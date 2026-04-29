# -*- encoding: utf-8 -*-
from pathlib import Path
from typing import cast
import rtoml

import numpy as np
from numpy.typing import NDArray
import polars as pl

from graspkit_config import MLCalConfig


def csfs_idxs_ci_loader(
    csfs_id_ci_file_path: str | Path,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    加载CSF索引文件（npz格式），并校验数据结构。

    与 csfs_idxs_ci_storage 对应，读取其保存的 idxs / ci_squared 两个数组。

    Args:
        csfs_id_ci_file_path: 索引文件路径（无扩展名时自动补 .npz）

    Returns:
        ``(csfs_idx, csfs_ci_squared)``，其中 ``csfs_idx`` 是一维 CSF
        整数索引数组，``csfs_ci_squared`` 是形状为
        ``(n_levels, n_csfs)`` 的 CI 系数平方值数组。

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: npz 缺少必要键，或数组维度/长度不一致
        TypeError: 数组 dtype 不符合要求
    """
    file_path = Path(csfs_id_ci_file_path)
    if not file_path.suffix:
        file_path = file_path.with_suffix(".npz")

    if not file_path.exists():
        raise FileNotFoundError(f"CSF索引文件不存在: {file_path}")

    with np.load(file=file_path) as data_container:
        missing = {"idxs", "ci_squared"} - set(data_container.files)
        if missing:
            raise ValueError(f"npz 文件缺少必要键: {missing}")

        idxs = data_container["idxs"]
        ci_squared = data_container["ci_squared"]

    # dtype 校验
    if not np.issubdtype(idxs.dtype, np.integer):
        raise TypeError(f"'idxs' 应为整数 dtype，实际为 {idxs.dtype}")
    if not np.issubdtype(ci_squared.dtype, np.floating):
        raise TypeError(f"'ci_squared' 应为浮点 dtype，实际为 {ci_squared.dtype}")

    # 维度与长度校验
    if idxs.ndim != 1:
        raise ValueError(f"'idxs' 应为一维数组，实际维度为 {idxs.ndim}")
    if ci_squared.ndim != 2:
        raise ValueError(f"'ci_squared' 应为二维数组 (n_levels, n_csfs)，实际维度为 {ci_squared.ndim}")
    if idxs.shape[0] != ci_squared.shape[1]:
        raise ValueError(
            f"'idxs' 与 'ci_squared' 的 CSF 数不一致: "
            f"idxs.shape[0]={idxs.shape[0]} != ci_squared.shape[1]={ci_squared.shape[1]}"
        )

    return cast(NDArray[np.int64], idxs), cast(NDArray[np.float64], ci_squared)


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

    # 使用Pydantic的model_validate进行类型转换和验证
    return MLCalConfig.model_validate(raw)


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

    result: list[np.ndarray] = []
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
) -> tuple[pl.LazyFrame, dict[str, int]]:
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

    meta: dict[str, int] = _get_parquet_metadata(df)

    return df, meta


def _get_parquet_metadata(scan_parquet: pl.LazyFrame) -> dict[str, int]:
    """
    获取 parquet 文件的元数据（无需加载全部数据）

    Args:
        scan_parquet: polars scan_parquet 对象

    Returns:
        dict: 包含 n_rows, n_columns, columns信息

    """

    # 获取总行数
    n_rows = scan_parquet.select(pl.len()).collect().item()

    # 获取列信息（不读取数据）
    schema = scan_parquet.collect_schema()

    return {
        "n_rows": n_rows,
        "n_columns": len(schema),
    }
