# -*- encoding: utf-8 -*-
"""Strict data-contract validation for CSF selection indices, headers, and paths.

Every check here fails fast with a locatable error instead of silently
truncating, filtering, or padding data. Callers must not catch these errors
to fall back to partial results.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def validate_selection_idxs(
    idxs: np.ndarray,
    *,
    row_count: int | None = None,
    allow_duplicates: bool = False,
    source: str | Path | None = None,
) -> NDArray[np.int64]:
    """Validate a selection index array under a strict data contract.

    Args:
        idxs: Candidate index array.
        row_count: If given, every index must fall in ``[0, row_count)``.
        allow_duplicates: If False (default), duplicate indices are rejected.
        source: Optional label (e.g. a file path) included in error messages.

    Returns:
        The indices as a one-dimensional ``int64`` array, in their original
        order.

    Raises:
        ValueError: If ``idxs`` is not a one-dimensional integer-kind array,
            or contains negative, out-of-range, or (unless allowed)
            duplicate values.
    """
    label = f" ({source})" if source is not None else ""
    array = np.asarray(idxs)
    if array.ndim != 1:
        raise ValueError(f"选择索引必须是一维数组{label}: ndim={array.ndim}")
    if array.dtype.kind not in ("i", "u"):
        raise ValueError(f"选择索引必须是整数 dtype{label}: 实际 dtype={array.dtype}")

    int_array = array.astype(np.int64, copy=False)
    if int_array.size == 0:
        return int_array

    min_value = int(int_array.min())
    if min_value < 0:
        raise ValueError(f"选择索引不能为负数{label}: min={min_value}")

    if row_count is not None:
        max_value = int(int_array.max())
        if max_value >= row_count:
            raise ValueError(
                f"选择索引越界{label}: max={max_value}, row_count={row_count}"
            )

    if not allow_duplicates:
        unique_count = int(np.unique(int_array).shape[0])
        if unique_count != int_array.shape[0]:
            raise ValueError(
                f"选择索引包含重复值{label}: "
                f"共 {int_array.shape[0]} 项，去重后 {unique_count} 项"
            )

    return int_array


def load_selection_idxs(
    idx_file: Path,
    *,
    row_count: int | None = None,
    allow_duplicates: bool = False,
) -> NDArray[np.int64]:
    """Load and strictly validate a ``.npy`` selection index file.

    Args:
        idx_file: Path to a ``.npy`` file containing a 1D integer index array.
        row_count: If given, every index must fall in ``[0, row_count)``.
        allow_duplicates: If False (default), duplicate indices are rejected.

    Returns:
        The indices as a one-dimensional ``int64`` array.

    Raises:
        FileNotFoundError: If ``idx_file`` does not exist.
        ValueError: If the loaded array fails :func:`validate_selection_idxs`.
    """
    if not idx_file.is_file():
        raise FileNotFoundError(f"idx 文件不存在: {idx_file}")
    loaded = np.load(idx_file, allow_pickle=False)
    return validate_selection_idxs(
        loaded,
        row_count=row_count,
        allow_duplicates=allow_duplicates,
        source=idx_file,
    )


def validate_header_lines(header_lines: Sequence[str]) -> list[str]:
    """Validate that a CSF header is exactly 5 string lines.

    Raises:
        ValueError: If ``header_lines`` is not a sequence of exactly 5
            strings.
    """
    if not isinstance(header_lines, (list, tuple)) or not all(
        isinstance(line, str) for line in header_lines
    ):
        raise ValueError("CSFs header 必须是字符串列表")
    if len(header_lines) != 5:
        raise ValueError(f"CSFs header 必须是 5 行: 实际 {len(header_lines)} 行")
    return list(header_lines)


def validate_headers_match(
    all_header_lines: Sequence[Sequence[str]],
    *,
    labels: Sequence[str | None] | None = None,
) -> list[str]:
    """Validate that every source's header lines are identical.

    Args:
        all_header_lines: One header-line sequence per source.
        labels: Optional per-source labels used to identify a mismatch.

    Returns:
        The shared header lines.

    Raises:
        ValueError: If ``all_header_lines`` is empty, or any entry differs
            from the first.
    """
    if not all_header_lines:
        raise ValueError("没有可比较的 CSFs header")

    def _label(index: int) -> str:
        if labels is not None and index < len(labels) and labels[index] is not None:
            return str(labels[index])
        return f"#{index}"

    first = validate_header_lines(all_header_lines[0])
    for index, header_lines in enumerate(all_header_lines[1:], start=1):
        candidate = validate_header_lines(header_lines)
        if candidate != first:
            raise ValueError(
                f"多个 CSFs 来源的 header 不一致: {_label(0)} 与 {_label(index)} "
                "的 header_lines 不同"
            )
    return first


def validate_output_path_disjoint(
    output_path: Path,
    input_paths: Iterable[Path | None],
) -> None:
    """Validate that ``output_path`` does not resolve to any input path.

    Raises:
        ValueError: If ``output_path`` equals any non-``None`` entry in
            ``input_paths`` once both are resolved.
    """
    resolved_output = Path(output_path).resolve()
    for input_path in input_paths:
        if input_path is None:
            continue
        if Path(input_path).resolve() == resolved_output:
            raise ValueError(f"输出路径与输入路径冲突: {output_path} == {input_path}")


def validate_coupling_level(coupling_level: int | None) -> int | None:
    """Validate that ``coupling_level`` is ``None`` or a positive integer.

    Raises:
        ValueError: If ``coupling_level`` is given and not positive.
    """
    if coupling_level is not None and coupling_level <= 0:
        raise ValueError(f"coupling_level 必须是 None 或正整数: {coupling_level}")
    return coupling_level


def validate_csf_records(
    block_csfs: Sequence[Sequence[str]],
    *,
    block_index: int | None = None,
) -> None:
    """Validate that every CSF record in a block has exactly 3 lines.

    Raises:
        ValueError: If any record does not have exactly 3 elements, reporting
            the offending block/row for easy lookup.
    """
    block_label = "" if block_index is None else f" block={block_index}"
    for row_index, record in enumerate(block_csfs):
        if len(record) != 3:
            raise ValueError(
                f"CSF 记录必须是 3 行{block_label} row={row_index}: "
                f"实际 {len(record)} 行"
            )
