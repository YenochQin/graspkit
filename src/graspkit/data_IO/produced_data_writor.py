# -*- encoding: utf-8 -*-
"""Writers and header loader for stable GRASP CSF data formats."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from pathlib import Path
import tempfile
from typing import TextIO, cast

import polars as pl
import rtoml

CSF_HEADER_LINE_COUNT = 5


def _atomic_write(
    output_file: str | Path,
    write_body: Callable[[TextIO], None],
) -> None:
    """Write via a same-directory temp file, then atomically replace the target.

    On any failure, the temp file is removed and ``output_file`` is left
    untouched.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            write_body(file)
        temp_path.replace(output_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def write_csfs_blocks_to_cfile(
    header_lines: list[str],
    blocks: Sequence[pl.DataFrame],
    output_file: str | Path,
) -> None:
    """Write multiple CSF blocks from a shared header to one GRASP ``.c`` file.

    Blocks are separated by a bare ``" *"`` line, matching GRASP's own
    multi-block ``.c`` format. The write is atomic: on failure, ``output_file``
    is left untouched.

    Args:
        header_lines: Exactly 5 GRASP header lines, shared by every block.
        blocks: CSF blocks to write in order; each must contain ``line1``,
            ``line2``, and ``line3`` columns.
        output_file: Destination ``.c`` file.

    Raises:
        ValueError: If ``header_lines`` is not exactly 5 lines, or ``blocks``
            is empty.
    """
    if len(header_lines) != CSF_HEADER_LINE_COUNT:
        raise ValueError("CSFs file header info error!")
    if not blocks:
        raise ValueError("没有可写入的 CSFs block")

    def _write(file: TextIO) -> None:
        for line in header_lines:
            file.write(f"{line}\n")
        for block_idx, csfs_df in enumerate(blocks):
            for row in csfs_df.select(["line1", "line2", "line3"]).iter_rows():
                file.write("\n".join(cast(tuple[str, str, str], row)))
                file.write("\n")
            if block_idx != len(blocks) - 1:
                file.write(" *\n")

    _atomic_write(output_file, _write)


def csfs_header_path_for_parquet(parquet_path: str | Path) -> Path:
    """Return the sidecar header TOML path for an ``rcsfs``-converted parquet file."""
    parquet_path = Path(parquet_path)
    return parquet_path.with_name(f"{parquet_path.stem}_header.toml")


def load_csfs_header_lines(header_path: str | Path) -> list[str]:
    """Load and validate the 5-line ``[header_info].header_lines`` sidecar.

    Args:
        header_path: Path to the header TOML file (see
            :func:`csfs_header_path_for_parquet`).

    Returns:
        The 5-line GRASP header.

    Raises:
        FileNotFoundError: If ``header_path`` does not exist.
        ValueError: If the TOML file is missing ``[header_info].header_lines``
            or it is not exactly 5 strings.
    """
    header_path = Path(header_path)
    if not header_path.is_file():
        raise FileNotFoundError(f"CSFs header TOML 文件不存在: {header_path}")

    header = cast(dict[str, object], rtoml.load(header_path))

    raw_header_info = header.get("header_info")
    if not isinstance(raw_header_info, dict):
        raise ValueError(f"header TOML 缺少 [header_info]: {header_path}")
    header_info = cast(dict[str, object], raw_header_info)

    raw_header_lines = header_info.get("header_lines")
    if not isinstance(raw_header_lines, list):
        raise ValueError(f"header TOML 缺少 header_info.header_lines: {header_path}")

    header_lines: list[str] = []
    for line in cast(list[object], raw_header_lines):
        if not isinstance(line, str):
            raise ValueError(
                f"header TOML 缺少 header_info.header_lines: {header_path}"
            )
        header_lines.append(line)

    if len(header_lines) != CSF_HEADER_LINE_COUNT:
        raise ValueError(f"CSFs header 必须是 5 行: {header_path}")
    return header_lines
