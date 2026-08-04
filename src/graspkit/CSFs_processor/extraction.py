# -*- encoding: utf-8 -*-
"""Config-free CSF extraction: select rows from a source and merge/write blocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..data_IO import write_csfs_blocks_to_cfile
from .validation import (
    validate_header_lines,
    validate_headers_match,
    validate_output_path_disjoint,
)


@dataclass(frozen=True)
class SelectedCsfsBlock:
    """Selected CSF rows plus their source header.

    Attributes:
        header_lines: The 5-line GRASP header for ``csfs_df``'s source.
        csfs_df: Selected CSF rows (must contain ``line1``/``line2``/``line3``).
        source_path: Path the rows were read from, used for path-conflict
            checks when writing output.
        label: Optional caller-supplied label (e.g. a J value) for
            diagnostics.
    """

    header_lines: list[str]
    csfs_df: pl.DataFrame
    source_path: Path
    label: str | None = None

    def __post_init__(self) -> None:
        validate_header_lines(self.header_lines)


def select_csfs_block(
    *,
    header_lines: Sequence[str],
    csfs_df: pl.DataFrame,
    source_path: Path,
    label: str | None = None,
) -> SelectedCsfsBlock:
    """Build a :class:`SelectedCsfsBlock` after validating its header.

    Args:
        header_lines: The 5-line GRASP header for ``csfs_df``'s source.
        csfs_df: Already-selected CSF rows.
        source_path: Path the rows were read from.
        label: Optional caller-supplied label (e.g. a J value).

    Returns:
        A validated :class:`SelectedCsfsBlock`.

    Raises:
        ValueError: If ``header_lines`` is not exactly 5 strings.
    """
    return SelectedCsfsBlock(
        header_lines=validate_header_lines(header_lines),
        csfs_df=csfs_df,
        source_path=source_path,
        label=label,
    )


def merge_and_write_csfs_blocks(
    blocks: Sequence[SelectedCsfsBlock],
    output_file: Path,
    *,
    extra_input_paths: Sequence[Path | None] = (),
) -> None:
    """Validate header/path consistency, then atomically write merged CSFs.

    Args:
        blocks: Selected CSF blocks to merge, in output order.
        output_file: Destination ``.c`` file.
        extra_input_paths: Additional input paths (e.g. idx files) that must
            not collide with ``output_file``, beyond each block's
            ``source_path``.

    Raises:
        ValueError: If ``blocks`` is empty, headers across blocks differ, or
            ``output_file`` collides with any input path.
    """
    if not blocks:
        raise ValueError("没有可写入的 CSFs block")

    header_lines = validate_headers_match(
        [block.header_lines for block in blocks],
        labels=[block.label for block in blocks],
    )
    validate_output_path_disjoint(
        output_file,
        [*(block.source_path for block in blocks), *extra_input_paths],
    )

    write_csfs_blocks_to_cfile(
        header_lines,
        [block.csfs_df for block in blocks],
        output_file,
    )
