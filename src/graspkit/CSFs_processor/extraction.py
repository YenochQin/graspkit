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


_CSF_ROW_COLUMNS = ("line1", "line2", "line3")


def _validate_csf_frame(csfs_df: pl.DataFrame) -> None:
    missing = set(_CSF_ROW_COLUMNS).difference(csfs_df.columns)
    if missing:
        raise ValueError(f"CSF DataFrame 缺少必要列: {', '.join(sorted(missing))}")
    if csfs_df.is_empty():
        raise ValueError("CSF DataFrame 不能为空")
    for column in _CSF_ROW_COLUMNS:
        if csfs_df.schema[column] != pl.String:
            raise ValueError(f"{column} 必须是 Polars String 列")
        if csfs_df[column].null_count() != 0:
            raise ValueError(f"{column} 不能包含 null")
    if "block_id" in csfs_df.columns:
        if not csfs_df.schema["block_id"].is_integer():
            raise ValueError("block_id 必须是整数列")
        if csfs_df["block_id"].null_count() != 0:
            raise ValueError("block_id 不能包含 null")


@dataclass(frozen=True)
class CsfDocument:
    """Validated CSF rows and their shared five-line GRASP header.

    Attributes:
        header_lines: The 5-line GRASP header for ``csfs_df``'s source.
        csfs_df: CSF rows; a ``block_id`` column may describe multiple blocks.
        source_path: Path the rows were read from, used for path-conflict
            checks when writing output.
        label: Optional caller-supplied label (e.g. a J value) for
            diagnostics.
    """

    header_lines: tuple[str, str, str, str, str]
    csfs_df: pl.DataFrame
    source_path: Path
    label: str | None = None

    def __post_init__(self) -> None:
        validated_header = validate_header_lines(self.header_lines)
        object.__setattr__(self, "header_lines", tuple(validated_header))
        _validate_csf_frame(self.csfs_df)

    def blocks(self) -> list[pl.DataFrame]:
        """Return source-ordered frames ready for GRASP block writing."""
        if "block_id" not in self.csfs_df.columns:
            return [self.csfs_df]
        return self.csfs_df.partition_by("block_id", maintain_order=True)


SelectedCsfsBlock = CsfDocument


def create_csf_document(
    *,
    header_lines: Sequence[str],
    csfs_df: pl.DataFrame,
    source_path: Path,
    label: str | None = None,
) -> CsfDocument:
    """Create the unified CSF document used across selection workflows."""
    validated_header = validate_header_lines(header_lines)
    return CsfDocument(
        header_lines=(
            validated_header[0],
            validated_header[1],
            validated_header[2],
            validated_header[3],
            validated_header[4],
        ),
        csfs_df=csfs_df,
        source_path=source_path,
        label=label,
    )


def select_csfs_block(
    *,
    header_lines: Sequence[str],
    csfs_df: pl.DataFrame,
    source_path: Path,
    label: str | None = None,
) -> CsfDocument:
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
    return create_csf_document(
        header_lines=header_lines,
        csfs_df=csfs_df,
        source_path=source_path,
        label=label,
    )


def write_csf_documents(
    documents: Sequence[CsfDocument],
    output_file: Path,
    *,
    extra_input_paths: Sequence[Path | None] = (),
) -> None:
    """Validate header/path consistency, then atomically write merged CSFs.

    Args:
        documents: CSF documents to merge in source and block order.
        output_file: Destination ``.c`` file.
        extra_input_paths: Additional input paths (e.g. idx files) that must
            not collide with ``output_file``, beyond each block's
            ``source_path``.

    Raises:
        ValueError: If ``blocks`` is empty, headers across blocks differ, or
            ``output_file`` collides with any input path.
    """
    if not documents:
        raise ValueError("没有可写入的 CSFs block")

    header_lines = validate_headers_match(
        [document.header_lines for document in documents],
        labels=[document.label for document in documents],
    )
    validate_output_path_disjoint(
        output_file,
        [*(document.source_path for document in documents), *extra_input_paths],
    )

    write_csfs_blocks_to_cfile(
        header_lines,
        [block for document in documents for block in document.blocks()],
        output_file,
    )


def merge_and_write_csfs_blocks(
    blocks: Sequence[CsfDocument],
    output_file: Path,
    *,
    extra_input_paths: Sequence[Path | None] = (),
) -> None:
    """Compatibility alias for :func:`write_csf_documents`."""
    write_csf_documents(
        blocks,
        output_file,
        extra_input_paths=extra_input_paths,
    )
