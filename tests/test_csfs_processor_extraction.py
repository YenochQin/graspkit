from pathlib import Path

import polars as pl
import pytest

from graspkit.CSFs_processor.extraction import (
    SelectedCsfsBlock,
    merge_and_write_csfs_blocks,
    select_csfs_block,
)

HEADER = ["h1", "h2", "h3", "h4", "h5"]


def _df(label: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "line1": [f"{label}-1"],
            "line2": [f"{label}-2"],
            "line3": [f"{label}-3"],
        }
    )


def test_select_csfs_block_validates_header() -> None:
    block = select_csfs_block(
        header_lines=HEADER,
        csfs_df=_df("a"),
        source_path=Path("a.parquet"),
        label="0",
    )

    assert block.header_lines == HEADER
    assert block.label == "0"


def test_select_csfs_block_rejects_malformed_header() -> None:
    with pytest.raises(ValueError, match="5 行"):
        select_csfs_block(
            header_lines=["h1", "h2"],
            csfs_df=_df("a"),
            source_path=Path("a.parquet"),
        )


def test_merge_and_write_csfs_blocks_writes_header_and_all_blocks(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "merged.c"
    blocks = [
        select_csfs_block(
            header_lines=HEADER,
            csfs_df=_df("j0"),
            source_path=tmp_path / "j0.parquet",
            label="0",
        ),
        select_csfs_block(
            header_lines=HEADER,
            csfs_df=_df("j2"),
            source_path=tmp_path / "j2.parquet",
            label="2",
        ),
    ]

    merge_and_write_csfs_blocks(blocks, output_file)

    text = output_file.read_text(encoding="utf-8")
    assert text == (
        "h1\nh2\nh3\nh4\nh5\n"
        "j0-1\nj0-2\nj0-3\n"
        " *\n"
        "j2-1\nj2-2\nj2-3\n"
    )


def test_merge_and_write_csfs_blocks_rejects_empty_blocks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="没有可写入"):
        merge_and_write_csfs_blocks([], tmp_path / "out.c")


def test_merge_and_write_csfs_blocks_rejects_mismatched_headers(
    tmp_path: Path,
) -> None:
    blocks = [
        select_csfs_block(
            header_lines=HEADER,
            csfs_df=_df("j0"),
            source_path=tmp_path / "j0.parquet",
            label="0",
        ),
        select_csfs_block(
            header_lines=["h1", "h2", "h3", "h4", "DIFFERENT"],
            csfs_df=_df("j2"),
            source_path=tmp_path / "j2.parquet",
            label="2",
        ),
    ]

    with pytest.raises(ValueError, match="header 不一致"):
        merge_and_write_csfs_blocks(blocks, tmp_path / "out.c")


def test_merge_and_write_csfs_blocks_rejects_output_matching_source(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "j0.parquet"
    blocks = [
        select_csfs_block(
            header_lines=HEADER,
            csfs_df=_df("j0"),
            source_path=source_path,
            label="0",
        ),
    ]

    with pytest.raises(ValueError, match="冲突"):
        merge_and_write_csfs_blocks(blocks, source_path)


def test_merge_and_write_csfs_blocks_rejects_output_matching_extra_input_path(
    tmp_path: Path,
) -> None:
    idx_file = tmp_path / "idxs.npy"
    blocks = [
        select_csfs_block(
            header_lines=HEADER,
            csfs_df=_df("j0"),
            source_path=tmp_path / "j0.parquet",
            label="0",
        ),
    ]

    with pytest.raises(ValueError, match="冲突"):
        merge_and_write_csfs_blocks(
            blocks,
            idx_file,
            extra_input_paths=[idx_file],
        )


def test_selected_csfs_block_is_a_dataclass_with_expected_fields() -> None:
    block = SelectedCsfsBlock(
        header_lines=HEADER,
        csfs_df=_df("a"),
        source_path=Path("a.parquet"),
    )

    assert block.label is None
