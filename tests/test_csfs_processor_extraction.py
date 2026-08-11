from pathlib import Path

import polars as pl
import pytest

from graspkit.CSFs_processor.extraction import (
    CsfDocument,
    SelectedCsfsBlock,
    create_csf_document,
    merge_and_write_csfs_blocks,
    select_csfs_block,
    write_csf_documents,
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


def test_create_csf_document_normalizes_header_and_validates_rows() -> None:
    document = create_csf_document(
        header_lines=HEADER,
        csfs_df=_df("a"),
        source_path=Path("a.parquet"),
        label="0",
    )

    assert isinstance(document, CsfDocument)
    assert document.header_lines == tuple(HEADER)
    assert document.label == "0"


def test_csf_document_rejects_missing_row_columns() -> None:
    with pytest.raises(ValueError, match="line3"):
        create_csf_document(
            header_lines=HEADER,
            csfs_df=_df("a").drop("line3"),
            source_path=Path("a.parquet"),
        )


def test_write_csf_documents_partitions_a_multiblock_frame(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "block_id": [0, 1],
            "line1": ["a1", "b1"],
            "line2": ["a2", "b2"],
            "line3": ["a3", "b3"],
        },
        schema_overrides={"block_id": pl.UInt32},
    )
    document = create_csf_document(
        header_lines=HEADER,
        csfs_df=frame,
        source_path=tmp_path / "source.c",
    )
    output = tmp_path / "selected.c"

    write_csf_documents([document], output)

    assert output.read_text(encoding="utf-8") == (
        "h1\nh2\nh3\nh4\nh5\n"
        "a1\na2\na3\n"
        " *\n"
        "b1\nb2\nb3\n"
    )


def test_select_csfs_block_validates_header() -> None:
    block = select_csfs_block(
        header_lines=HEADER,
        csfs_df=_df("a"),
        source_path=Path("a.parquet"),
        label="0",
    )

    assert block.header_lines == tuple(HEADER)
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
