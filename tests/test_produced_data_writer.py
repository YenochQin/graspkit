import sys
from pathlib import Path

import polars as pl
import pytest
import rtoml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graspkit.data_IO import (  # noqa: E402
    csfs_header_path_for_parquet,
    load_csfs_header_lines,
    write_CSFs_pl_to_cfile,
    write_csfs_blocks_to_cfile,
    write_sorted_CSFs_to_cfile,
)


def test_write_sorted_csfs_to_cfile_preserves_block_structure(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "sorted.c"

    write_sorted_CSFs_to_cfile(
        ["header-1\n", "header-2\n", "header-3\n", "header-4\n"],
        [
            [["block-1-line-1\n", "block-1-line-2\n", "block-1-line-3\n"]],
            [["block-2-line-1\n", "block-2-line-2\n", "block-2-line-3\n"]],
        ],
        output_file,
    )

    assert output_file.read_text() == (
        "header-1\nheader-2\nheader-3\nheader-4\nCSF(s):\n"
        "block-1-line-1\nblock-1-line-2\nblock-1-line-3\n *\n"
        "block-2-line-1\nblock-2-line-2\nblock-2-line-3\n"
    )


def test_write_polars_csfs_to_cfile_writes_selected_columns(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "polars.c"
    dataframe = pl.DataFrame(
        {
            "line1": ["first-line"],
            "line2": ["second-line"],
            "line3": ["third-line"],
            "ignored": ["not-written"],
        }
    )

    write_CSFs_pl_to_cfile(
        ["header-1", "header-2", "header-3", "header-4", "header-5"],
        dataframe,
        output_file,
    )

    assert output_file.read_text() == (
        "header-1\nheader-2\nheader-3\nheader-4\nheader-5\n"
        "first-line\nsecond-line\nthird-line\n"
    )


def test_write_csfs_blocks_to_cfile_writes_all_blocks_with_separator(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "blocks.c"
    header = ["h1", "h2", "h3", "h4", "h5"]
    block_a = pl.DataFrame({"line1": ["a1"], "line2": ["a2"], "line3": ["a3"]})
    block_b = pl.DataFrame({"line1": ["b1"], "line2": ["b2"], "line3": ["b3"]})

    write_csfs_blocks_to_cfile(header, [block_a, block_b], output_file)

    assert output_file.read_text(encoding="utf-8") == (
        "h1\nh2\nh3\nh4\nh5\n"
        "a1\na2\na3\n"
        " *\n"
        "b1\nb2\nb3\n"
    )


def test_write_csfs_blocks_to_cfile_rejects_wrong_header_length(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="header info error"):
        write_csfs_blocks_to_cfile(
            ["h1", "h2"],
            [pl.DataFrame({"line1": ["a"], "line2": ["b"], "line3": ["c"]})],
            tmp_path / "out.c",
        )


def test_write_csfs_blocks_to_cfile_rejects_empty_blocks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="没有可写入"):
        write_csfs_blocks_to_cfile(
            ["h1", "h2", "h3", "h4", "h5"],
            [],
            tmp_path / "out.c",
        )


def test_write_csfs_blocks_to_cfile_does_not_clobber_existing_target_on_failure(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "existing.c"
    output_file.write_text("original contents\n", encoding="utf-8")
    good_block = pl.DataFrame({"line1": ["a"], "line2": ["b"], "line3": ["c"]})
    bad_block = pl.DataFrame({"line1": ["a"], "oops": ["b"]})

    with pytest.raises(Exception):
        write_csfs_blocks_to_cfile(
            ["h1", "h2", "h3", "h4", "h5"],
            [good_block, bad_block],
            output_file,
        )

    assert output_file.read_text(encoding="utf-8") == "original contents\n"
    leftover_temp_files = list(tmp_path.glob(".*existing.c*"))
    assert leftover_temp_files == []


def test_write_polars_csfs_to_cfile_does_not_clobber_existing_target_on_failure(
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "existing.c"
    output_file.write_text("original contents\n", encoding="utf-8")
    bad_dataframe = pl.DataFrame({"line1": ["a"], "oops": ["b"]})

    with pytest.raises(Exception):
        write_CSFs_pl_to_cfile(
            ["h1", "h2", "h3", "h4", "h5"],
            bad_dataframe,
            output_file,
        )

    assert output_file.read_text(encoding="utf-8") == "original contents\n"


def test_csfs_header_path_for_parquet_derives_sidecar_name(tmp_path: Path) -> None:
    parquet_path = tmp_path / "raw.parquet"

    assert csfs_header_path_for_parquet(parquet_path) == tmp_path / "raw_header.toml"


def test_load_csfs_header_lines_reads_valid_sidecar(tmp_path: Path) -> None:
    header_path = tmp_path / "raw_header.toml"
    rtoml.dump(
        {"header_info": {"header_lines": ["h1", "h2", "h3", "h4", "h5"]}},
        header_path,
    )

    assert load_csfs_header_lines(header_path) == ["h1", "h2", "h3", "h4", "h5"]


def test_load_csfs_header_lines_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csfs_header_lines(tmp_path / "missing_header.toml")


def test_load_csfs_header_lines_rejects_missing_header_info(tmp_path: Path) -> None:
    header_path = tmp_path / "bad_header.toml"
    rtoml.dump({"other": {}}, header_path)

    with pytest.raises(ValueError, match="header_info"):
        load_csfs_header_lines(header_path)


def test_load_csfs_header_lines_rejects_wrong_line_count(tmp_path: Path) -> None:
    header_path = tmp_path / "bad_header.toml"
    rtoml.dump({"header_info": {"header_lines": ["h1", "h2"]}}, header_path)

    with pytest.raises(ValueError, match="5 行"):
        load_csfs_header_lines(header_path)


def test_load_csfs_header_lines_rejects_non_string_line(tmp_path: Path) -> None:
    header_path = tmp_path / "bad_header.toml"
    rtoml.dump(
        {"header_info": {"header_lines": ["h1", "h2", "h3", "h4", 5]}},
        header_path,
    )

    with pytest.raises(ValueError, match="header_lines"):
        load_csfs_header_lines(header_path)
