import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graspkit.data_IO import (  # noqa: E402
    write_CSFs_pl_to_cfile,
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
