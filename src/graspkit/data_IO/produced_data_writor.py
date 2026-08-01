# -*- encoding: utf-8 -*-
"""Writers for stable GRASP CSF data formats."""

from pathlib import Path

import polars as pl


def write_sorted_CSFs_to_cfile(
    CSFs_file_info: list[str],
    sorted_CSFs_data_list: list[list[list[str]]],
    output_file: str | Path,
) -> None:
    """Write block-grouped CSF records to a GRASP ``.c`` file."""
    if len(CSFs_file_info) != 4:
        raise ValueError("CSFs file header info error!")
    blocks_num = len(sorted_CSFs_data_list)
    with Path(output_file).open("w") as file:
        file.writelines(CSFs_file_info)
        _ = file.write("CSF(s):\n")
        for idx, block in enumerate(sorted_CSFs_data_list):
            for csf in block:
                file.writelines(csf)
            if idx != blocks_num - 1:
                _ = file.write(" *\n")


def write_CSFs_pl_to_cfile(
    CSFs_file_info: list[str],
    CSFs_data_df: pl.DataFrame,
    output_file: str | Path,
) -> None:
    """Write CSF records from a Polars DataFrame to a GRASP ``.c`` file."""
    if len(CSFs_file_info) != 5:
        raise ValueError("CSFs file header info error!")
    with Path(output_file).open("w") as file:
        for line in CSFs_file_info:
            _ = file.write(f"{line}\n")
        for row in CSFs_data_df.select(["line1", "line2", "line3"]).iter_rows():
            _ = file.write("\n".join(row) + "\n")
