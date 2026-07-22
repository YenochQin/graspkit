from pathlib import Path

from graspkit.data_IO import CSFLoader


def test_csf_loader_reads_multiple_blocks_after_stripping_line_endings(
    tmp_path: Path,
) -> None:
    csfs_file = tmp_path / "multi.c"
    csfs_file.write_text(
        """Core subshells:
1s
Peel subshells:
2s 2p
CSF(s):
block-0-line-1
block-0-line-2
coupling 0 +
 *
block-1-line-1
block-1-line-2
coupling 1/2 -
""",
        encoding="utf-8",
    )

    result = CSFLoader(csfs_file).load()

    assert result.block_num == 2
    assert result.CSFs_block_length == [1, 1]
    assert result.CSFs_block_j_value == ["0", "1/2"]
    assert result.CSFs_block_data == [
        [["block-0-line-1", "block-0-line-2", "coupling 0 +"]],
        [["block-1-line-1", "block-1-line-2", "coupling 1/2 -"]],
    ]
