import pytest

from graspkit.CSFs_processor.coupling import (
    single_block_csfs_final_coupling_J_collector,
)


def test_collector_groups_by_full_coupling_pattern_by_default() -> None:
    block_csfs = [
        ["line1", "line2", "a b J0"],
        ["line1", "line2", "a b J1"],
        ["line1", "line2", "a b J0"],
    ]

    result = single_block_csfs_final_coupling_J_collector(block_csfs)

    assert result[("a", "b", "J0")]["count"] == 2
    assert result[("a", "b", "J0")]["idxs"] == [0, 2]
    assert result[("a", "b", "J1")]["count"] == 1
    assert result[("a", "b", "J1")]["idxs"] == [1]


def test_collector_keeps_trailing_n_tokens_when_coupling_level_set() -> None:
    block_csfs = [
        ["line1", "line2", "a b J0"],
        ["line1", "line2", "c b J0"],
    ]

    result = single_block_csfs_final_coupling_J_collector(block_csfs, coupling_level=1)

    assert set(result) == {("J0",)}
    assert result[("J0",)]["count"] == 2


def test_collector_rejects_non_positive_coupling_level() -> None:
    with pytest.raises(ValueError, match="正整数"):
        single_block_csfs_final_coupling_J_collector(
            [["line1", "line2", "a J0"]], coupling_level=0
        )


def test_collector_rejects_csf_record_without_three_lines() -> None:
    with pytest.raises(ValueError, match="3 行"):
        single_block_csfs_final_coupling_J_collector([["line1", "line2"]])
