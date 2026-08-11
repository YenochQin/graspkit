import numpy as np
import polars as pl
import pytest

from graspkit.CSFs_processor.coupling import (
    collect_coupling_groups,
    select_csfs_by_coupling_theme,
    single_block_csfs_final_coupling_J_collector,
    summarize_coupling_ci_squared,
)
from graspkit.utils.data_modules import MixCoefficientBlock, MixCoefficientData


def _csfs_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "idx": [0, 1, 2, 3, 4],
            "block_id": [0, 0, 0, 1, 1],
            "line1": ["l1-0", "l1-1", "l1-2", "l1-3", "l1-4"],
            "line2": ["l2-0", "l2-1", "l2-2", "l2-3", "l2-4"],
            "line3": ["l3-0", "l3-1", "l3-2", "l3-3", "l3-4"],
            "coupling_signature": [
                [1, 2, 8],
                [3, 4, 8],
                [1, 2, 8],
                [5, 6],
                [7, 6],
            ],
        },
        schema_overrides={
            "idx": pl.UInt64,
            "block_id": pl.UInt32,
            "coupling_signature": pl.List(pl.Int32),
        },
    )


def _mix_block(
    block_index: int,
    coefficients: np.ndarray,
    *,
    j_value: str | None = None,
) -> MixCoefficientBlock:
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    if j_value is None:
        j_value = {0: "4", 1: "3"}[block_index]
    return MixCoefficientBlock(
        block_index=block_index,
        csf_count=coefficient_array.shape[1],
        level_count=coefficient_array.shape[0],
        j_value_location=block_index + 1,
        j_value=j_value,
        parity=1,
        level_ids=np.arange(coefficient_array.shape[0], dtype=np.int64),
        base_energy=float(block_index),
        level_energies=np.arange(coefficient_array.shape[0], dtype=np.float64),
        mix_coefficients=coefficient_array,
    )


def test_select_csfs_by_coupling_theme_returns_threshold_and_dominant_union() -> None:
    mix_data = MixCoefficientData(
        blocks=[
            _mix_block(
                0,
                np.sqrt(
                    np.array(
                        [[0.60, 0.00, 0.10], [0.00, 0.80, 0.00]],
                        dtype=np.float64,
                    )
                ),
            ),
            _mix_block(1, np.sqrt(np.array([[0.10, 0.70]], dtype=np.float64))),
        ],
        sorted_level_energies=[0.0, 0.1, 1.0],
    )

    result = select_csfs_by_coupling_theme(
        _csfs_df(),
        mix_data,
        asf_row_indices=[[0, 1], [0]],
        ci_squared_cutoff=0.25,
        coupling_level=3,
    )

    assert result.get_column("idx").to_list() == [0, 1, 2, 4]


def test_select_csfs_by_coupling_theme_uses_signature_not_raw_line3() -> None:
    csfs_df = _csfs_df().with_columns(pl.lit("same raw line").alias("line3"))
    mix_data = MixCoefficientData(
        blocks=[
            _mix_block(0, np.sqrt(np.array([[0.60, 0.00, 0.10]]))),
            _mix_block(1, np.sqrt(np.array([[0.70, 0.10]]))),
        ],
        sorted_level_energies=[0.0, 1.0],
    )

    result = select_csfs_by_coupling_theme(
        csfs_df,
        mix_data,
        ci_squared_cutoff=0.5,
        coupling_level=3,
    )

    assert result.get_column("idx").to_list() == [0, 2, 3]


def test_select_csfs_by_coupling_theme_rejects_block_count_mismatch() -> None:
    mix_data = MixCoefficientData(
        blocks=[_mix_block(0, np.array([[1.0, 0.0, 0.0]]))],
        sorted_level_energies=[0.0],
    )

    with pytest.raises(ValueError, match="block IDs"):
        select_csfs_by_coupling_theme(
            _csfs_df(),
            mix_data,
            ci_squared_cutoff=0.1,
            coupling_level=1,
        )


@pytest.mark.parametrize("bad_level", [0, -1])
def test_select_csfs_by_coupling_theme_rejects_non_positive_level(
    bad_level: int,
) -> None:
    mix_data = MixCoefficientData(
        blocks=[
            _mix_block(0, np.array([[1.0, 0.0, 0.0]])),
            _mix_block(1, np.array([[1.0, 0.0]])),
        ],
        sorted_level_energies=[0.0, 1.0],
    )

    with pytest.raises(ValueError, match="正整数"):
        select_csfs_by_coupling_theme(
            _csfs_df(),
            mix_data,
            ci_squared_cutoff=0.1,
            coupling_level=bad_level,
        )


def test_collect_coupling_groups_uses_rcsfs_signature_and_block_local_indices() -> None:
    result = collect_coupling_groups(_csfs_df())

    assert result.schema == {
        "block_id": pl.UInt32,
        "coupling_signature": pl.List(pl.Int32),
        "count": pl.UInt32,
        "idxs": pl.List(pl.UInt64),
        "global_idxs": pl.List(pl.UInt64),
    }
    assert result.to_dicts() == [
        {
            "block_id": 0,
            "coupling_signature": [1, 2, 8],
            "count": 2,
            "idxs": [0, 2],
            "global_idxs": [0, 2],
        },
        {
            "block_id": 0,
            "coupling_signature": [3, 4, 8],
            "count": 1,
            "idxs": [1],
            "global_idxs": [1],
        },
        {
            "block_id": 1,
            "coupling_signature": [5, 6],
            "count": 1,
            "idxs": [0],
            "global_idxs": [3],
        },
        {
            "block_id": 1,
            "coupling_signature": [7, 6],
            "count": 1,
            "idxs": [1],
            "global_idxs": [4],
        },
    ]


def test_collect_coupling_groups_excludes_total_j_before_taking_trailing_values() -> None:
    result = collect_coupling_groups(_csfs_df(), coupling_level=1)

    assert result.to_dicts() == [
        {
            "block_id": 0,
            "coupling_signature": [2],
            "count": 2,
            "idxs": [0, 2],
            "global_idxs": [0, 2],
        },
        {
            "block_id": 0,
            "coupling_signature": [4],
            "count": 1,
            "idxs": [1],
            "global_idxs": [1],
        },
        {
            "block_id": 1,
            "coupling_signature": [5],
            "count": 1,
            "idxs": [0],
            "global_idxs": [3],
        },
        {
            "block_id": 1,
            "coupling_signature": [7],
            "count": 1,
            "idxs": [1],
            "global_idxs": [4],
        },
    ]


@pytest.mark.parametrize(
    ("signature", "coupling_level", "expected"),
    [
        ([2, 4, 6, 8], 2, [4, 6]),
        ([5, 8], 2, [5]),
        ([8], 2, []),
    ],
)
def test_collect_coupling_groups_takes_requested_middle_coupling_levels(
    signature: list[int],
    coupling_level: int,
    expected: list[int],
) -> None:
    csfs_df = pl.DataFrame(
        {
            "idx": [0],
            "block_id": [0],
            "coupling_signature": [signature],
        },
        schema_overrides={
            "idx": pl.UInt64,
            "block_id": pl.UInt32,
            "coupling_signature": pl.List(pl.Int32),
        },
    )

    result = collect_coupling_groups(csfs_df, coupling_level=coupling_level)

    assert result.get_column("coupling_signature").to_list() == [expected]


@pytest.mark.parametrize(
    ("column", "expected_message"),
    [
        ("block_id", "block_id"),
        ("idx", "idx"),
        ("coupling_signature", "coupling_signature"),
    ],
)
def test_collect_coupling_groups_rejects_missing_contract_column(
    column: str,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        collect_coupling_groups(_csfs_df().drop(column))


def test_collect_coupling_groups_rejects_wrong_signature_dtype() -> None:
    csfs_df = _csfs_df().with_columns(
        pl.col("coupling_signature").cast(pl.List(pl.Int64))
    )

    with pytest.raises(ValueError, match=r"List\(Int32\)"):
        collect_coupling_groups(csfs_df)


@pytest.mark.parametrize("bad_level", [0, -1, 1.5, True])
def test_collect_coupling_groups_rejects_invalid_coupling_level(
    bad_level: object,
) -> None:
    with pytest.raises(ValueError, match="正整数"):
        collect_coupling_groups(_csfs_df(), coupling_level=bad_level)  # type: ignore[arg-type]


def test_summarize_coupling_ci_squared_returns_long_polars_summary() -> None:
    mix_data = MixCoefficientData(
        blocks=[
            _mix_block(
                0,
                np.array(
                    [
                        [0.5, 0.1, 0.0],
                        [0.0, 0.3, 0.1],
                    ]
                ),
            ),
            _mix_block(1, np.array([[0.4, 0.2]])),
        ],
        sorted_level_energies=[0.0, 0.1, 1.0],
    )

    result = summarize_coupling_ci_squared(
        _csfs_df(),
        mix_data,
        asf_row_indices=[[1], [0]],
        coupling_level=2,
    )

    assert result.schema == {
        "block_id": pl.UInt32,
        "asf_index": pl.Int64,
        "coupling_signature": pl.List(pl.Int32),
        "count": pl.UInt32,
        "idxs": pl.List(pl.UInt64),
        "global_idxs": pl.List(pl.UInt64),
        "sum_ci": pl.Float64,
    }
    assert result.to_dicts() == [
        {
            "block_id": 0,
            "asf_index": 1,
            "coupling_signature": [1, 2],
            "count": 2,
            "idxs": [0, 2],
            "global_idxs": [0, 2],
            "sum_ci": pytest.approx(0.01),
        },
        {
            "block_id": 0,
            "asf_index": 1,
            "coupling_signature": [3, 4],
            "count": 1,
            "idxs": [1],
            "global_idxs": [1],
            "sum_ci": pytest.approx(0.09),
        },
        {
            "block_id": 1,
            "asf_index": 0,
            "coupling_signature": [5],
            "count": 1,
            "idxs": [0],
            "global_idxs": [3],
            "sum_ci": pytest.approx(0.16),
        },
        {
            "block_id": 1,
            "asf_index": 0,
            "coupling_signature": [7],
            "count": 1,
            "idxs": [1],
            "global_idxs": [4],
            "sum_ci": pytest.approx(0.04),
        },
    ]


def test_summarize_coupling_ci_squared_rejects_block_count_mismatch() -> None:
    mix_data = MixCoefficientData(
        blocks=[_mix_block(0, np.array([[0.5, 0.1, 0.0]]))],
        sorted_level_energies=[0.0],
    )

    with pytest.raises(ValueError, match="block"):
        summarize_coupling_ci_squared(_csfs_df(), mix_data)


def test_summarize_coupling_ci_squared_rejects_csf_count_mismatch() -> None:
    mix_data = MixCoefficientData(
        blocks=[
            _mix_block(0, np.array([[0.5, 0.1]])),
            _mix_block(1, np.array([[0.4, 0.2]])),
        ],
        sorted_level_energies=[0.0, 1.0],
    )

    with pytest.raises(ValueError, match="CSF"):
        summarize_coupling_ci_squared(_csfs_df(), mix_data)


def test_summarize_coupling_ci_squared_rejects_j_value_mismatch() -> None:
    mix_data = MixCoefficientData(
        blocks=[
            _mix_block(0, np.array([[0.5, 0.1, 0.0]]), j_value="7/2"),
            _mix_block(1, np.array([[0.4, 0.2]])),
        ],
        sorted_level_energies=[0.0, 1.0],
    )

    with pytest.raises(ValueError, match=r"J.*不一致"):
        summarize_coupling_ci_squared(_csfs_df(), mix_data)


def test_summarize_coupling_ci_squared_accepts_half_integer_j_value() -> None:
    csfs_df = pl.DataFrame(
        {
            "idx": [0, 1],
            "block_id": [0, 0],
            "coupling_signature": [[3, 1], [5, 1]],
        },
        schema_overrides={
            "idx": pl.UInt64,
            "block_id": pl.UInt32,
            "coupling_signature": pl.List(pl.Int32),
        },
    )
    mix_data = MixCoefficientData(
        blocks=[
            _mix_block(
                0,
                np.array([[0.5, 0.1], [0.2, 0.4]]),
                j_value="1/2",
            )
        ],
        sorted_level_energies=[0.0, 0.1],
    )

    result = summarize_coupling_ci_squared(csfs_df, mix_data)

    assert result.height == 4


def test_summarize_coupling_ci_squared_rejects_multiple_j_values_in_csf_block() -> None:
    csfs_df = _csfs_df().with_columns(
        pl.when(pl.col("idx") == 0)
        .then(pl.lit([1, 2, 7], dtype=pl.List(pl.Int32)))
        .otherwise(pl.col("coupling_signature"))
        .alias("coupling_signature")
    )
    mix_data = MixCoefficientData(
        blocks=[
            _mix_block(0, np.array([[0.5, 0.1, 0.0]])),
            _mix_block(1, np.array([[0.4, 0.2]])),
        ],
        sorted_level_energies=[0.0, 1.0],
    )

    with pytest.raises(ValueError, match="多个总 J"):
        summarize_coupling_ci_squared(csfs_df, mix_data)


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
