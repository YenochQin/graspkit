import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from graspkit.CSFs_processor import (
    collect_coupling_groups,
    select_csfs_by_coupling_theme,
    select_csfs_by_cumulative_coupling,
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


@pytest.mark.parametrize(
    "use_parquet",
    [
        pytest.param(False, id="dataframe"),
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                os.environ.get("GRASPKIT_TEST_RCSFS_V3") != "1",
                reason="requires an rCSFs build that writes and reads V3",
            ),
            id="parquet",
        ),
    ],
)
def test_v3_coupling_groups_and_selects_csfs_by_block(
    use_parquet: bool,
    tmp_path: Path,
) -> None:
    signatures = [[0, 1, 0, 1], [0, 1, 2, 1], [0, 0, 1], [0, 0, 1]]
    block_ids = [0, 0, 0, 1]
    if use_parquet:
        from rcsfs import convert_csfs, read_csfs

        fixture = Path(__file__).parent / "fixtures" / "coupling_v3.c"
        text_path = tmp_path / fixture.name
        text_path.write_text(
            fixture.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
        )
        parquet_path = tmp_path / "coupling_desc.parquet"
        convert_csfs(text_path, parquet_path)
        assert pl.read_parquet_metadata(parquet_path)["descriptor_version"] == "3"
        _, csfs_df = read_csfs(
            parquet_path, include_block_id=True, include_coupling_signature=True
        )
        assert csfs_df["total_two_j_plus_one"].to_list() == [2, 2, 2, 2]
        assert csfs_df["parity"].to_list() == [1, 1, 1, 1]
        source_indices = [0, 1, 2, 3]
    else:
        source_indices = [7, 12, 25, 42]
        csfs_df = pl.DataFrame(
            {
                "idx": source_indices,
                "block_id": block_ids,
                "coupling_signature": signatures,
            },
            schema={
                "idx": pl.UInt64,
                "block_id": pl.UInt32,
                "coupling_signature": pl.List(pl.Int32),
            },
        )

    assert csfs_df.select("idx", "block_id", "coupling_signature").rows() == list(
        zip(source_indices, block_ids, signatures, strict=True)
    )
    assert collect_coupling_groups(csfs_df).select(
        "block_id", "coupling_signature", "idxs", "global_idxs"
    ).rows() == [
        (0, signatures[0], [0], [source_indices[0]]),
        (0, signatures[1], [1], [source_indices[1]]),
        (0, signatures[2], [2], [source_indices[2]]),
        (1, signatures[3], [0], [source_indices[3]]),
    ]
    assert collect_coupling_groups(csfs_df, coupling_level=1).rows() == [
        (0, [0], 2, [0, 2], [source_indices[0], source_indices[2]]),
        (0, [2], 1, [1], [source_indices[1]]),
        (1, [0], 1, [0], [source_indices[3]]),
    ]

    mix_data = MixCoefficientData(
        blocks=[
            _mix_block(0, np.sqrt(np.array([[0.30, 0.45, 0.25]])), j_value="1/2"),
            _mix_block(1, np.array([[1.0]]), j_value="1/2"),
        ],
        sorted_level_energies=[0.0, 1.0],
    )
    summary = summarize_coupling_ci_squared(csfs_df, mix_data, coupling_level=1)
    assert summary["sum_ci"].to_list() == pytest.approx([0.55, 0.45, 1.0])

    theme = select_csfs_by_coupling_theme(
        csfs_df, mix_data, ci_squared_cutoff=0.5, coupling_level=1
    )
    assert theme["idx"].to_list() == [
        source_indices[0],
        source_indices[2],
        source_indices[3],
    ]

    cumulative = select_csfs_by_cumulative_coupling(
        csfs_df, mix_data, cumulative_threshold=0.4, coupling_level=1
    )
    assert cumulative.prefix_counts == {0: 1, 1: 1}
    assert cumulative.dominant_channels.select(
        "block_id", "coupling_signature"
    ).rows() == [
        (0, [0]),
        (1, [0]),
    ]
    assert cumulative.selected_csfs["idx"].to_list() == source_indices

    bad_total_j = csfs_df.with_columns(
        pl.when(pl.col("block_id") == 1)
        .then(pl.lit([0, 0, 2], dtype=pl.List(pl.Int32)))
        .otherwise(pl.col("coupling_signature"))
        .alias("coupling_signature")
    )
    with pytest.raises(ValueError, match=r"Block 1: CSF J=1 与 rmix J=1/2 不一致"):
        summarize_coupling_ci_squared(bad_total_j, mix_data, coupling_level=1)


def test_v3_closed_core_signature_has_no_intermediate_entries() -> None:
    csfs_df = pl.DataFrame(
        {"idx": [15], "block_id": [0], "coupling_signature": [[0]]},
        schema_overrides={"coupling_signature": pl.List(pl.Int32)},
    )
    mix_data = MixCoefficientData(
        blocks=[_mix_block(0, np.array([[1.0]]), j_value="0")],
        sorted_level_energies=[0.0],
    )

    assert collect_coupling_groups(csfs_df)["coupling_signature"].to_list() == [[0]]
    summary = summarize_coupling_ci_squared(csfs_df, mix_data, coupling_level=1)
    assert summary.select(
        "coupling_signature", "idxs", "global_idxs", "sum_ci"
    ).rows() == [([], [0], [15], 1.0)]


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


def test_collect_coupling_groups_excludes_total_j_before_taking_trailing_values() -> (
    None
):
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


def test_cumulative_coupling_unions_prefix_and_dominant_members_in_source_order() -> (
    None
):
    # Four exactly equal weights. The prefix tie selects later CSFs, whereas
    # two equal channel totals select the channel first seen in source order.
    frame = pl.DataFrame(
        {
            "idx": [7, 12, 4, 9],
            "block_id": [0] * 4,
            "coupling_signature": [[2, 8], [6, 8], [2, 8], [6, 8]],
        },
        schema_overrides={"coupling_signature": pl.List(pl.Int32)},
    )
    mix = MixCoefficientData(
        blocks=[
            _mix_block(0, np.array([[0.5, -0.5, 0.5, -0.5], [1.0, 0.0, 0.0, 0.0]]))
        ],
        sorted_level_energies=[],
    )
    result = select_csfs_by_cumulative_coupling(
        frame, mix, cumulative_threshold=0.4, coupling_level=1, asf_row_indices=[[0]]
    )
    assert result.prefix_counts == {0: 2}
    assert result.dominant_channels["coupling_signature"].to_list() == [[2]]
    assert result.selected_csfs["idx"].to_list() == [7, 4, 9]


def test_cumulative_coupling_requires_complete_asf_like_ml_presampling() -> None:
    mix = MixCoefficientData(
        blocks=[
            _mix_block(0, np.array([[0.1, 0.2, 0.0]])),
            _mix_block(1, np.array([[1.0, 0.0]])),
        ],
        sorted_level_energies=[],
    )
    with pytest.raises(ValueError, match="weights must sum to 1"):
        select_csfs_by_cumulative_coupling(
            _csfs_df(), mix, cumulative_threshold=1.0, coupling_level=1
        )
