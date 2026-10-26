"""Exact jj threshold/index contracts on small in-memory CSF documents."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from graspkit.CSFs_processor.extraction import create_csf_document
from graspkit.CSFs_processor.mr_selection import MrTarget, select_mr_csfs
from graspkit.utils.data_modules import MixCoefficientBlock, MixCoefficientData


HEADER = ["Core subshells:", "", "Peel subshells:", "  2s   2p-  2p   3s", "CSF(s):"]


def inputs(coefficients=None, *, empty_first=False):
    coefficients = (
        np.array([[0.8, -0.6, 0, 0], [0, 0, 0.6, -0.8]])
        if coefficients is None
        else np.array(coefficients)
    )
    block = MixCoefficientBlock(
        0,
        4,
        len(coefficients),
        1,
        "0",
        1,
        np.array([1, 4][: len(coefficients)], dtype=np.int64),
        -2.0,
        np.zeros(len(coefficients)),
        coefficients,
    )
    lines = ["  2s ( 2)", "  2p-( 2)", "  2p ( 2)", "  3s ( 2)"]
    frame = pl.DataFrame(
        {
            "idx": [80, 81, 82, 83],
            "block_id": [0] * 4,
            "line1": lines,
            "line2": [""] * 4,
            "line3": ["       0+"] * 4,
        }
    )
    blocks = [block]
    lengths = [4]
    if empty_first:
        blocks = [
            replace(
                block,
                level_count=0,
                level_ids=np.empty(0, dtype=np.int64),
                level_energies=np.empty(0),
                mix_coefficients=np.empty((0, 4)),
            ),
            replace(block, block_index=1),
        ]
        lengths = [4, 4]
        frame = pl.concat(
            [frame, frame.with_columns(pl.lit(1, dtype=pl.Int64).alias("block_id"))]
        )
    return (
        create_csf_document(
            header_lines=HEADER, csfs_df=frame, source_path=Path("source.c")
        ),
        lengths,
        MixCoefficientData(blocks, [], 2, 4),
    )


def test_sparse_level_ids_union_signed_coefficients_and_source_offsets():
    doc, lengths, mixing = inputs(empty_first=True)
    result = select_mr_csfs(
        doc, lengths, mixing, [MrTarget(1, 1), MrTarget(1, 4)], amplitude_cutoff=0.7
    )
    assert result.document.csfs_df["line1"].to_list() == ["  2s ( 2)", "  3s ( 2)"]
    assert [row["source_row_index_0based"] for row in result.selected_rows] == [
        4,
        7,
        4,
        7,
    ]
    assert [row["asf_row_index_0based"] for row in result.target_summaries] == [0, 1]
    assert [row["level_id_1based"] for row in result.target_summaries] == [2, 5]
    assert [row["coefficient"] for row in result.selected_rows] == [0.8, 0, 0, -0.8]
    assert [row["triggering_level_ids_1based"] for row in result.selected_rows] == [
        "2",
        "5",
        "2",
        "5",
    ]
    assert result.block_mapping[0]["output_block_1based"] == 1
    assert result.target_summaries[0]["weight_kept"] == pytest.approx(0.64)


def test_family_summary_does_not_expand_couplings():
    doc, lengths, mixing = inputs([[0, -0.8, 0.6, 0]])
    result = select_mr_csfs(
        doc, lengths, mixing, [MrTarget(0, 1)], amplitude_cutoff=0.7
    )
    assert result.document.csfs_df.height == 1
    family = result.configuration_families[0]
    assert family["configuration_family"] == "2p(2)"
    assert family["source_csf_count"] == 2
    assert family["selected_csf_count"] == 1
    assert family["family_weight_all"] == pytest.approx(1)
    assert family["family_weight_selected"] == pytest.approx(0.64)


@pytest.mark.parametrize("mode", ["amplitude_cutoff", "weight_cutoff"])
def test_exact_boundary_and_nextafter(mode):
    values = np.array([0.5, np.nextafter(0.5, 0), -np.nextafter(0.5, 1), 0.5])
    doc, lengths, mixing = inputs([values])
    result = select_mr_csfs(
        doc,
        lengths,
        mixing,
        [MrTarget(0, 1)],
        **{mode: 0.5 if mode == "amplitude_cutoff" else 0.25},
    )
    assert [row["csf_index_in_block_1based"] for row in result.selected_rows] == [3]


def test_weight_union_is_not_cross_state_sum_or_cumulative():
    doc, lengths, mixing = inputs([[0.5] * 4, [0.5] * 4])
    with pytest.raises(ValueError, match="empty selection"):
        select_mr_csfs(
            doc, lengths, mixing, [MrTarget(0, 1), MrTarget(0, 4)], weight_cutoff=0.3
        )


@pytest.mark.parametrize("cutoff", [-1, 1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("mode", ["amplitude_cutoff", "weight_cutoff"])
def test_bad_cutoffs(cutoff, mode):
    with pytest.raises(ValueError, match="finite"):
        select_mr_csfs(*inputs(), [MrTarget(0, 1)], **{mode: cutoff})


@pytest.mark.parametrize(
    "kwargs", [{}, {"amplitude_cutoff": 0.1, "weight_cutoff": 0.1}]
)
def test_requires_one_cutoff(kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        select_mr_csfs(*inputs(), [MrTarget(0, 1)], **kwargs)


@pytest.mark.parametrize(
    "targets",
    [
        [],
        [MrTarget(0, 1)] * 2,
        [MrTarget(-1, 1)],
        [MrTarget(1, 1)],
        [MrTarget(0, 0)],
        [MrTarget(0, 5)],
    ],
)
def test_invalid_targets_do_not_fall_back(targets):
    with pytest.raises(ValueError):
        select_mr_csfs(*inputs(), targets, amplitude_cutoff=0.1)


def test_requested_zero_asf_block_fails():
    with pytest.raises(ValueError, match="unknown level"):
        select_mr_csfs(*inputs(empty_first=True), [MrTarget(0, 1)], amplitude_cutoff=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_even_in_unrequested_asf(value):
    doc, lengths, mixing = inputs()
    mixing.blocks[0].mix_coefficients[1, 0] = value
    with pytest.raises(ValueError, match="Non-finite"):
        select_mr_csfs(doc, lengths, mixing, [MrTarget(0, 1)], amplitude_cutoff=0)


def test_no_renormalization():
    with pytest.raises(ValueError, match="norm"):
        select_mr_csfs(*inputs([[0.1] * 4]), [MrTarget(0, 1)], amplitude_cutoff=0)


@pytest.mark.parametrize(
    "field,value,message",
    [("electron_count", 3, "electron"), ("orbital_count", 5, "orbital")],
)
def test_metadata_mismatch(field, value, message):
    doc, lengths, mixing = inputs()
    with pytest.raises(ValueError, match=message):
        select_mr_csfs(
            doc,
            lengths,
            replace(mixing, **{field: value}),
            [MrTarget(0, 1)],
            amplitude_cutoff=0,
        )


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("length", "dimensions"),
        ("block_id", "offsets"),
        ("j", "J/parity"),
        ("parity", "J/parity"),
        ("electrons", "electron"),
        ("occupation", "occupation"),
        ("block_order", "order"),
    ],
)
def test_pair_mismatch(mutation, message):
    doc, lengths, mixing = inputs()
    if mutation == "length":
        lengths = [3]
    elif mutation == "block_id":
        doc = replace(
            doc,
            csfs_df=doc.csfs_df.with_columns(
                pl.lit(1, dtype=pl.Int64).alias("block_id")
            ),
        )
    elif mutation in ("j", "parity"):
        doc = replace(
            doc,
            csfs_df=doc.csfs_df.with_columns(
                pl.lit("       1+" if mutation == "j" else "       0-").alias("line3")
            ),
        )
    elif mutation in ("electrons", "occupation"):
        doc = replace(
            doc,
            csfs_df=doc.csfs_df.with_columns(
                pl.lit("  2s ( 1)" if mutation == "electrons" else "  2s ( 3)").alias(
                    "line1"
                )
            ),
        )
    else:
        mixing = replace(mixing, blocks=[replace(mixing.blocks[0], block_index=1)])
    with pytest.raises(ValueError, match=message):
        select_mr_csfs(doc, lengths, mixing, [MrTarget(0, 1)], amplitude_cutoff=0)


def test_output_block_order_follows_source_not_request_order():
    doc, _, mixing = inputs()
    second = replace(mixing.blocks[0], block_index=1, j_value_location=5, j_value="2")
    second_frame = doc.csfs_df.with_columns(
        pl.lit(1, dtype=pl.Int64).alias("block_id"), pl.lit("       2+").alias("line3")
    )
    doc = replace(doc, csfs_df=pl.concat([doc.csfs_df, second_frame]))
    mixing = replace(mixing, blocks=[mixing.blocks[0], second])
    result = select_mr_csfs(
        doc, [4, 4], mixing, [MrTarget(1, 4), MrTarget(0, 1)], amplitude_cutoff=0.7
    )
    assert result.document.csfs_df["block_id"].to_list() == [0, 1]
    assert [row["source_row_index_0based"] for row in result.selected_rows] == [0, 7]
