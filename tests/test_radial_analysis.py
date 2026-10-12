from dataclasses import replace
from pathlib import Path
import struct

import numpy as np
import pytest

from graspkit.data_IO.loaders.radial_analysis_loader import (
    RadialOrbital,
    load_radial_input,
)
from graspkit.grasp_data_extractor.radial_analysis import (
    AnalysisOptions,
    analyze_nodes,
    analyze_orbital,
    compare_orbitals,
    node_tables,
    quadratic_integral,
)


def orbital(r, p, q=None):
    return RadialOrbital(
        "2s",
        2,
        -1,
        np.array(r, dtype=float),
        np.array(p, dtype=float),
        np.zeros(len(r)) if q is None else np.array(q, dtype=float),
        Path("fixture"),
    )


def test_native_nonuniform_dr_and_raw_norm_are_preserved():
    a = orbital([0, 0.2, 0.9, 2], [0, 2, 1, 0], [0, 0.4, 0.2, 0])
    before = a.p.copy()
    result = analyze_orbital(a, AnalysisOptions())
    expected = np.trapezoid(a.p**2 + a.q**2, a.r)
    assert result.metrics["raw_norm"] == pytest.approx(expected)
    assert result.metrics["norm_minus_one"] == pytest.approx(expected - 1)
    assert result.f[-1] == pytest.approx(1)
    assert result.metrics["mean_r"] == pytest.approx(
        np.trapezoid(a.r * (a.p**2 + a.q**2), a.r) / expected
    )
    np.testing.assert_array_equal(before, a.p)


@pytest.mark.parametrize("r", [[0, 0.1, 0.7, 2, 3], [0, 0.1, 0.7, 2]])
def test_irregular_quadratic_rule(r):
    r = np.array(r)
    # Quadratic is exact except for a final unmatched trapezoidal interval.
    expected = r[-1] ** 3 / 3
    if len(r) % 2 == 0:
        expected += (r[-1] - r[-2]) ** 3 / 6
    assert quadratic_integral(r, r**2) == pytest.approx(expected)


def test_zero_plateaus_touching_and_boundary_padding():
    r = np.arange(11, dtype=float)
    y = np.array([0, 2, 0, 0, -1, 0, -2, 0, 3, 0, 0], dtype=float)
    result = analyze_nodes(r, y, (0, 0.1))
    assert result.raw_count == 2
    assert [e.kind for e in result.events] == ["crossing", "touching_zero", "crossing"]
    assert [e.radius for e in result.events] == [2.5, 5, 7]
    assert len(result.transitions[0.1]) == 2


def test_filter_keeps_entire_lobes_and_original_node_locations():
    r = np.arange(9, dtype=float)
    y = np.array([0, 1, 0.001, -0.001, -1, -0.0001, 0.0001, -0.001, 0], dtype=float)
    result = analyze_nodes(r, y, (0, 0.01))
    assert result.raw_count == 3
    assert len(result.transitions[0.01]) == 1
    assert result.events[0].radius == 2.5
    rows, lobes = node_tables(r, result)
    assert (
        next(
            row
            for row in rows
            if row["record"] == "filtered" and row["threshold"] == 0.01
        )["r"]
        == 2.5
    )
    assert any(row["excluded"] and row["amplitude"] == 0.0001 for row in lobes)
    assert any(row["threshold_sensitive"] for row in rows)


def test_removed_lobes_do_not_fabricate_a_unique_node():
    r = np.arange(6, dtype=float)
    y = np.array([0, 1, -0.001, 0.001, -1, 0])
    result = analyze_nodes(r, y, (0, 0.01))
    assert result.transitions[0.01] == ((0, 1, 2),)
    rows, _ = node_tables(r, result)
    filtered = [
        row for row in rows if row["record"] == "filtered" and row["threshold"] == 0.01
    ]
    assert filtered[0]["r"] is None
    assert filtered[0]["kind"] == "ambiguous_span"


def test_all_zero_component_is_not_an_infinite_zero_plateau():
    result = analyze_nodes(np.arange(5, dtype=float), np.zeros(5), (0, 0.1))
    assert result.raw_count == 0
    assert not result.lobes
    assert result.transitions[0.1] == ()


def test_phase_alignment_flips_both_components_and_keeps_inputs():
    a = orbital([0, 0.1, 1, 2], [0, 1, 0.5, 0], [0, -0.2, 0.1, 0])
    b = replace(a, p=-3 * a.p, q=-3 * a.q)
    pair = compare_orbitals(a, b, AnalysisOptions())
    assert pair.metrics["phase_b"] == -1
    assert pair.metrics["abs_overlap"] == pytest.approx(1)
    assert pair.metrics["distance"] == pytest.approx(0, abs=3e-8)
    np.testing.assert_allclose(pair.pb, pair.pa)
    np.testing.assert_allclose(pair.qb, pair.qa)
    np.testing.assert_array_equal(b.p, -3 * a.p)


def test_interpolation_common_range_and_truncation_are_explicit():
    a = orbital([0, 1, 2, 3], [0, 1, 2, 1])
    b = orbital([0.5, 1.5, 2.5], [0.5, 1.5, 1.5])
    result = compare_orbitals(a, b, AnalysisOptions(core_radius=3))
    assert result.r[0] == 0.5 and result.r[-1] == 2.5
    assert result.metrics["range_restricted"] is True
    assert result.metrics["retained_norm_fraction_a"] < 1
    assert result.metrics["core_radius_in_common_range"] is False
    assert result.metrics["F_core_a_common"] is None
    assert result.metrics["interpolation_norm_change_b"] != 0


def test_padding_does_not_hide_a_large_endpoint():
    result = analyze_orbital(
        orbital([0, 1, 2, 3, 4], [0, 1, 0.9, 0, 0]),
        AnalysisOptions(tail_amplitude_tolerance=0.1),
    )
    assert result.metrics["endpoint_relative_amplitude"] == 0
    assert result.metrics["last_nonzero_relative_amplitude"] == 0.9
    assert "tail_amplitude_candidate" in result.metrics["flags"]
    assert (
        result.metrics["norm_without_terminal_zero_segment"]
        < result.metrics["raw_norm"]
    )


@pytest.mark.parametrize(
    "r,p",
    [([0, 1, 1], [0, 1, 0]), ([0, 2, 1], [0, 1, 0]), ([0, 1, 2], [0, float("nan"), 0])],
)
def test_invalid_grids_and_data_are_not_silently_repaired(r, p):
    with pytest.raises(ValueError):
        analyze_orbital(orbital(r, p), AnalysisOptions())


@pytest.mark.parametrize(
    "header",
    ["r(a.u),P(2p ),P(2p ),Q(2p )", "r(a.u),P(2p ),P(2p+),Q(2p )", "r(a.u),P(2p )"],
)
def test_csv_rejects_duplicate_or_incomplete_orbitals(tmp_path, header):
    path = tmp_path / "data.csv"
    path.write_text(
        header
        + "\n"
        + "\n".join(
            ",".join([str(r)] + ["1"] * (len(header.split(",")) - 1)) for r in range(3)
        )
    )
    with pytest.raises(ValueError):
        load_radial_input(path)


def test_csv_label_units_and_trailing_zeros(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "r(a.u),P(2p-),Q(2p-),P(2p ),Q(2p )\n0,0,0,0,0\n1,2,1,3,1\n2,0,0,0,0\n"
    )
    a, b = load_radial_input(path, radius_scale=2, amplitude_scale=0.5)
    assert (a.label, a.kappa, b.label, b.kappa) == ("2p-", 1, "2p", -2)
    np.testing.assert_array_equal(a.r, [0, 2, 4])
    np.testing.assert_array_equal(a.p, [0, 1, 0])
    assert "unknown" in a.provenance


def test_binary_adapter_preserves_each_native_grid(tmp_path):
    def record(payload):
        marker = struct.pack("i", len(payload))
        return marker + payload + marker

    content = record(b"G92RWF")
    for n, r in ((1, [0.0, 1.0, 3.0]), (2, [0.0, 0.5, 1.0, 2.0])):
        content += record(struct.pack("iidi", n, -1, -0.5, len(r)))
        content += record(
            struct.pack(f"{1 + 2 * len(r)}d", 0, *([1] * len(r)), *([0] * len(r)))
        )
        content += record(struct.pack(f"{len(r)}d", *r))
    path = tmp_path / "rwfn.out"
    path.write_bytes(content)
    a, b = load_radial_input(path)
    np.testing.assert_array_equal(a.r, [0, 1, 3])
    np.testing.assert_array_equal(b.r, [0, 0.5, 1, 2])


def test_hydrogenic_scale_and_nodes_against_analytic_values():
    r = np.concatenate(([0.0], np.geomspace(1e-8, 50, 6000)))
    one = orbital(r, 2 * r * np.exp(-r))
    two = orbital(r, r * (2 - r) * np.exp(-r / 2) / (2 * np.sqrt(2)))
    a, b = (
        analyze_orbital(one, AnalysisOptions()),
        analyze_orbital(two, AnalysisOptions()),
    )
    assert a.metrics["raw_norm"] == pytest.approx(1, abs=3e-6)
    assert a.metrics["mean_r"] == pytest.approx(1.5, abs=1e-5)
    assert b.metrics["mean_r"] == pytest.approx(6, abs=2e-5)
    assert b.metrics["P_filtered_nodes"] == 1
    assert b.nodes["P"].events[0].radius == pytest.approx(2, abs=1e-5)
    assert (
        abs(compare_orbitals(one, two, AnalysisOptions()).metrics["raw_overlap_common"])
        < 1e-8
    )
