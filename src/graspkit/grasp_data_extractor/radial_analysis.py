"""Native-grid radial diagnostics and explicitly bounded comparisons.

All primary integrals are trapezoidal in actual r (never multiplied by r**2).
Node filtering removes complete same-sign lobes, not low-amplitude samples.
This is not a claim to reproduce a particular GRASP runtime COUNT/THRESH.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from graspkit.data_IO.loaders.radial_analysis_loader import FloatArray, RadialOrbital

Scalar = str | float | int | bool | None
Row = dict[str, Scalar]


@dataclass(frozen=True)
class AnalysisOptions:
    node_thresholds: tuple[float, ...] = (1e-8, 1e-6, 1e-4)
    primary_threshold: float = 1e-6
    norm_tolerance: float | None = None
    orthogonality_tolerance: float | None = None
    tail_amplitude_tolerance: float | None = None
    shape_tolerance: float | None = None
    low_overlap: float = 0.8
    tail_radius_fraction: float = 0.1
    core_radius: float | None = None
    interpolation: Literal["linear", "none"] = "linear"
    normalization_domain: Literal["common_range", "require_full_range"] = "common_range"

    def __post_init__(self) -> None:
        thresholds = (*self.node_thresholds, self.primary_threshold)
        if not self.node_thresholds or any(
            not math.isfinite(v) or not 0 <= v < 1 for v in thresholds
        ):
            raise ValueError("Node thresholds must be finite, in [0, 1)")
        if self.primary_threshold not in self.node_thresholds:
            raise ValueError("primary_threshold must be in node_thresholds")
        if self.interpolation not in (
            "linear",
            "none",
        ) or self.normalization_domain not in ("common_range", "require_full_range"):
            raise ValueError("Unsupported interpolation or normalization domain")
        for value in (
            self.norm_tolerance,
            self.orthogonality_tolerance,
            self.tail_amplitude_tolerance,
            self.shape_tolerance,
            self.core_radius,
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(
                    "Tolerances and core radius must be finite and non-negative"
                )
        if not 0 < self.tail_radius_fraction < 1 or not 0 <= self.low_overlap <= 1:
            raise ValueError("Invalid tail fraction or low-overlap threshold")


def integral(r: FloatArray, values: FloatArray) -> float:
    return float(np.trapezoid(values, r))


def cumulative(r: FloatArray, density: FloatArray) -> FloatArray:
    return np.concatenate(
        (np.zeros(1), np.cumsum(np.diff(r) * (density[1:] + density[:-1]) / 2))
    )


def quadratic_integral(r: FloatArray, values: FloatArray) -> float:
    """Independent composite irregular-grid quadratic diagnostic.

    A final unmatched interval uses a trapezoid. Negative weights are possible
    on strongly unequal grids; this is a sensitivity check, not ground truth.
    """
    stop = len(r) - 2
    h0 = r[1 : stop + 1 : 2] - r[:stop:2]
    h1 = r[2 : stop + 2 : 2] - r[1 : stop + 1 : 2]
    width = h0 + h1
    value = float(
        np.sum(
            width
            / 6
            * (
                (2 - h1 / h0) * values[:stop:2]
                + width**2 / (h0 * h1) * values[1 : stop + 1 : 2]
                + (2 - h0 / h1) * values[2 : stop + 2 : 2]
            )
        )
    )
    if len(r) % 2 == 0:
        value += integral(r[-2:], values[-2:])
    return value


@dataclass(frozen=True)
class Lobe:
    start: int
    end: int
    sign: int
    peak_index: int
    amplitude: float
    relative_amplitude: float


@dataclass(frozen=True)
class ZeroEvent:
    left: int
    right: int
    radius: float
    kind: str
    lower: float
    upper: float


@dataclass(frozen=True)
class NodeAnalysis:
    lobes: tuple[Lobe, ...]
    events: tuple[ZeroEvent, ...]
    # Each filtered transition refers to actual raw crossings. Several IDs
    # mean a bracket after removing lobes, not a newly invented zero location.
    transitions: dict[float, tuple[tuple[int, ...], ...]]
    retained_lobes: dict[float, tuple[int, ...]]

    @property
    def raw_count(self) -> int:
        return sum(event.kind == "crossing" for event in self.events)


def analyze_nodes(
    r: FloatArray, y: FloatArray, thresholds: tuple[float, ...]
) -> NodeAnalysis:
    """Deduplicate zero plateaus, omit external zeros, retain touching zeros."""
    nonzero = np.flatnonzero(y != 0)
    maximum = float(np.max(np.abs(y)))
    events: list[ZeroEvent] = []
    lobes: list[Lobe] = []
    if len(nonzero):
        start = previous = int(nonzero[0])
        for value in nonzero[1:]:
            index = int(value)
            opposite = bool(np.signbit(y[index]) != np.signbit(y[previous]))
            gap = index > previous + 1
            if opposite or gap:
                segment = np.abs(y[start : previous + 1])
                peak = start + int(np.argmax(segment))
                lobes.append(
                    Lobe(
                        start,
                        previous,
                        1 if y[start] > 0 else -1,
                        peak,
                        float(abs(y[peak])),
                        float(abs(y[peak]) / maximum),
                    )
                )
                if gap:
                    lower, upper = float(r[previous + 1]), float(r[index - 1])
                    radius = (lower + upper) / 2
                else:
                    lower, upper = float(r[previous]), float(r[index])
                    radius = lower + (upper - lower) * float(
                        abs(y[previous]) / (abs(y[previous]) + abs(y[index]))
                    )
                events.append(
                    ZeroEvent(
                        previous,
                        index,
                        radius,
                        "crossing" if opposite else "touching_zero",
                        lower,
                        upper,
                    )
                )
                start = index
            previous = index
        peak = start + int(np.argmax(np.abs(y[start : previous + 1])))
        lobes.append(
            Lobe(
                start,
                previous,
                1 if y[start] > 0 else -1,
                peak,
                float(abs(y[peak])),
                float(abs(y[peak]) / maximum),
            )
        )
    retained: dict[float, tuple[int, ...]] = {}
    transitions: dict[float, tuple[tuple[int, ...], ...]] = {}
    for threshold in thresholds:
        keep = tuple(
            i for i, lobe in enumerate(lobes) if lobe.relative_amplitude >= threshold
        )
        retained[threshold] = keep
        transitions[threshold] = tuple(
            tuple(
                i
                for i, event in enumerate(events)
                if event.kind == "crossing"
                and lobes[left].end <= event.left
                and event.right <= lobes[right].start
            )
            for left, right in zip(keep, keep[1:])
            if lobes[left].sign != lobes[right].sign
        )
    return NodeAnalysis(tuple(lobes), tuple(events), transitions, retained)


def node_tables(r: FloatArray, nodes: NodeAnalysis) -> tuple[list[Row], list[Row]]:
    rows: list[Row] = []
    lobes: list[Row] = []
    for index, event in enumerate(nodes.events):
        states = {
            threshold: (
                "retained"
                if (index,) in transitions
                else "ambiguous_span"
                if any(index in group for group in transitions)
                else "removed"
            )
            for threshold, transitions in nodes.transitions.items()
        }
        for threshold, state in states.items():
            rows.append(
                {
                    "record": "raw",
                    "node_id": index,
                    "kind": event.kind,
                    "r": event.radius,
                    "r_lower": event.lower,
                    "r_upper": event.upper,
                    "threshold": threshold,
                    "status": state
                    if event.kind == "crossing"
                    else "not_a_sign_change",
                    "threshold_sensitive": len(set(states.values())) > 1,
                    "raw_ids": str(index),
                }
            )
    for threshold, transitions in nodes.transitions.items():
        for index, group in enumerate(transitions):
            first, last = nodes.events[group[0]], nodes.events[group[-1]]
            rows.append(
                {
                    "record": "filtered",
                    "node_id": index,
                    "kind": "crossing" if len(group) == 1 else "ambiguous_span",
                    "r": first.radius if len(group) == 1 else None,
                    "r_lower": first.lower,
                    "r_upper": last.upper,
                    "threshold": threshold,
                    "status": "retained",
                    "threshold_sensitive": any(
                        group not in other for other in nodes.transitions.values()
                    ),
                    "raw_ids": ";".join(map(str, group)),
                }
            )
        for index, lobe in enumerate(nodes.lobes):
            lobes.append(
                {
                    "lobe_id": index,
                    "r_start": float(r[lobe.start]),
                    "r_end": float(r[lobe.end]),
                    "r_peak": float(r[lobe.peak_index]),
                    "sign": lobe.sign,
                    "amplitude": lobe.amplitude,
                    "relative_amplitude": lobe.relative_amplitude,
                    "threshold": threshold,
                    "excluded": index not in nodes.retained_lobes[threshold],
                }
            )
    return rows, lobes


@dataclass(frozen=True)
class OrbitalAnalysis:
    metrics: Row
    w: FloatArray
    f: FloatArray
    nodes: dict[str, NodeAnalysis]
    shape_intervals: list[Row]


def analyze_orbital(
    orbital: RadialOrbital, options: AnalysisOptions
) -> OrbitalAnalysis:
    orbital.validate()
    r, p, q = orbital.r, orbital.p, orbital.q
    w = p**2 + q**2
    norm = integral(r, w)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError(f"{orbital.label}: non-positive or non-finite raw norm")
    f = cumulative(r, w) / norm
    active = np.flatnonzero(w > 0)
    last, first = int(active[-1]), int(active[0])
    amplitude = np.sqrt(w)
    maximum = float(np.max(amplitude))
    trailing = len(r) - last - 1
    flags: list[str] = []
    if trailing:
        flags.append("terminal_zeros_provenance_unknown")
    if r[0] > 0:
        flags.append("origin_not_sampled")
    elif w[0] != 0:
        flags.append("nonzero_at_origin")
    endpoint = float(amplitude[-1] / maximum)
    before_zeros = float(amplitude[last] / maximum)
    if options.norm_tolerance is not None and abs(norm - 1) > options.norm_tolerance:
        flags.append("raw_norm_outside_configured_tolerance")
    if (
        options.tail_amplitude_tolerance is not None
        and before_zeros > options.tail_amplitude_tolerance
    ):
        flags.append("tail_amplitude_candidate")
    moments: Row = {
        "raw_norm": norm,
        "norm_minus_one": norm - 1,
        "quadratic_norm_diagnostic": quadratic_integral(r, w),
        "quadrature_difference": quadratic_integral(r, w) - norm,
        "mean_r": integral(r, r * w) / norm,
        "rms_r": math.sqrt(integral(r, r**2 * w) / norm),
        "peak_r": float(r[np.argmax(w)]),
        "points": len(r),
        "finite_values": True,
        "strictly_increasing_grid": True,
        "duplicate_grid_points": 0,
        "r_min": float(r[0]),
        "r_max": float(r[-1]),
        "nonzero_r_min": float(r[first]),
        "last_nonzero_r": float(r[last]),
        "trailing_zero_points": trailing,
        "leading_zero_points": first,
        "endpoint_relative_amplitude": endpoint,
        "last_nonzero_relative_amplitude": before_zeros,
        "terminal_zero_start_r": float(r[last + 1]) if trailing else None,
        "norm_without_terminal_zero_segment": integral(r[: last + 1], w[: last + 1]),
        "tail_window_start_r": float(r[last] * (1 - options.tail_radius_fraction)),
        "tail_window_probability": 1.0
        - float(np.interp(float(r[last]) * (1 - options.tail_radius_fraction), r, f)),
        "r_50": float(np.interp(0.5, f, r)),
        "r_90": float(np.interp(0.9, f, r)),
        "r_99": float(np.interp(0.99, f, r)),
    }
    nodes: dict[str, NodeAnalysis] = {}
    shape: list[Row] = []
    for component, y in (("P", p), ("Q", q)):
        nodes[component] = analyze_nodes(r, y, options.node_thresholds)
        counts = [len(v) for v in nodes[component].transitions.values()]
        moments[f"{component}_raw_nodes"] = nodes[component].raw_count
        moments[f"{component}_filtered_nodes"] = len(
            nodes[component].transitions[options.primary_threshold]
        )
        moments[f"{component}_node_count_min"] = min(counts)
        moments[f"{component}_node_count_max"] = max(counts)
        if min(counts) != max(counts):
            flags.append(f"{component}_threshold_sensitive")
        scale = float(np.max(np.abs(y)))
        moments[f"{component}_total_variation_relative"] = (
            float(np.sum(np.abs(np.diff(y)))) / scale if scale else 0.0
        )
        moments[f"{component}_turning_points"] = int(
            np.count_nonzero(np.diff(y)[:-1] * np.diff(y)[1:] < 0)
        )
        if scale == 0:
            moments[f"{component}_shape_score"] = 0.0
            continue
        # Residual against the straight line through immediate neighbours on
        # the actual nonuniform grid. Rank, do not diagnose a physical defect.
        predicted = y[:-2] + (y[2:] - y[:-2]) * (r[1:-1] - r[:-2]) / (r[2:] - r[:-2])
        scores = np.abs(y[1:-1] - predicted) / scale
        moments[f"{component}_shape_score"] = float(scores.max())
        if (
            options.shape_tolerance is not None
            and scores.max() > options.shape_tolerance
        ):
            flags.append(f"{component}_shape_candidate")
        for i in np.argsort(scores)[-5:][::-1]:
            j = int(i) + 1
            shape.append(
                {
                    "component": component,
                    "r_start": float(r[j - 1]),
                    "r": float(r[j]),
                    "r_end": float(r[j + 1]),
                    "linear_residual_relative": float(scores[i]),
                    "turning_point": bool((y[j] - y[j - 1]) * (y[j + 1] - y[j]) < 0),
                }
            )
    moments["flags"] = ";".join(flags)
    return OrbitalAnalysis(moments, w, f, nodes, shape)


@dataclass(frozen=True)
class PairAnalysis:
    metrics: Row
    r: FloatArray
    pa: FloatArray
    qa: FloatArray
    pb: FloatArray
    qb: FloatArray
    wa: FloatArray
    wb: FloatArray
    fa: FloatArray
    fb: FloatArray


def common_samples(
    a: RadialOrbital, b: RadialOrbital
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    lower, upper = max(a.r[0], b.r[0]), min(a.r[-1], b.r[-1])
    if lower >= upper:
        raise ValueError("No common radial interval")
    grid = np.union1d(
        a.r[(a.r >= lower) & (a.r <= upper)], b.r[(b.r >= lower) & (b.r <= upper)]
    )
    return (
        grid,
        np.interp(grid, a.r, a.p),
        np.interp(grid, a.r, a.q),
        np.interp(grid, b.r, b.p),
        np.interp(grid, b.r, b.q),
    )


def clipped_native_norm(orbital: RadialOrbital, lower: float, upper: float) -> float:
    r = np.union1d(
        orbital.r[(orbital.r >= lower) & (orbital.r <= upper)], [lower, upper]
    )
    p, q = np.interp(r, orbital.r, orbital.p), np.interp(r, orbital.r, orbital.q)
    return integral(r, p**2 + q**2)


def compare_orbitals(
    a: RadialOrbital, b: RadialOrbital, options: AnalysisOptions
) -> PairAnalysis:
    """Normalize analysis copies on the common range; flip BOTH components.

    Restricted-domain comparisons must not be read as full-domain agreement.
    Native and before/after interpolation norms and retained mass are exposed.
    """
    if a.kappa != b.kappa:
        raise ValueError("Radial comparison requires equal kappa")
    a.validate()
    b.validate()
    if options.interpolation == "none" and not np.array_equal(a.r, b.r):
        raise ValueError("Different grids but interpolation=none")
    if options.normalization_domain == "require_full_range" and (
        a.r[0] != b.r[0] or a.r[-1] != b.r[-1]
    ):
        raise ValueError(
            "Different radial bounds but normalization_domain=require_full_range"
        )
    r, pa, qa, pb, qb = common_samples(a, b)
    na, nb = integral(r, pa**2 + qa**2), integral(r, pb**2 + qb**2)
    if not all(math.isfinite(v) and v > 0 for v in (na, nb)):
        raise ValueError("Non-positive norm on common radial interval")
    native_a, native_b = integral(a.r, a.p**2 + a.q**2), integral(b.r, b.p**2 + b.q**2)
    clipped_a = clipped_native_norm(a, float(r[0]), float(r[-1]))
    clipped_b = clipped_native_norm(b, float(r[0]), float(r[-1]))
    raw_overlap = integral(r, pa * pb + qa * qb)
    pa, qa, pb, qb = (
        pa / math.sqrt(na),
        qa / math.sqrt(na),
        pb / math.sqrt(nb),
        qb / math.sqrt(nb),
    )
    overlap = integral(r, pa * pb + qa * qb)
    phase = -1 if overlap < 0 else 1
    pb, qb = phase * pb, phase * qb
    wa, wb = pa**2 + qa**2, pb**2 + qb**2
    fa, fb = cumulative(r, wa), cumulative(r, wb)
    absolute = min(1.0, abs(overlap))
    metrics: Row = {
        "same_grid": bool(np.array_equal(a.r, b.r)),
        "r_min_common": float(r[0]),
        "r_max_common": float(r[-1]),
        "common_points": len(r),
        "interpolation": "linear_in_r_union_grid_no_extrapolation",
        "normalization": "analysis_copies_common_range",
        "raw_overlap_common": raw_overlap,
        "signed_overlap": overlap,
        "abs_overlap": abs(overlap),
        "distance": math.sqrt(2 - 2 * absolute),
        "phase_b": phase,
        "raw_norm_a": native_a,
        "raw_norm_b": native_b,
        "clipped_native_norm_a": clipped_a,
        "clipped_native_norm_b": clipped_b,
        "interpolated_norm_a": na,
        "interpolated_norm_b": nb,
        "interpolation_norm_change_a": na - clipped_a,
        "interpolation_norm_change_b": nb - clipped_b,
        "retained_norm_fraction_a": clipped_a / native_a,
        "retained_norm_fraction_b": clipped_b / native_b,
        "delta_mean_r_common": integral(r, r * wb) - integral(r, r * wa),
        "delta_rms_r_common": math.sqrt(integral(r, r**2 * wb))
        - math.sqrt(integral(r, r**2 * wa)),
        "delta_peak_r_common": float(r[np.argmax(wb)] - r[np.argmax(wa)]),
        "max_abs_delta_F": float(np.max(np.abs(fb - fa))),
        "max_delta_F_r": float(r[np.argmax(np.abs(fb - fa))]),
        "max_abs_delta_P_r": float(r[np.argmax(np.abs(pb - pa))]),
        "max_abs_delta_Q_r": float(r[np.argmax(np.abs(qb - qa))]),
        "max_abs_delta_w_r": float(r[np.argmax(np.abs(wb - wa))]),
        "low_overlap_candidate": abs(overlap) < options.low_overlap,
        "range_restricted": bool(a.r[0] != b.r[0] or a.r[-1] != b.r[-1]),
    }
    if options.core_radius is not None:
        rc = options.core_radius
        within = bool(r[0] <= rc <= r[-1])
        metrics.update(
            {
                "core_radius": rc,
                "core_radius_in_common_range": within,
                "F_core_a_common": float(np.interp(rc, r, fa)) if within else None,
                "F_core_b_common": float(np.interp(rc, r, fb)) if within else None,
            }
        )
    return PairAnalysis(metrics, r, pa, qa, pb, qb, wa, wb, fa, fb)
