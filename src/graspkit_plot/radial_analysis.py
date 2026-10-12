"""Static radial diagnostics; signed components always keep linear y axes."""

# Matplotlib plotting APIs expose partially typed kwargs and Axes arrays.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAny=false

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import SymmetricalLogLocator
import numpy as np

from graspkit.data_IO.loaders.radial_analysis_loader import RadialOrbital
from graspkit.grasp_data_extractor.radial_analysis import OrbitalAnalysis, PairAnalysis


def _radius_axis(ax: Axes, r: np.typing.NDArray[np.float64], log_radius: bool) -> None:
    # Fix limits after changing scale: linear autoscale margins would otherwise
    # create a large, physically meaningless negative-radius logarithmic range.
    if log_radius:
        threshold = max(float(r[r > 0][0]), float(r[-1]) * 1e-6)
        ax.set_xscale("symlog", linthresh=threshold)
        locator = SymmetricalLogLocator(base=10, linthresh=threshold)
        locator.set_params(numticks=7)
        ax.xaxis.set_major_locator(locator)
    ax.set_xlim(float(r[0]), float(r[-1]))
    ax.set_xlabel("r [bohr]")
    ax.grid(alpha=0.2)


def plot_orbital(
    path: Path,
    orbital: RadialOrbital,
    analysis: OrbitalAnalysis,
    title: str,
    *,
    log_radius: bool = True,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), layout="constrained")
    r = orbital.r
    for ax, y, name in zip(axes[0], (orbital.p, orbital.q), ("P", "Q")):
        ax.plot(r, y, lw=1)
        ax.set_ylabel(f"{name} [bohr$^{{-1/2}}$]")
        ax.axhline(0, color="0.7", lw=0.5)
    axes[1, 0].plot(r, analysis.w, lw=1)
    axes[1, 0].set_ylabel("raw w [bohr$^{-1}$]")
    axes[1, 1].plot(r, analysis.f, lw=1)
    axes[1, 1].set_ylabel("F (norm corrected)")
    for ax in axes[:2].flat:
        _radius_axis(ax, r, log_radius)
    for ax, y, component in zip(axes[2], (orbital.p, orbital.q), ("P", "Q")):
        nodes = analysis.nodes[component]
        # Overview of the region containing all internal zeros. Individual
        # local node windows are rendered separately below.
        ax.plot(r, y, lw=1)
        for event in nodes.events:
            ax.axvline(
                event.radius,
                color="#D55E00" if event.kind == "crossing" else "0.6",
                lw=0.6,
                alpha=0.6,
            )
        ax.set_ylabel(f"{component}; internal zero marks")
        _radius_axis(ax, r, log_radius)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_node_windows(
    path: Path, orbital: RadialOrbital, analysis: OrbitalAnalysis, title: str
) -> None:
    """Paginate every raw node; no undocumented selection of tail nodes."""
    windows = [
        (component, event)
        for component, nodes in analysis.nodes.items()
        for event in nodes.events
    ]
    for page_start in range(0, len(windows), 12):
        page = windows[page_start : page_start + 12]
        fig, axes = plt.subplots(
            (len(page) + 2) // 3,
            3,
            figsize=(12, 3 * ((len(page) + 2) // 3)),
            squeeze=False,
            layout="constrained",
        )
        for ax, (component, event) in zip(axes.flat, page):
            y = orbital.p if component == "P" else orbital.q
            low, high = max(0, event.left - 3), min(len(y), event.right + 4)
            ax.plot(orbital.r[low:high], y[low:high], ".-", lw=0.8, ms=3)
            ax.axhline(0, color="0.6", lw=0.5)
            ax.axvline(event.radius, color="#D55E00", lw=0.7)
            ax.set_title(f"{component}: {event.kind}, r={event.radius:.5g}", fontsize=9)
            ax.set_xlabel("r [bohr]")
            ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
            ax.grid(alpha=0.2)
        for ax in list(axes.flat)[len(page) :]:
            ax.set_visible(False)
        fig.suptitle(title + " — native-grid node windows")
        fig.savefig(path.with_name(f"{path.stem}_{page_start // 12 + 1}.png"), dpi=150)
        plt.close(fig)


def plot_pair(
    path: Path,
    pair: PairAnalysis,
    title: str,
    labels: tuple[str, str],
    *,
    log_radius: bool = True,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), layout="constrained")
    r = pair.r
    for ax, a, b, name in zip(
        axes[:2].flat,
        (pair.pa, pair.qa, pair.wa, pair.fa),
        (pair.pb, pair.qb, pair.wb, pair.fb),
        ("P", "Q", "w", "F"),
    ):
        ax.plot(r, a, label=labels[0], lw=1)
        ax.plot(r, b, label=labels[1], lw=1, ls="--")
        ax.set_ylabel(name + " (normalized common range)")
        ax.legend(fontsize=8)
    for ax, delta, name in zip(
        axes[2:].flat,
        (pair.pb - pair.pa, pair.qb - pair.qa, pair.wb - pair.wa, pair.fb - pair.fa),
        ("delta P", "delta Q", "delta w", "delta F"),
    ):
        ax.plot(r, delta, color="#009E73", lw=1)
        ax.set_ylabel(name + " (B - A)")
        ax.axhline(0, color="0.7", lw=0.5)
    for ax in axes.flat:
        _radius_axis(ax, r, log_radius)
    fig.suptitle(title + f"; phase B={pair.metrics['phase_b']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_matrix(
    path: Path,
    matrix: list[list[float]],
    rows: list[str],
    columns: list[str],
    title: str,
    *,
    residual: bool = False,
) -> None:
    data = np.asarray(matrix)
    if residual:
        data = data - np.eye(len(rows))
    vmax = max(float(np.max(np.abs(data))), 1e-16)
    fig, ax = plt.subplots(
        figsize=(max(5, len(columns) * 0.7), max(4, len(rows) * 0.65)),
        layout="constrained",
    )
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(columns)):
            ax.text(
                j,
                i,
                f"{data[i, j]:.2g}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(data[i, j]) > 0.6 * vmax else "black",
            )
    ax.set_title(title + ("; S - identity" if residual else ""))
    fig.colorbar(im, ax=ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
