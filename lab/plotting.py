"""Shared matplotlib style for the lab's static plots.

Rules followed (from a data-viz checklist): categorical colours in a fixed
order (never cycled by rank), thin 2px lines, recessive grid and axes, text in
neutral ink rather than series colours, one y-axis per chart, a legend whenever
there are two or more series.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "legend.frameon": False, "legend.fontsize": 8, "font.size": 9, "text.color": INK,
    })


def color(i: int) -> str:
    return SERIES[i % len(SERIES)]
