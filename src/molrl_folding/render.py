from typing import Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .design import Staple, cover

SCAFFOLD = "#888780"
STAPLE = "#1D9E75"
LAST = "#EF9F27"
UNCOVERED = "#D85A30"
OFFSET = 0.3  # distance, in helix spacings, between a scaffold line and the staple domain next to it


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every run of True in a 1D boolean array."""
    edges = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def draw_design(
    shape: np.ndarray,
    staples: Sequence[Staple],
    title: str | None = None,
    dpi: int = 100,
    highlight_last: bool = True,
    show_legend: bool = True,
) -> Figure:
    """Top view of a design.

    Each helix is a row. The scaffold (grey) runs along every helix and turns around at the
    ends. Each staple (green) sits next to the scaffold, with one domain on each of two
    neighbouring helices joined by a short bar, the crossover. Stretches of scaffold that no
    staple covers yet have an orange-red core. The last staple placed is highlighted.
    """
    shape = shape.astype(bool)
    covered = np.zeros_like(shape)
    for st in staples:
        cover(covered, st)

    rows = np.flatnonzero(shape.any(axis=1))
    cols = np.flatnonzero(shape.any(axis=0))
    r_lo, r_hi, c_lo, c_hi = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
    n_rows, n_cols = r_hi - r_lo + 1, c_hi - c_lo + 1

    width = max(6.5, 0.1 * n_cols + 2.0)
    legend_cols = 4 if width >= 8.5 else 2
    margin = (1.2 + (0.35 if legend_cols == 4 else 0.6) + 0.3) if show_legend else 1.2
    height = 0.5 * n_rows + margin
    fig = Figure(figsize=(width, height), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    # half-width of the scaffold U-turns, in nt, chosen so that they look round
    rx = max(1.0, 0.5 * ((height - margin) / n_rows) / ((width - 1.4) / (n_cols + 8)))

    runs = {r: _runs(shape[r]) for r in range(r_lo, r_hi + 1)}

    for r, row_runs in runs.items():
        for a, b in row_runs:
            ax.add_patch(Rectangle((a - 0.5, r - 0.5), b - a, 1.0, facecolor=SCAFFOLD, alpha=0.10, edgecolor="none"))
            ax.plot([a - 0.5, b - 0.5], [r, r], color=SCAFFOLD, lw=4.5, solid_capstyle="butt", zorder=2)
        for a, b in _runs(shape[r] & ~covered[r]):
            ax.plot([a - 0.5, b - 0.5], [r, r], color=UNCOVERED, lw=2.0, solid_capstyle="butt", zorder=3)

    t = np.linspace(0, np.pi, 30)
    for r in range(r_lo, r_hi):
        if len(runs[r]) != 1 or len(runs[r + 1]) != 1:
            continue
        (a0, b0), (a1, b1) = runs[r][0], runs[r + 1][0]
        if r % 2 == 0 and b0 == b1:    # scaffold turns on the right
            x, sign = b0 - 0.5, 1
        elif r % 2 == 1 and a0 == a1:  # scaffold turns on the left
            x, sign = a0 - 0.5, -1
        else:
            continue
        ax.plot(x + sign * rx * np.sin(t), r + 0.5 - 0.5 * np.cos(t), color=SCAFFOLD, lw=4.5, solid_capstyle="round", zorder=2)

    for i, st in enumerate(staples):
        is_last = highlight_last and i == len(staples) - 1
        colour, lw = (LAST, 3.0) if is_last else (STAPLE, 2.2)
        inward = OFFSET if st.row1 > st.row0 else -OFFSET
        ra, a0, a1 = st.span1
        rb, b0, b1 = st.span2
        ya, yb = ra + inward, rb - inward
        for y, lo, hi in ((ya, a0, a1), (yb, b0, b1)):
            ax.plot([lo - 0.4, hi + 0.4], [y, y], color=colour, lw=lw, solid_capstyle="round", zorder=4)
        ax.plot([st.col_x, st.col_x], [ya, yb], color=colour, lw=lw, solid_capstyle="round", zorder=4)
        if is_last:
            ax.scatter([st.col_x], [(ya + yb) / 2], s=40, color=LAST, zorder=5)
            ax.scatter([st.col0, st.col_end], [ya, yb], s=45, facecolors="none", edgecolors=LAST, linewidths=1.5, zorder=5)

    ax.set_xlim(c_lo - 0.5 - rx - 0.5, c_hi + 0.5 + rx + 0.5)
    ax.set_ylim(r_hi + 0.6, r_lo - 0.6)
    ax.set_yticks(range(r_lo, r_hi + 1))
    ax.set_yticklabels([f"h{r}" for r in range(r_lo, r_hi + 1)])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("position along the helix (nt)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)

    if show_legend:
        handles = [
            Line2D([0], [0], color=SCAFFOLD, lw=4.5, label="scaffold"),
            Line2D([0], [0], color=UNCOVERED, lw=2.0, label="scaffold not covered yet"),
            Line2D([0], [0], color=STAPLE, lw=2.2, label="staples"),
        ]
        if highlight_last and staples:
            handles.append(Line2D([0], [0], color=LAST, lw=3.0, label="last staple placed"))
        fig.legend(handles=handles, loc="lower center", ncol=legend_cols, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, (0.9 if legend_cols == 2 else 0.6) / height if show_legend else 0, 1, 1))
    return fig


def figure_to_array(fig: Figure) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()