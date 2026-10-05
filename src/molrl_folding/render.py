from typing import Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from .design import Staple

_PALETTE = ["#1D9E75", "#378ADD", "#D85A30", "#7F77DD", "#BA7517", "#D4537E", "#639922", "#888780"]


def draw_design(shape: np.ndarray, staples: Sequence[Staple], title: str | None = None, dpi: int = 100) -> Figure:
    """Top view of a design: grey = target shape, one colour per staple, bar = crossover."""
    rows = np.flatnonzero(shape.any(axis=1))
    cols = np.flatnonzero(shape.any(axis=0))
    r_lo, r_hi, c_lo, c_hi = rows.min(), rows.max(), cols.min(), cols.max()

    width = max(5.0, 0.1 * (c_hi - c_lo + 1) + 1.0)
    height = max(2.2, 0.38 * (r_hi - r_lo + 1) + 1.2)
    fig = Figure(figsize=(width, height), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    ax.imshow(
        shape[r_lo:r_hi + 1, c_lo:c_hi + 1],
        cmap=ListedColormap(["white", "#E4E2DA"]), vmin=0, vmax=1, interpolation="nearest",
        extent=(c_lo - 0.5, c_hi + 0.5, r_hi + 0.5, r_lo - 0.5), aspect="auto",
    )
    for i, st in enumerate(staples):
        colour = _PALETTE[i % len(_PALETTE)]
        inward = 0.18 if st.row1 > st.row0 else -0.18
        ra, a0, a1 = st.span1
        rb, b0, b1 = st.span2
        ya, yb = ra + inward, rb - inward
        ax.plot([a0 - 0.4, a1 + 0.4], [ya, ya], color=colour, lw=2.2, solid_capstyle="round")
        ax.plot([b0 - 0.4, b1 + 0.4], [yb, yb], color=colour, lw=2.2, solid_capstyle="round")
        ax.plot([st.col_x, st.col_x], [ya, yb], color=colour, lw=2.2, solid_capstyle="round")

    ax.set_xlim(c_lo - 0.5, c_hi + 0.5)
    ax.set_ylim(r_hi + 0.5, r_lo - 0.5)
    ax.set_xlabel("position along the helix (nt)")
    ax.set_ylabel("helix")
    ax.set_yticks(range(int(r_lo), int(r_hi) + 1))
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def figure_to_array(fig: Figure) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
