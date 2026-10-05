import hashlib

import numpy as np

from .config import EnvConfig


def make_shape(rng: np.random.Generator, cfg: EnvConfig, h: int | None = None) -> np.ndarray:
    """Random target shape as a boolean (h_max, l_max) mask.

    Every helix is a single interval, and two consecutive helices share the end where the
    scaffold turns around (right end between helices 0-1, left end between 1-2, and so on).
    This guarantees that the serpentine scaffold routing exists.
    """
    if h is None:
        h = int(rng.integers(cfg.min_rows, cfg.h_max + 1))
    if not 1 <= h <= cfg.h_max:
        raise ValueError("h must be between 1 and h_max")

    start = int(rng.integers(0, cfg.l_max - cfg.min_row_len + 1))
    end = int(rng.integers(start + cfg.min_row_len, cfg.l_max + 1))
    rows = [(start, end)]
    for r in range(1, h):
        delta = int(rng.integers(-cfg.shape_step, cfg.shape_step + 1))
        if r % 2 == 1:  # turn on the right: keep the end, move the start
            start = int(np.clip(start + delta, 0, end - cfg.min_row_len))
        else:           # turn on the left: keep the start, move the end
            end = int(np.clip(end + delta, start + cfg.min_row_len, cfg.l_max))
        rows.append((start, end))

    shape = np.zeros((cfg.h_max, cfg.l_max), dtype=bool)
    for r, (a, b) in enumerate(rows):
        shape[r, a:b] = True
    return shape


def row_intervals(shape: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every non-empty helix, in row order. Raises if a row is not one interval."""
    out = []
    for r in range(shape.shape[0]):
        idx = np.flatnonzero(shape[r])
        if idx.size == 0:
            continue
        if idx[-1] - idx[0] + 1 != idx.size:
            raise ValueError(f"row {r} is not a single interval")
        out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def shape_id(shape: np.ndarray) -> str:
    """Stable identifier of a shape, to split train/test by shape rather than by seed."""
    return hashlib.sha1(np.ascontiguousarray(shape, dtype=np.uint8).tobytes()).hexdigest()[:12]
