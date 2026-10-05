from dataclasses import dataclass

import numpy as np

from .config import EnvConfig


@dataclass(frozen=True)
class Staple:
    """A two-domain staple. Both ends are inclusive.

    Domain 1 lies on helix `row0` between columns `col0` and `col_x`. The crossover is at
    column `col_x`. Domain 2 lies on the neighbouring helix (row0 + 1 if side == 1, else
    row0 - 1) between `col_x` and `col_end`. Length = |x_c - x_0| + |x_e - x_c| + 2.
    """

    row0: int
    col0: int
    col_x: int
    col_end: int
    side: int

    @property
    def row1(self) -> int:
        return self.row0 + 1 if self.side else self.row0 - 1

    @property
    def len1(self) -> int:
        return abs(self.col_x - self.col0) + 1

    @property
    def len2(self) -> int:
        return abs(self.col_end - self.col_x) + 1

    @property
    def length(self) -> int:
        return self.len1 + self.len2

    @property
    def span1(self) -> tuple[int, int, int]:
        """(row, first column, last column) of domain 1, columns inclusive and ordered."""
        return self.row0, min(self.col0, self.col_x), max(self.col0, self.col_x)

    @property
    def span2(self) -> tuple[int, int, int]:
        return self.row1, min(self.col_x, self.col_end), max(self.col_x, self.col_end)

    @classmethod
    def from_action(cls, action) -> "Staple":
        r0, c0, cx, ce, side = (int(a) for a in action)
        return cls(r0, c0, cx, ce, 1 if side else 0)

    def to_action(self) -> np.ndarray:
        return np.array([self.row0, self.col0, self.col_x, self.col_end, self.side], dtype=np.int64)


def free_from_obs(obs: np.ndarray) -> np.ndarray:
    """Cells that belong to the shape and are not covered yet."""
    return (obs[0] > 0.5) & (obs[1] < 0.5)


def is_legal(staple: Staple, free: np.ndarray, cfg: EnvConfig) -> bool:
    h, l = free.shape
    r0, r1 = staple.row0, staple.row1
    if not (0 <= r0 < h and 0 <= r1 < h):
        return False
    if not all(0 <= c < l for c in (staple.col0, staple.col_x, staple.col_end)):
        return False
    if not cfg.l_staple_min <= staple.length <= cfg.l_staple_max:
        return False
    if staple.len1 < cfg.l_domain_min or staple.len2 < cfg.l_domain_min:
        return False
    if staple.col_x % cfg.crossover_period != 0:
        return False
    ra, a0, a1 = staple.span1
    rb, b0, b1 = staple.span2
    return bool(free[ra, a0:a1 + 1].all() and free[rb, b0:b1 + 1].all())


def free_runs(free: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """left[r, c] (right[r, c]): number of consecutive free cells ending (starting) at c, c included."""
    h, l = free.shape
    left = np.zeros((h, l), dtype=np.int32)
    right = np.zeros((h, l), dtype=np.int32)
    for c in range(l):
        prev = left[:, c - 1] if c > 0 else 0
        left[:, c] = (prev + 1) * free[:, c]
    for c in range(l - 1, -1, -1):
        nxt = right[:, c + 1] if c < l - 1 else 0
        right[:, c] = (nxt + 1) * free[:, c]
    return left, right


def feasible_crossovers(free: np.ndarray, cfg: EnvConfig) -> np.ndarray:
    """Boolean (h - 1, l) array: ok[r, x] is True iff a legal staple with its crossover at
    column x between helices r and r + 1 exists.

    The longest domain that fits on a helix from x is the longest free run through x in one
    direction, so a legal staple exists iff both reaches are at least l_domain_min and they
    add up to at least total_min.
    """
    left, right = free_runs(free)
    reach = np.maximum(left, right)
    cols = np.arange(free.shape[1])
    ok = free[:-1] & free[1:]
    ok &= (reach[:-1] >= cfg.l_domain_min) & (reach[1:] >= cfg.l_domain_min)
    ok &= (reach[:-1] + reach[1:] >= cfg.total_min)
    ok &= (cols % cfg.crossover_period == 0)[None, :]
    return ok


def has_legal_action(free: np.ndarray, cfg: EnvConfig) -> bool:
    return bool(feasible_crossovers(free, cfg).any())


def cover(covered: np.ndarray, staple: Staple) -> None:
    for r, a, b in (staple.span1, staple.span2):
        covered[r, a:b + 1] = True
