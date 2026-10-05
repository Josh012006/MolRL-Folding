"""Reference policies. A policy maps (obs, cfg, rng) to an action array, or None if no legal action exists."""

import numpy as np

from .config import EnvConfig
from .design import Staple, feasible_crossovers, free_from_obs, free_runs


def random_legal(obs: np.ndarray, cfg: EnvConfig, rng: np.random.Generator) -> np.ndarray | None:
    """Uniformly pick a feasible crossover, then random domain lengths and directions."""
    free = free_from_obs(obs)
    ok = feasible_crossovers(free, cfg)
    cand = np.argwhere(ok)
    if cand.size == 0:
        return None
    r, cx = cand[rng.integers(len(cand))]
    r_a, r_b = (r, r + 1) if rng.random() < 0.5 else (r + 1, r)

    left, right = free_runs(free)
    reach_a = max(left[r_a, cx], right[r_a, cx])
    reach_b = max(left[r_b, cx], right[r_b, cx])

    total = int(rng.integers(cfg.total_min, min(cfg.l_staple_max, reach_a + reach_b) + 1))
    len_lo = max(cfg.l_domain_min, total - reach_b)
    len_hi = min(reach_a, total - cfg.l_domain_min)
    len1 = int(rng.integers(len_lo, len_hi + 1))
    len2 = total - len1

    def pick_end(row: int, length: int) -> int:
        options = []
        if left[row, cx] >= length:
            options.append(cx - (length - 1))
        if right[row, cx] >= length:
            options.append(cx + (length - 1))
        return int(options[rng.integers(len(options))])

    side = 1 if r_b == r_a + 1 else 0
    return Staple(int(r_a), pick_end(r_a, len1), int(cx), pick_end(r_b, len2), side).to_action()


def _cover_cell(free, left, right, cfg: EnvConfig, r: int, c: int, side: int, target_len: int) -> Staple | None:
    """Best staple whose first domain starts at cell (r, c) and extends to the right."""
    r1 = r + 1 if side else r - 1
    if not 0 <= r1 < free.shape[0]:
        return None
    max_len1 = min(int(right[r, c]), cfg.l_staple_max - cfg.l_domain_min)
    if max_len1 < cfg.l_domain_min:
        return None

    len1 = np.arange(cfg.l_domain_min, max_len1 + 1)
    cx = c + len1 - 1
    usable = free[r1, cx] & (cx % cfg.crossover_period == 0)
    best, best_score = None, None
    for direction, run in ((-1, left), (1, right)):
        lo = np.maximum(cfg.l_domain_min, cfg.total_min - len1)
        hi = np.minimum(run[r1, cx], cfg.l_staple_max - len1)
        feasible = usable & (hi >= lo)
        if not feasible.any():
            continue
        len2 = np.clip(target_len - len1, lo, np.maximum(hi, lo))
        score = np.abs(len1 + len2 - target_len) * 1000 + np.abs(len1 - len2)
        score = np.where(feasible, score, np.iinfo(np.int64).max)
        i = int(np.argmin(score))
        if best_score is None or score[i] < best_score:
            best_score = score[i]
            best = Staple(r, c, int(cx[i]), int(cx[i]) + direction * (int(len2[i]) - 1), side)
    return best


def greedy_sweep(obs: np.ndarray, cfg: EnvConfig, rng: np.random.Generator, target_len: int = 40) -> np.ndarray | None:
    """Scan the free cells in reading order and cover the first one that can be covered,
    with a staple whose total length is as close as possible to target_len."""
    free = free_from_obs(obs)
    left, right = free_runs(free)
    target_len = int(np.clip(target_len, cfg.total_min, cfg.l_staple_max))
    for r, c in np.argwhere(free):
        for side in (1, 0):
            staple = _cover_cell(free, left, right, cfg, int(r), int(c), side, target_len)
            if staple is not None:
                return staple.to_action()
    return random_legal(obs, cfg, rng)


POLICIES = {"random": random_legal, "greedy": greedy_sweep}
