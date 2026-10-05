import itertools

import numpy as np
import pytest

from molrl_folding.config import EnvConfig
from molrl_folding.design import Staple, feasible_crossovers, free_runs, is_legal


def test_staple_length_formula():
    st = Staple(2, 10, 26, 10, 1)
    assert st.len1 == 17 and st.len2 == 17 and st.length == 34
    assert st.row1 == 3 and Staple(2, 10, 26, 10, 0).row1 == 1


def test_free_runs():
    free = np.array([[1, 1, 0, 1, 1, 1]], dtype=bool)
    left, right = free_runs(free)
    assert left[0].tolist() == [1, 2, 0, 1, 2, 3]
    assert right[0].tolist() == [2, 1, 0, 3, 2, 1]


def test_legality_rules():
    cfg = EnvConfig(h_max=4, l_max=40, l_staple_min=10, l_staple_max=20)
    free = np.ones((4, 40), dtype=bool)
    assert is_legal(Staple(1, 5, 10, 5, 1), free, cfg)          # 6 + 6 = 12 nt
    assert not is_legal(Staple(1, 5, 7, 5, 1), free, cfg)       # 3 + 3 = 6 nt, too short
    assert not is_legal(Staple(1, 0, 19, 0, 1), free, cfg)      # 20 + 20 = 40 nt, too long
    assert not is_legal(Staple(0, 5, 10, 5, 0), free, cfg)      # neighbour row -1 does not exist
    assert not is_legal(Staple(3, 5, 10, 5, 1), free, cfg)      # neighbour row 4 does not exist
    free[2, 8] = False
    assert not is_legal(Staple(1, 5, 10, 5, 1), free, cfg)      # domain 2 would cross a covered cell


def _brute_force_has_staple(free: np.ndarray, cfg: EnvConfig, r: int, x: int) -> bool:
    h, l = free.shape
    for side in (0, 1):
        for c0, ce in itertools.product(range(l), repeat=2):
            if is_legal(Staple(r, c0, x, ce, side), free, cfg) and (r + 1 if side else r - 1) == r + 1:
                return True
    return False


@pytest.mark.parametrize("cfg", [
    EnvConfig(h_max=3, l_max=14, l_staple_min=6, l_staple_max=10, min_row_len=3, shape_step=3),
    EnvConfig(h_max=3, l_max=14, l_staple_min=6, l_staple_max=10, l_domain_min=3, min_row_len=3, shape_step=3),
    EnvConfig(h_max=3, l_max=14, l_staple_min=6, l_staple_max=10, crossover_period=3, min_row_len=3, shape_step=3),
])
def test_feasible_crossovers_matches_brute_force(cfg):
    rng = np.random.default_rng(0)
    for _ in range(25):
        free = rng.random((cfg.h_max, cfg.l_max)) < 0.75
        ok = feasible_crossovers(free, cfg)
        for r in range(cfg.h_max - 1):
            for x in range(cfg.l_max):
                assert ok[r, x] == _brute_force_has_staple(free, cfg, r, x), (r, x)
