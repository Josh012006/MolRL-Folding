import numpy as np

from molrl_folding.config import EnvConfig
from molrl_folding.shapes import make_shape, row_intervals, shape_id


def test_rows_are_single_intervals_and_turns_are_shared():
    cfg = EnvConfig()
    for seed in range(200):
        shape = make_shape(np.random.default_rng(seed), cfg)
        iv = row_intervals(shape)
        assert 2 <= len(iv) <= cfg.h_max
        assert all(b - a >= cfg.min_row_len for a, b in iv)
        for r in range(1, len(iv)):
            if r % 2 == 1:   # scaffold turns on the right between rows r - 1 and r
                assert iv[r][1] == iv[r - 1][1]
            else:            # ... and on the left between rows r - 1 and r when r is even
                assert iv[r][0] == iv[r - 1][0]


def test_shape_is_reproducible_and_hash_is_stable():
    cfg = EnvConfig()
    a = make_shape(np.random.default_rng(3), cfg)
    b = make_shape(np.random.default_rng(3), cfg)
    assert (a == b).all() and shape_id(a) == shape_id(b)
    assert shape_id(a) != shape_id(make_shape(np.random.default_rng(4), cfg))


def test_rows_are_contiguous_from_the_top():
    shape = make_shape(np.random.default_rng(0), EnvConfig(), h=5)
    assert shape[:5].any(axis=1).all() and not shape[5:].any()
