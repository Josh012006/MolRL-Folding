import numpy as np

from molrl_folding import EnvConfig, FoldingEnv
from molrl_folding.baselines import POLICIES
from molrl_folding.render import draw_design, figure_to_array


def test_draw_design_with_and_without_staples():
    env = FoldingEnv(EnvConfig())
    obs, _ = env.reset(seed=0)
    empty = figure_to_array(draw_design(env.shape, []))
    assert empty.ndim == 3 and empty.shape[2] == 3

    rng = np.random.default_rng(0)
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(POLICIES["greedy"](obs, env.cfg, rng))
        done = terminated or truncated
    full = figure_to_array(draw_design(env.shape, env.staples, "title", show_legend=False))
    assert full.dtype == np.uint8
