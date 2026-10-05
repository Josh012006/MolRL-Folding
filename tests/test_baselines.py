import numpy as np

from molrl_folding import EnvConfig, FoldingEnv
from molrl_folding.baselines import POLICIES


def mean_coverage(name, seeds, cfg=None):
    env = FoldingEnv(cfg)
    out = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(POLICIES[name](obs, env.cfg, rng))
            done = terminated or truncated
        out.append(info["covered_fraction"])
    return float(np.mean(out))


def test_greedy_covers_more_than_random():
    seeds = range(15)
    assert mean_coverage("greedy", seeds) > mean_coverage("random", seeds)
