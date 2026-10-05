import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import molrl_folding  # noqa: F401  (registers MolRLFolding-v0)
from molrl_folding import EnvConfig, FoldingEnv
from molrl_folding.baselines import POLICIES


def run_episode(env, policy, seed):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    done = False
    while not done:
        action = policy(obs, env.cfg, rng)
        assert action is not None
        obs, reward, terminated, truncated, info = env.step(action)
        assert not info["invalid_action"]
        assert (obs[1] <= obs[0]).all()
        done = terminated or truncated
    return obs, reward, terminated, truncated, info


def test_gymnasium_checker():
    check_env(FoldingEnv(), skip_render_check=True)


def test_registered_id():
    env = gym.make("MolRLFolding-v0")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_reset_is_reproducible():
    env = FoldingEnv()
    a, ia = env.reset(seed=7)
    b, ib = env.reset(seed=7)
    assert (a == b).all() and ia["shape_id"] == ib["shape_id"]


def test_invalid_action_changes_nothing():
    env = FoldingEnv()
    obs, _ = env.reset(seed=1)
    after, reward, terminated, truncated, info = env.step(np.array([0, 0, 0, 0, 0]))
    assert info["invalid_action"] and reward == 0.0 and not terminated
    assert (after == obs).all() and env.staples == []


def test_no_cell_is_covered_twice():
    env = FoldingEnv()
    for seed in range(5):
        run_episode(env, POLICIES["random"], seed)
        total = sum(s.length for s in env.staples)
        assert total == int(env.covered.sum())


def test_termination_flags_and_terminal_reward():
    env = FoldingEnv()
    for seed in range(10):
        obs, reward, terminated, truncated, info = run_episode(env, POLICIES["random"], seed)
        if terminated:
            assert info["complete"] != info["dead_end"]
            assert reward == info["covered_fraction"]
            if info["complete"]:
                assert info["covered_fraction"] == 1.0


def test_reward_is_zero_before_the_end():
    env = FoldingEnv()
    obs, _ = env.reset(seed=2)
    rng = np.random.default_rng(2)
    obs, reward, terminated, truncated, _ = env.step(POLICIES["greedy"](obs, env.cfg, rng))
    if not (terminated or truncated):
        assert reward == 0.0


def test_truncation_on_step_limit():
    env = FoldingEnv(EnvConfig(max_steps_factor=0.0))
    env.reset(seed=0)
    *_, terminated, truncated, info = env.step(np.array([0, 0, 0, 0, 0]))
    assert truncated and not terminated


def test_custom_reward_function():
    env = FoldingEnv(reward_fn=lambda shape, covered, staples, cfg: 42.0)
    _, reward, terminated, truncated, _ = run_episode(env, POLICIES["greedy"], 0)
    assert terminated and reward == 42.0


def test_constraints_are_respected_with_domain_and_crossover_rules():
    cfg = EnvConfig(l_domain_min=8, crossover_period=8)
    env = FoldingEnv(cfg)
    for name in POLICIES:
        run_episode(env, POLICIES[name], 3)
        for s in env.staples:
            assert s.len1 >= 8 and s.len2 >= 8 and s.col_x % 8 == 0
            assert cfg.l_staple_min <= s.length <= cfg.l_staple_max


def test_render_returns_an_image():
    env = FoldingEnv(render_mode="rgb_array")
    run_episode(env, POLICIES["greedy"], 0)
    img = env.render()
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
