"""Run the baseline policies on the same shapes and print a comparison table.

    python scripts/evaluate_baselines.py --episodes 100
"""
import argparse
import time

import numpy as np

from molrl_folding import EnvConfig, FoldingEnv
from molrl_folding.baselines import POLICIES


def run(policy_name: str, cfg: EnvConfig, seeds) -> dict[str, float]:
    env = FoldingEnv(cfg)
    policy = POLICIES[policy_name]
    rows = []
    start = time.time()
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        done = False
        while not done:
            action = policy(obs, cfg, rng)
            if action is None:
                break
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        rows.append((info["covered_fraction"], info["complete"], info["dead_end"], info["n_staples"]))
    a = np.array(rows, dtype=float)
    return {
        "coverage": a[:, 0].mean(),
        "complete": a[:, 1].mean(),
        "dead_end": a[:, 2].mean(),
        "staples": a[:, 3].mean(),
        "sec/ep": (time.time() - start) / len(seeds),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="first seed; episodes use seed..seed+episodes-1")
    p.add_argument("--h-max", type=int, default=16)
    p.add_argument("--l-max", type=int, default=128)
    p.add_argument("--l-domain-min", type=int, default=1)
    p.add_argument("--crossover-period", type=int, default=1)
    args = p.parse_args()

    cfg = EnvConfig(h_max=args.h_max, l_max=args.l_max, l_domain_min=args.l_domain_min,
                    crossover_period=args.crossover_period)
    seeds = range(args.seed, args.seed + args.episodes)

    print(f"{args.episodes} episodes, canvas {cfg.h_max} x {cfg.l_max}, "
          f"domain >= {cfg.l_domain_min} nt, crossover every {cfg.crossover_period} nt\n")
    print(f"{'policy':<10}{'coverage':>10}{'complete':>10}{'dead_end':>10}{'staples':>9}{'sec/ep':>9}")
    for name in POLICIES:
        r = run(name, cfg, seeds)
        print(f"{name:<10}{r['coverage']:>10.3f}{r['complete']:>10.2f}{r['dead_end']:>10.2f}"
              f"{r['staples']:>9.1f}{r['sec/ep']:>9.2f}")


if __name__ == "__main__":
    main()
