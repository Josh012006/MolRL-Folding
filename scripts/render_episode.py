"""Play one episode with a baseline policy and save the final design as a PNG.

    python scripts/render_episode.py --policy greedy --seed 3 --out outputs/episode.png
"""
import argparse
from pathlib import Path

import numpy as np

from molrl_folding import EnvConfig, FoldingEnv
from molrl_folding.baselines import POLICIES
from molrl_folding.render import draw_design


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=list(POLICIES), default="greedy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/episode.png"))
    args = p.parse_args()

    env = FoldingEnv(EnvConfig())
    obs, info = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    done = False
    while not done:
        action = POLICIES[args.policy](obs, env.cfg, rng)
        if action is None:
            break
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    state = "complete" if info["complete"] else "dead end" if info["dead_end"] else "truncated"
    title = f"{args.policy}, seed {args.seed}: {info['n_staples']} staples, {100 * info['covered_fraction']:.0f}% covered ({state})"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    draw_design(env.shape, env.staples, title).savefig(args.out)
    print(f"saved {args.out}: {title}")


if __name__ == "__main__":
    main()
