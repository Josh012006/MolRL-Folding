from typing import Any

import gymnasium as gym
import numpy as np

from . import design
from .config import EnvConfig
from .design import Staple
from .render import draw_design, figure_to_array
from .rewards import RewardFn, coverage_reward
from .shapes import make_shape, shape_id


class FoldingEnv(gym.Env):
    """Build the staple layout of a target DNA origami sheet, one staple at a time.

    Observation: float32 array (2, h_max, l_max). Channel 0 is the target shape, channel 1
    marks the cells already covered by a staple.

    Action: MultiDiscrete [row0, col0, col_x, col_end, side], see `design.Staple`.

    Reward: 0 until the episode terminates, then `reward_fn(shape, covered, staples, cfg)`.

    Termination: the shape is fully covered, or no legal staple remains (dead end).
    Truncation: step limit. An illegal action leaves the state unchanged but still counts as a step.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: EnvConfig | None = None,
        reward_fn: RewardFn = coverage_reward,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self.reward_fn = reward_fn
        self.render_mode = render_mode

        h, l = self.cfg.h_max, self.cfg.l_max
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, h, l), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([h, l, l, l, 2])

        self._obs = np.zeros((2, h, l), dtype=np.float32)
        self._staples: list[Staple] = []
        self._n_steps = 0
        self._max_steps = 0

    # ---------- state access ----------

    @property
    def shape(self) -> np.ndarray:
        return self._obs[0] > 0.5

    @property
    def covered(self) -> np.ndarray:
        return self._obs[1] > 0.5

    @property
    def staples(self) -> list[Staple]:
        return list(self._staples)

    def _free(self) -> np.ndarray:
        return design.free_from_obs(self._obs)

    def _get_info(self) -> dict[str, Any]:
        n_cells = int(self.shape.sum())
        return {
            "n_cells": n_cells,
            "n_staples": len(self._staples),
            "covered_fraction": float(self.covered.sum()) / max(n_cells, 1),
            "shape_id": shape_id(self.shape),
        }

    # ---------- gym API ----------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        while True:  # resample until at least one staple can be placed
            shape = make_shape(self.np_random, self.cfg)
            if design.has_legal_action(shape, self.cfg):
                break

        self._obs[0] = shape
        self._obs[1] = 0
        self._staples = []
        self._n_steps = 0
        n_cells = int(shape.sum())
        self._max_steps = int(self.cfg.max_steps_factor * max(1, n_cells // self.cfg.l_staple_min))

        return self._obs.copy(), self._get_info()

    def step(self, action):
        self._n_steps += 1

        staple = Staple.from_action(action)
        legal = design.is_legal(staple, self._free(), self.cfg)
        if legal:
            covered = self.covered
            design.cover(covered, staple)
            self._obs[1] = covered
            self._staples.append(staple)

        free = self._free()
        complete = not free.any()
        dead_end = (not complete) and (not design.has_legal_action(free, self.cfg))

        terminated = complete or dead_end
        truncated = (not terminated) and self._n_steps >= self._max_steps
        reward = self.reward_fn(self.shape, self.covered, self._staples, self.cfg) if terminated else 0.0

        info = self._get_info()
        info.update(invalid_action=not legal, complete=complete, dead_end=dead_end)
        return self._obs.copy(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        title = f"{len(self._staples)} staples, {100 * self._get_info()['covered_fraction']:.0f}% covered"
        return figure_to_array(draw_design(self.shape, self._staples, title))
