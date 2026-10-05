import hashlib
from typing import Any

import gymnasium as gym
import numpy as np


class MolRLFoldingEnv(gym.Env):
	def __init__(self) -> None:
		super().__init__()

		self._h_max: int = 16   # max helixes
		self._l_max: int = 128  # max helix number of nt

		# staple length computed as |x_c - x_0| + |x_e - x_c| + 2
		self._l_staple_min = 20  # in nt
		self._l_staple_max = 60  # in nt

		self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, self._h_max, self._l_max), dtype=np.float32)
		self.action_space = gym.spaces.MultiDiscrete([self._h_max, self._l_max, self._l_max, self._l_max, 2])

		# obs[0] is target shape and obs[1] is what's already covered
		self.__observation = np.zeros((2, self._h_max, self._l_max), dtype=np.float32)
		self.__staples: list[tuple[int, int, int, int, bool]] = []
		self.__n_steps = 0
		self.__max_steps = 0

	# ---------- shape generation ----------

	def _make_shape(self, h: int, min_row_len: int = 8, step: int = 6) -> np.ndarray:
		"""h rows, each one a single interval; consecutive rows share the scaffold turn end."""
		rng = self.np_random
		l = int(rng.integers(0, self._l_max - min_row_len + 1))
		e = int(rng.integers(l + min_row_len, self._l_max + 1))
		rows = [(l, e)]
		for r in range(1, h):
			delta = int(rng.integers(-step, step + 1))
			if r % 2 == 1:  # turn on the right: keep e, move l
				l = int(np.clip(l + delta, 0, e - min_row_len))
			else:           # turn on the left: keep l, move e
				e = int(np.clip(e + delta, l + min_row_len, self._l_max))
			rows.append((l, e))

		shape = np.zeros((self._h_max, self._l_max), dtype=np.float32)
		for r, (a, b) in enumerate(rows):
			shape[r, a:b] = 1
		return shape

	# ---------- helpers ----------

	def _free_cells(self) -> np.ndarray:
		return (self.__observation[0] == 1) & (self.__observation[1] == 0)

	def _decode(self, action) -> tuple[int, int, int, int, int, int]:
		r0, c0, cx, ce, side = (int(a) for a in action)
		r1 = r0 + 1 if side else r0 - 1
		length = abs(cx - c0) + abs(ce - cx) + 2
		return r0, c0, cx, ce, r1, length

	def _is_legal(self, action) -> bool:
		r0, c0, cx, ce, r1, length = self._decode(action)
		if not 0 <= r1 < self._h_max:
			return False
		if not self._l_staple_min <= length <= self._l_staple_max:
			return False
		free = self._free_cells()
		return bool(free[r0, min(c0, cx):max(c0, cx) + 1].all() and free[r1, min(cx, ce):max(cx, ce) + 1].all())

	def _has_legal_action(self, free: np.ndarray) -> bool:
		"""A legal staple exists iff some crossover column is free on two adjacent rows
		and the longest free runs reachable from it add up to at least l_staple_min."""
		left = np.zeros(free.shape, dtype=np.int32)
		right = np.zeros(free.shape, dtype=np.int32)
		for c in range(self._l_max):
			prev = left[:, c - 1] if c > 0 else 0
			left[:, c] = (prev + 1) * free[:, c]
		for c in range(self._l_max - 1, -1, -1):
			nxt = right[:, c + 1] if c < self._l_max - 1 else 0
			right[:, c] = (nxt + 1) * free[:, c]
		reach = np.maximum(left, right)
		both = free[:-1] & free[1:]
		return bool((both & (reach[:-1] + reach[1:] >= self._l_staple_min)).any())

	def _terminal_reward(self) -> float:
		# TODO: replace by the robustness score (level 1, then level 2)
		return self._covered_fraction()

	def _covered_fraction(self) -> float:
		n_cells = float(self.__observation[0].sum())
		return float(self.__observation[1].sum()) / max(n_cells, 1.0)

	# ---------- gym API ----------

	def _get_info(self) -> dict[str, Any]:
		return {
			"n_cells": int(self.__observation[0].sum()),
			"n_staples": len(self.__staples),
			"covered_fraction": self._covered_fraction(),
			"shape_id": hashlib.sha1(self.__observation[0].tobytes()).hexdigest()[:12],
		}

	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
		super().reset(seed=seed)

		while True:  # resample until at least one staple can be placed
			h = int(self.np_random.integers(2, self._h_max + 1))
			shape = self._make_shape(h)
			if self._has_legal_action(shape.astype(bool)):
				break

		self.__observation[0] = shape
		self.__observation[1] = 0
		self.__staples = []
		self.__n_steps = 0
		self.__max_steps = 2 * max(1, int(shape.sum()) // self._l_staple_min)

		return self.__observation.copy(), self._get_info()

	def step(self, action):
		self.__n_steps += 1

		legal = self._is_legal(action)
		if legal:
			r0, c0, cx, ce, r1, _ = self._decode(action)
			self.__observation[1, r0, min(c0, cx):max(c0, cx) + 1] = 1
			self.__observation[1, r1, min(cx, ce):max(cx, ce) + 1] = 1
			self.__staples.append((r0, c0, cx, ce, r1 > r0))

		free = self._free_cells()
		complete = not free.any()
		dead_end = (not complete) and (not self._has_legal_action(free))

		terminated = complete or dead_end
		truncated = (not terminated) and self.__n_steps >= self.__max_steps
		reward = self._terminal_reward() if terminated else 0.0

		info = self._get_info()
		info.update(invalid_action=not legal, complete=complete, dead_end=dead_end)
		return self.__observation.copy(), reward, terminated, truncated, info

	def render(self):
		...