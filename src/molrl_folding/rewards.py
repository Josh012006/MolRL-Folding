from typing import Callable, Sequence

import numpy as np

from .config import EnvConfig
from .design import Staple

# (shape, covered, staples, cfg) -> terminal reward of a finished design
RewardFn = Callable[[np.ndarray, np.ndarray, Sequence[Staple], EnvConfig], float]


def coverage_reward(shape: np.ndarray, covered: np.ndarray, staples: Sequence[Staple], cfg: EnvConfig) -> float:
    """Placeholder: fraction of the shape that is covered. To be replaced by the robustness score."""
    return float(covered.sum()) / max(float(shape.sum()), 1.0)
