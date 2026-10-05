import gymnasium as gym

from .config import EnvConfig
from .design import Staple
from .env import FoldingEnv

__all__ = ["EnvConfig", "FoldingEnv", "Staple"]

if "MolRLFolding-v0" not in gym.registry:
    gym.register(id="MolRLFolding-v0", entry_point="molrl_folding.env:FoldingEnv")
