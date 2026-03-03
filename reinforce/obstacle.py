from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .config import ObstacleConfig


@dataclass
class Obstacle:
    center: np.ndarray  # shape (2,), [x, y] in pixels
    radius: float       # radius in pixels

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)

    def describe(self) -> Dict:
        return {"center": self.center.copy(), "radius": self.radius}


class ObstacleManager:
    def __init__(self, cfg: ObstacleConfig) -> None:
        if cfg.random:
            raise NotImplementedError("Random obstacles TBD")
        if cfg.dynamic:
            raise NotImplementedError("Dynamic obstacles TBD")

        self.obstacles: List[Obstacle] = [
            Obstacle(center=np.asarray(pos, dtype=float), radius=cfg.radius)
            for pos in cfg.positions
        ]

    def get_render_data(self) -> List[Dict]:
        return [obs.describe() for obs in self.obstacles]


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
