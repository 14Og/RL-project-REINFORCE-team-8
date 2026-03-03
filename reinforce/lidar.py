import math
from typing import List

import numpy as np

from .config import LidarConfig
from .obstacle import Obstacle


class Lidar:
    def __init__(self, num_rays: int, ray_maxlen: float) -> None:
        self.ray_maxlen = ray_maxlen
        self.position = np.zeros(2, dtype=float)

        angles = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
        self.ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    def scan(self, obstacles: List[Obstacle]) -> np.ndarray:
        n = len(self.ray_dirs)
        readings = np.ones(n, dtype=float)

        for i, d in enumerate(self.ray_dirs):
            min_t = self.ray_maxlen
            for obs in obstacles:
                oc = self.position - obs.center
                b = 2.0 * float(np.dot(d, oc))
                c = float(np.dot(oc, oc)) - obs.radius ** 2
                disc = b * b - 4.0 * c
                if disc >= 0:
                    t = (-b - math.sqrt(disc)) / 2.0
                    if 0.0 <= t < min_t:
                        min_t = t
            readings[i] = min_t / self.ray_maxlen

        return readings


class LidarManager:
    def __init__(self, cfg: LidarConfig, n_dof: int) -> None:
        self.cfg = cfg
        self.n_dof = n_dof
        self._obstacles: List[Obstacle] = []

        n_lidars = n_dof * (int(cfg.lidar_joints) + int(cfg.lidar_midlinks))
        self.lidars: List[Lidar] = [Lidar(cfg.num_rays, cfg.ray_maxlen_px) for _ in range(n_lidars)]

    @property
    def n_lidars(self) -> int:
        return len(self.lidars)

    @property
    def n_rays_total(self) -> int:
        return self.n_lidars * self.cfg.num_rays

    def set_obstacles(self, obstacles: List[Obstacle]) -> None:
        self._obstacles = obstacles

    def update_positions(self, joints: np.ndarray) -> None:
        idx = 0
        if self.cfg.lidar_joints:
            for i in range(1, self.n_dof + 1):
                self.lidars[idx].position = joints[i].copy()
                idx += 1
        if self.cfg.lidar_midlinks:
            for i in range(self.n_dof):
                self.lidars[idx].position = ((joints[i] + joints[i + 1]) / 2.0).copy()
                idx += 1

    def scan(self) -> np.ndarray:
        return np.concatenate([lidar.scan(self._obstacles) for lidar in self.lidars])


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
