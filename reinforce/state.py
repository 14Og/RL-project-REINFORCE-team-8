import math

import numpy as np


class State:
    """Robot observation.

    Observation vector layout:
        [sin(th1), cos(th1), ..., sin(thN), cos(thN), ee_x, ee_y, dist_x, dist_y, ray_0, ..., ray_M]
    Total length: 2*n_dof + 4 + n_rays*n_lidars

    All spatial values (ee_x, ee_y, dist_x, dist_y) are normalized by reach_max (sum of link lengths).
    Lidar readings are normalized by ray_maxlen — 1.0 means no hit, values < 1.0 indicate a hit.
    """

    def __init__(self, thetas: np.ndarray, ee_x: float, ee_y: float,
                 dist_x: float, dist_y: float, rays: np.ndarray) -> None:
        self.thetas = np.asarray(thetas, dtype=float)  # joint angles, shape (n_dof,)
        self.ee_x: float = float(ee_x)   # normalized relative to base
        self.ee_y: float = float(ee_y)   # normalized relative to base
        self.dist_x: float = float(dist_x)  # (target_x - ee_x) / reach_max
        self.dist_y: float = float(dist_y)  # (target_y - ee_y) / reach_max
        self.lidar_rays: np.ndarray = np.asarray(rays, dtype=float)

    def __array__(self, dtype=np.float32) -> np.ndarray:
        trig = np.array([[math.sin(t), math.cos(t)] for t in self.thetas], dtype=float).ravel()
        tail = np.array([self.ee_x, self.ee_y, self.dist_x, self.dist_y], dtype=float)
        rays = self.lidar_rays.ravel()
        return np.concatenate([trig, tail, rays]).astype(dtype)



if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
