from state import State
from config import RobotConfig

import numpy as np

import math
from dataclasses import dataclass
from typing import Optional, Tuple

class Robot:
    def __init__(self, cfg: RobotConfig, seed: int = 42, theta: Optional[np.ndarray] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self.base = np.asarray(cfg.base_xy, dtype=float)
        self.L1, self.L2 = map(float, cfg.link_lengths)

        self._theta = np.zeros(2, dtype=float)
        if theta is not None:
            self.set_theta(theta)

    @property
    def theta(self) -> np.ndarray:
        """joint angles [theta1, theta2] (copy), radians."""
        return self._theta.copy()

    def joints_xy(self) -> np.ndarray:
        t1, t2 = float(self._theta[0]), float(self._theta[1])
        p0 = self.base
        p1 = p0 + np.array([self.L1 * math.cos(t1), self.L1 * math.sin(t1)], dtype=float)
        p2 = p1 + np.array([self.L2 * math.cos(t1 + t2), self.L2 * math.sin(t1 + t2)], dtype=float)
        return np.stack([p0, p1, p2], axis=0)

    def end_effector_xy(self) -> np.ndarray:
        return self.joints_xy()[-1]

    def obs(self, dtype=np.float32) -> State:
        """RL observation:
        [sin(th1), cos(th1), sin(th2), cos(th2), x_ee, y_ee]
        """
        t1, t2 = float(self._theta[0]), float(self._theta[1])
        ee = self.end_effector_xy()
        return State(
            sin_th1=math.sin(t1),
            cos_th1=math.cos(t1),
            sin_th2=math.sin(t2),
            cos_th2=math.cos(t2),
            ee_x=ee[0],
            ee_y=ee[1],
        )

    def reset(self, randomize: bool = True) -> np.ndarray:
        """reset robot angles. returns obs()."""
        if randomize:
            self._theta = self.rng.uniform(-math.pi, math.pi, size=2).astype(float)
        else:
            self._theta[:] = 0.0

        if self.cfg.wrap_angles:
            self._theta = np.array([self.wrap_angle(t) for t in self._theta], dtype=float)

        return self.obs()

    def set_theta(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float).reshape(2)
        if self.cfg.wrap_angles:
            theta = np.array([self.wrap_angle(float(theta[0])), self.wrap_angle(float(theta[1]))], dtype=float)
        self._theta = theta

    def step(self, dtheta: np.ndarray) -> Tuple[State, np.ndarray]:
        dtheta = np.asarray(dtheta, dtype=float).reshape(2)
        if self.cfg.dtheta_max is not None:
            dtheta = self.clip_dtheta(dtheta, self.cfg.dtheta_max)

        self._theta = self._theta + dtheta

        if self.cfg.wrap_angles:
            self._theta = np.array([self.wrap_angle(t) for t in self._theta], dtype=float)

        return self.obs(), dtheta

    @staticmethod
    def wrap_angle(theta: float) -> float:
        return (theta + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def clip_dtheta(dtheta: float, max_dtheta: float) -> float:
        return np.clip(dtheta, -max_dtheta, max_dtheta)


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
