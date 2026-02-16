import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

@dataclass(frozen=True)
class Config:
    """robot configuration (pure kinematics)."""

    base_xy: Tuple[float, float] = (200.0, 200.0)
    link_lengths: Tuple[float, float] = (100, 150)

    wrap_angles: bool = True
    dtheta_max: Optional[float] = 0.1  # if set, action will be clipped per joint


class Robot:
    """2dof planar arm robot model.

    state:  theta = [theta1, theta2] in radians
    action: dtheta = [dtheta1, dtheta2] in radians

    getters:
      - theta: return current joint positions
      - joints_xy -> (3,2) array: base, elbow, end-effector
      - end_effector_xy()
      - obs() -> [sin(th1), cos(th1), sin(th2), cos(th2), x_ee, y_ee]

    setters:
      - reset()
      - set_theta()
      - step(dtheta): adds increments to current state
    """

    def __init__(self, cfg: Config, seed: int = 42, theta: Optional[np.ndarray] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self.base = np.asarray(cfg.base_xy, dtype=float)
        self.L1, self.L2 = map(float, cfg.link_lengths)

        self._theta = np.zeros(2, dtype=float)
        if theta is not None:
            self.set_theta(theta)

    # -------- getters --------
    @property
    def theta(self) -> np.ndarray:
        """joint angles [theta1, theta2] (copy), radians."""
        return self._theta.copy()

    def joints_xy(self) -> np.ndarray:
        """forward kinematics points: base, elbow, end-effector. shape: (3, 2).

        coordinates are standard cartesian (y up).
        (your gui can convert to screen coords if needed.)
        """
        t1, t2 = float(self._theta[0]), float(self._theta[1])
        p0 = self.base
        p1 = p0 + np.array([self.L1 * math.cos(t1), self.L1 * math.sin(t1)], dtype=float)
        p2 = p1 + np.array([self.L2 * math.cos(t1 + t2), self.L2 * math.sin(t1 + t2)], dtype=float)
        return np.stack([p0, p1, p2], axis=0)

    def end_effector_xy(self) -> np.ndarray:
        """end-effector position [x, y]."""
        return self.joints_xy()[-1]

    def obs(self, dtype=np.float32) -> np.ndarray:
        """RL observation:
        [sin(th1), cos(th1), sin(th2), cos(th2), x_ee, y_ee]
        """
        t1, t2 = float(self._theta[0]), float(self._theta[1])
        ee = self.end_effector_xy()
        return np.array(
            [math.sin(t1), math.cos(t1), math.sin(t2), math.cos(t2), float(ee[0]), float(ee[1])],
            dtype=dtype,
        )

    # -------- setters / control --------
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

    def step(self, dtheta: np.ndarray) -> np.ndarray:
        """control step: add dtheta to current angles. returns obs()."""
        dtheta = np.asarray(dtheta, dtype=float).reshape(2)

        if self.cfg.dtheta_max is not None:
            m = float(self.cfg.dtheta_max)
            dtheta = np.clip(dtheta, -m, m)

        self._theta = self._theta + dtheta

        if self.cfg.wrap_angles:
            self._theta = np.array([self.wrap_angle(t) for t in self._theta], dtype=float)

        return self.obs()
    
    @staticmethod
    def wrap_angle(theta: float) -> float:
        """wrap to (-pi, pi] for numerical stability."""
        return (theta + math.pi) % (2 * math.pi) - math.pi


if __name__ == "__main__":
    raise NotImplementedError("This is a project module, not executable script")