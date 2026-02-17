import math
import numpy as np

from dataclasses import dataclass, field


@dataclass()
class State:
    sin_th1: float  # sin() for base joint
    cos_th1: float  # cos() for base joint
    sin_th2: float  # sin() for ee joint
    cos_th2: float  # cos() for ee joint
    ee_x: float  # ee (end-effector) x pos
    ee_y: float  # ee y pos
    dist_x: float = field(init=False)  # abs(target_x - ee_x)
    dist_y: float = field(init=False)  # abs(target_y - ee_y)

    def __array__(self, dtype = np.float32) -> np.ndarray:
        return np.array(
            [
                self.sin_th1,
                self.cos_th1,
                self.sin_th2,
                self.cos_th2,
                self.ee_x,
                self.ee_y,
                self.dist_x,
                self.dist_y,
            ], dtype
        )
