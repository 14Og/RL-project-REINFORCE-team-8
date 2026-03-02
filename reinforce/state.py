import numpy as np

from dataclasses import dataclass, field


@dataclass
class State:
    sin_th1: float  # sin() for base joint
    cos_th1: float  # cos() for base joint
    sin_th2: float  # sin() for middle joint
    cos_th2: float  # cos() for middle joint
    sin_th3: float  # sin() for ee joint
    cos_th3: float  # cos() for ee joint
    ee_x: float  # ee (end-effector) x pos
    ee_y: float  # ee y pos
    dist_x: float = field(init=False)  # abs(target_x - ee_x)
    dist_y: float = field(init=False)  # abs(target_y - ee_y)
    lidar_j1: np.ndarray = field(init=False)  # Joint 1 lidar readings (16 rays)
    lidar_j2: np.ndarray = field(init=False)  # Joint 2 lidar readings (16 rays)
    lidar_j3: np.ndarray = field(init=False)  # Joint 3 lidar readings (16 rays)

    def __array__(self, dtype = np.float32) -> np.ndarray:
        return np.concatenate([
            np.array(
                [
                    self.sin_th1,
                    self.cos_th1,
                    self.sin_th2,
                    self.cos_th2,
                    self.sin_th3,
                    self.cos_th3,
                    self.ee_x,
                    self.ee_y,
                    self.dist_x,
                    self.dist_y,
                ], dtype
            ),
            np.asarray(self.lidar_j1, dtype=dtype),
            np.asarray(self.lidar_j2, dtype=dtype),
            np.asarray(self.lidar_j3, dtype=dtype),
        ])


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
