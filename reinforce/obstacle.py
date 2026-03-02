import numpy as np
from dataclasses import dataclass


@dataclass
class Obstacle:
    center: np.ndarray  # shape (2,), [x, y] in pixels
    radius: float       # radius in pixels

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
