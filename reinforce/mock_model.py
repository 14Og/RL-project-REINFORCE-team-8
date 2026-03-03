"""
Mock model — random-action stub that satisfies the Environment's Model interface.

Use this to test env mechanics (kinematics, lidar, obstacle avoidance, reward
shaping, initial-state heuristics) without any learned policy.  Requires no
PyTorch; the only dependency is numpy.

Usage (notebook / script):
    from reinforce.mock_model import MockModel
    model = MockModel(act_dim=2, action_limit=0.1, seed=0)
    runner = HeadlessRunner(..., model=model)
    metrics = runner.train(episodes=300)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np


class MockModel:
    """Uniformly-random action stub implementing the same interface as ``Model``.

    Parameters
    ----------
    act_dim:
        Number of action dimensions (= number of robot DoFs).
    action_limit:
        Scalar limit; actions are sampled from U(-action_limit, +action_limit).
    seed:
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        act_dim: int,
        action_limit: float,
        seed: int = 42,
    ) -> None:
        self.act_dim = int(act_dim)
        self.action_limit = float(action_limit)
        self._rng = np.random.default_rng(seed)

        self._rewards: List[float] = []
        self._steps: int = 0

        self.train: Dict[str, List[float]] = {
            "total_reward": [],
            "success": [],
            "steps": [],
            "final_distance": [],
        }
        self.test: Dict[str, List[float]] = {
            "success": [],
            "final_distance": [],
            "steps": [],
        }

    # ------------------------------------------------------------------
    # Episode lifecycle (called by Environment)
    # ------------------------------------------------------------------

    def start_episode(self) -> None:
        self._rewards.clear()
        self._steps = 0

    def select_action(
        self,
        state: Union[np.ndarray, object],
        *,
        train: bool = True,
    ) -> np.ndarray:
        """Return a uniformly-random action, ignoring the state."""
        self._steps += 1
        return self._rng.uniform(
            -self.action_limit, self.action_limit, size=(self.act_dim,)
        ).astype(np.float32)

    def observe(self, reward: float) -> None:
        self._rewards.append(float(reward))

    def finish_episode(
        self,
        *,
        success: bool,
        final_distance: Optional[float] = None,
    ) -> Dict[str, float]:
        total_reward = float(sum(self._rewards))
        metrics = {
            "total_reward": total_reward,
            "success": float(bool(success)),
            "steps": float(self._steps),
            "final_distance": float(final_distance) if final_distance is not None else float("nan"),
        }
        for k, v in metrics.items():
            self.train[k].append(v)
        return metrics

    def record_test_episode(
        self,
        *,
        success: bool,
        final_distance: float,
        steps: int,
    ) -> None:
        self.test["success"].append(float(bool(success)))
        self.test["final_distance"].append(float(final_distance))
        self.test["steps"].append(float(steps))

    # ------------------------------------------------------------------
    # Metrics accessors
    # ------------------------------------------------------------------

    def get_train_metrics(self) -> Dict[str, List[float]]:
        return self.train

    def get_test_metrics(self) -> Dict[str, List[float]]:
        return self.test

    # ------------------------------------------------------------------
    # Persistence (no-ops — nothing to save for a random policy)
    # ------------------------------------------------------------------

    def save(self, path: str, **kwargs) -> None:  # noqa: ARG002
        pass

    def load(self, path: str, **kwargs) -> None:  # noqa: ARG002
        pass

    def set_train_mode(self) -> None:
        pass

    def set_eval_mode(self) -> None:
        pass
