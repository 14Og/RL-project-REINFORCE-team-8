from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from config import EnvConfig, RewardConfig, RobotConfig
from robot import Robot
from model import Model
from state import State


class Environment:
    def __init__(
        self,
        *,
        env_cfg: EnvConfig,
        reward_cfg: RewardConfig,
        robot_cfg: RobotConfig,
        model: Model,
        seed: int = 42,
    ) -> None:
        self.env_cfg = env_cfg
        self.rew_cfg = reward_cfg
        self.target = np.asarray(env_cfg.target_xy, dtype=np.float32)

        self.robot = Robot(robot_cfg, seed=seed)
        self.model = model
        self._rng = np.random.default_rng(seed)

        base = np.asarray(robot_cfg.base_xy, dtype=np.float32)
        L1, L2 = float(robot_cfg.link_lengths[0]), float(robot_cfg.link_lengths[1])
        self._base = base
        self._reach_max = L1 + L2
        self._reach_min = max(abs(L1 - L2), L1)

        self._train_mode: bool = True
        self._needs_reset: bool = True

        self.steps: int = 0
        self.done: bool = False
        self.success: bool = False
        self.reason: str = "not_started"
        self._prev_dist: float = float("nan")

        self._last_state: Optional[State] = None
        self._prev_action = None
        self._curr_action = None

    def reset_episode(self, *, train: bool = True, randomize_theta: bool = True) -> np.ndarray:
        self._train_mode = bool(train)

        self.robot.reset(randomize=randomize_theta)

        if self.env_cfg.randomize_target:
            self.target = self._sample_reachable_target()

        self.steps = 0
        self.done = False
        self.success = False
        self.reason = "running"

        if self._train_mode:
            self.model.start_episode()

        ee = self.robot.end_effector_xy()
        self._prev_dist = float(np.linalg.norm(ee - self.target))
        
        self._prev_action = None
        self._curr_action = None

        self._needs_reset = False

        st = self._get_state()
        self._last_state = st
        return np.asarray(st, dtype=np.float32)

    def step(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self._needs_reset:
            obs0 = self.reset_episode(train=self._train_mode, randomize_theta=True)
            return obs0, 0.0, False, {"reset": True}

        if self.done:
            self._needs_reset = True
            st = self._get_state()
            return np.asarray(st, dtype=np.float32), 0.0, True, {"needs_reset": True}

        st = self._get_state()
        self._last_state = st

        if self._train_mode:
            u = self.model.select_action(st, train=True)
        else:
            u = self.model.select_action(st, train=False)
        _, self._curr_action = self.robot.step(u)
        self.steps += 1

        reward, done, info = self._compute_reward_and_done()

        if self._train_mode:
            self.model.observe(reward)

        self.done = bool(done)

        if self.done:
            final_dist = float(info.get("final_distance", float("nan")))
            self.success = bool(info.get("success", False))
            self.reason = str(info.get("reason", "done"))

            if self._train_mode:
                self.model.finish_episode(success=self.success, final_distance=final_dist)
            else:
                self.model.record_test_episode(
                    success=self.success, final_distance=final_dist, steps=int(self.steps)
                )

            self._needs_reset = True

        st_next = self._get_state()
        self._last_state = st_next
        self._prev_action = self._curr_action
        return np.asarray(st_next, dtype=np.float32), float(reward), bool(self.done), info

    def get_render_data(self) -> Dict[str, Any]:
        joints = self.robot.joints_xy().astype(np.float32)
        ee = joints[-1]
        dist = float(np.linalg.norm(ee - self.target))

        return {
            "joints": joints,
            "end_effector": ee,
            "target": self.target.copy(),
            "distance": dist,
            "theta": self.robot.theta,
            "step": int(self.steps),
            "done": bool(self.done),
            "success": bool(self.success),
            "reason": self.reason,
            "train_mode": bool(self._train_mode),
        }

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "train": self.model.get_train_metrics(),
            "test": self.model.get_test_metrics(),
        }

    def _sample_reachable_target(self) -> np.ndarray:
        angle = self._rng.uniform(0, 2 * np.pi)
        r_sq = self._rng.uniform(self._reach_min ** 2, self._reach_max ** 2)
        r = np.sqrt(r_sq)
        return self._base + np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)

    def _get_state(self) -> State:
        st = self.robot.obs()
        st.ee_x = (st.ee_x - self._base[0]) / self._reach_max
        st.ee_y = (st.ee_y - self._base[1]) / self._reach_max

        dx = float(self.target[0] - self.robot.end_effector_xy()[0])
        dy = float(self.target[1] - self.robot.end_effector_xy()[1])

        if self.env_cfg.use_abs_dist:
            dx, dy = abs(dx), abs(dy)

        if self.env_cfg.normalize_dist:
            s = float(self.env_cfg.dist_scale)
            dx, dy = dx / s, dy / s

        st.dist_x = dx
        st.dist_y = dy 
        return st

    def _compute_reward_and_done(self) -> Tuple[float, bool, Dict[str, Any]]:
        joints = self.robot.joints_xy()
        p0, p1, p2 = joints[0], joints[1], joints[2]
        ee = p2
        dist = float(np.linalg.norm(ee - self.target))

        progress = float(self._prev_dist - dist)
        reward = float(self.rew_cfg.progress_scale) * progress - float(self.rew_cfg.step_penalty)
        
        a_t = self._curr_action
        if a_t is not None and self.rew_cfg.action_l2_scale != 0.0:
            reward -= self.rew_cfg.action_l2_scale * float(np.linalg.norm(a_t, a_t))

        if a_t is not None and self._prev_action is not None and self.rew_cfg.action_delta_scale != 0.0:
            da = a_t - self._prev_action
            reward -= self.rew_cfg.action_delta_scale * float(np.dot(da, da))
            
        goal_reached = dist < float(self.env_cfg.target_thresh)

        fail = False
        fail_reason = ""

        if self.env_cfg.forbid_link_target_intersection and (not goal_reached):
            r = float(self.env_cfg.target_point_radius)
            d01 = _point_to_segment_distance(self.target, p0, p1)
            d12 = _point_to_segment_distance(self.target, p1, p2)
            if (d01 < r) or (d12 < r):
                fail = True
                fail_reason = "link_target_intersection"

        timeout = self.steps >= int(self.env_cfg.max_steps)
        done = goal_reached or fail or timeout

        info: Dict[str, Any] = {
            "success": bool(goal_reached),
            "final_distance": dist,
            "progress": progress,
            "timeout": bool(timeout),
            "fail": bool(fail),
            "reason": (
                "goal" if goal_reached else ("timeout" if timeout else (fail_reason or "fail"))
            ),
        }

        if goal_reached:
            reward += float(self.rew_cfg.goal_reward)
        elif fail:
            reward -= float(self.rew_cfg.fail_penalty)
        elif timeout:
            reward -= float(self.rew_cfg.fail_penalty)

        self._prev_dist = dist
        return float(reward), bool(done), info


def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).reshape(2)
    a = np.asarray(a, dtype=float).reshape(2)
    b = np.asarray(b, dtype=float).reshape(2)

    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))

    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
