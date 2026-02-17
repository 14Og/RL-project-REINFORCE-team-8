import numpy as np

from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass(frozen=True)
class RobotConfig:
    """robot configuration (pure kinematics)."""

    base_xy: Tuple[float, float] = (200.0, 200.0)
    link_lengths: Tuple[float, float] = (100, 150)

    wrap_angles: bool = True
    dtheta_max: Optional[float] = 0.03  # if set, action will be clipped per joint

@dataclass
class RewardConfig:
    """Reward coefficients"""

    # Dense shaping
    progress_scale: float = 1.0      # r += progress_scale * (prev_dist - dist)
    step_penalty: float = 0.01       # r -= step_penalty each step

    # Terminal bonuses/penalties
    goal_reward: float = 10.0        # added when goal reached
    fail_penalty: float = 10.0       # subtracted when failure triggered
    
    # Trajectory smoothness
    action_l2_scale: float = 0.0      # λ_a
    action_delta_scale: float = 0.0   # λ_Δ

@dataclass
class EnvConfig:
    """Environment/termination/state-feature parameters."""

    target_xy: Tuple[float, float] = (200.0, 50.0)

    # Termination
    target_thresh: float = 20.0
    max_steps: int = 200

    # Constraints / failure conditions
    forbid_link_target_intersection: bool = True
    target_point_radius: float = 2.0   # distance threshold from target point to a link segment

    # Distance features in State (dist_x, dist_y)
    # Your comment says abs(target - ee); keep that as default.
    use_abs_dist: bool = True
    normalize_dist: bool = False
    dist_scale: float = 300.0

    # Reset behavior
    auto_reset: bool = True
    
@dataclass
class GUIConfig:
    window_size: Tuple[int, int] = (1600, 800)
    sim_width: int = 800  # left panel width
    fps: int = 60
    control_hz: int = 30  # how often to call env.step()

    plot_update_sec: float = 0.25  # update matplotlib surface at this rate

    train_episodes: int = 500
    test_episodes: int = 100

    model_path: str = "best_policy.pt"

    pause_on_done_sec: float = 0.0  # set >0 to pause after terminal
