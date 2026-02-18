import numpy as np

from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass(frozen=True)
class RobotConfig:
    """robot configuration (pure kinematics)."""

    base_xy: Tuple[float, float] = (400, 400)
    link_lengths: Tuple[float, float] = (100, 80)

    wrap_angles: bool = True
    dtheta_max: Optional[float] = 0.1  # if set, action will be clipped per joint

@dataclass
class ModelConfig:
    gamma: float = 0.99
    lr_start: float = 1e-4
    lr_min: float = 1e-5
    baseline_buf_len: int = 200
    grad_clip_norm: float = 1.0
    hidden_sizes: Tuple[int, ...] = (128, 128)
    log_std_min: float = -3.0
    log_std_max: float = -0.5

@dataclass
class RewardConfig:
    """Reward coefficients"""

    # Dense shaping
    progress_scale: float = 0.03     # r += progress_scale * (prev_dist - dist)
    step_penalty: float = 0.01     # r -= step_penalty each step

    # Terminal bonuses/penalties
    goal_reward: float = 15.0        # added when goal reached
    fail_penalty: float = 5.0       # subtracted when failure triggered
    
    # Trajectory smoothness
    action_l2_scale: float = 0.0     # penalise large joint velocities: ||a_t||²
    action_delta_scale: float = 0.0  # penalise jerk: ||a_t - a_{t-1}||²

@dataclass
class EnvConfig:
    """Environment/termination/state-feature parameters."""

    target_xy: Tuple[float, float] = (100, 300.0)
    randomize_target: bool = True         # randomize target each episode within reachable workspace

    # Termination
    target_thresh: float = 30.0
    max_steps: int = 200
    
    # Constraints / failure conditions
    forbid_link_target_intersection: bool = True
    target_point_radius: float = 1.0   # distance threshold from target point to a link segment

    # Distance features in State (dist_x, dist_y)
    # Your comment says abs(target - ee); keep that as default.
    use_abs_dist: bool = False
    normalize_dist: bool = True
    dist_scale: float = 300.0
    
@dataclass
class GUIConfig:
    window_size: Tuple[int, int] = (1600, 800)
    sim_width: int = 800                 # left panel width
    plot_update_every: int = 10          # re-render plots every N frames
    pause_on_done_frames: int = 0        # optional pause after episode end (0 disables)
    steps_per_frame: int = 1            # env steps per GUI frame (higher = faster training)
    steps_per_frame_no_sim: int = 500    # env steps per frame when sim panel hidden
    model_path: str = "best_policy.pt"
    train_episodes: int = 5000
    test_episodes: int = 200
