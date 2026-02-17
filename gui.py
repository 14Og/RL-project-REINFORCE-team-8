"""gui.py

Pygame "god GUI" that visualizes:
- left panel: robot simulation (arm links + target)
- right panel: metrics plots (matplotlib rendered offscreen and blitted into pygame)

Two modes:
- TRAIN: environment trains (REINFORCE updates). GUI shows train metrics.
- TEST: loads model from disk, environment runs with train=False (no learning). GUI shows test metrics.

This file intentionally keeps env/model/robot free of any pygame/matplotlib deps.

Expected project modules:
  - config.py: RobotConfig, EnvConfig, RewardConfig
  - env.py: Environment  (use your env_fixed version)
  - model.py: Model with save/load (use model_with_io.py or patch your model.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import time
import numpy as np
import pygame

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from config import RobotConfig, EnvConfig, RewardConfig, GUIConfig
from env import Environment
from model import Model

class PlotPanel:
    """Renders metrics into a pygame.Surface via offscreen matplotlib Agg."""

    def __init__(self, size: Tuple[int, int], update_sec: float = 0.25) -> None:
        self.w, self.h = size
        self.update_sec = float(update_sec)
        self.last_t = 0.0
        self.surface: Optional[pygame.Surface] = None

        # dpi=100 => figsize in inches = pixels / 100
        self.fig = Figure(figsize=(self.w / 100.0, self.h / 100.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)

        # 3 stacked plots
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)

        self.fig.tight_layout(pad=1.0)

    def render(self, metrics: Dict[str, Any], mode: str, now: float) -> pygame.Surface:
        if self.surface is not None and (now - self.last_t) < self.update_sec:
            return self.surface

        self.last_t = now
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear()

        if mode == "train":
            train = metrics["train"]
            r = np.asarray(train["total_reward"], dtype=float)
            s = np.asarray(train["success"], dtype=float)
            steps = np.asarray(train["steps"], dtype=float)
            print(f"Success: {s.sum() / s.shape[0]}")

            if r.size:
                self.ax1.plot(r)
            self.ax1.set_title("train: total_reward")

            if s.size:
                win = min(50, s.size)
                ma = np.convolve(s, np.ones(win)/win, mode="valid")
                self.ax2.plot(np.arange(win-1, win-1+ma.size), ma)
            self.ax2.set_ylim(-0.05, 1.05)
            self.ax2.set_title("train: success_rate (moving avg)")

            if steps.size:
                self.ax3.plot(steps)
            self.ax3.set_title("train: steps/episode")

        else:  # test
            test = metrics["test"]
            s = np.asarray(test["success"], dtype=float)
            dist = np.asarray(test["final_distance"], dtype=float)
            steps = np.asarray(test["steps"], dtype=float)

            if s.size:
                self.ax1.plot(s, marker="o", linestyle="-", markersize=2)
                self.ax1.set_ylim(-0.05, 1.05)
            self.ax1.set_title("test: success (0/1)")

            if dist.size:
                self.ax2.plot(dist)
                self.ax2.set_title(f"test: final_distance (avg={dist.mean():.2f})")
            else:
                self.ax2.set_title("test: final_distance")

            if steps.size:
                self.ax3.plot(steps)
                self.ax3.set_title(f"test: steps (avg={steps.mean():.1f})")
            else:
                self.ax3.set_title("test: steps")

        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()

        buf = np.asarray(self.canvas.buffer_rgba())
        rgb = buf[..., :3]
        surf = pygame.surfarray.make_surface(np.transpose(np.ascontiguousarray(rgb), (1, 0, 2)))
        self.surface = surf
        return surf


class GUIApp:
    def __init__(
        self,
        gui_cfg: GUIConfig,
        robot_cfg: RobotConfig,
        env_cfg: EnvConfig,
        reward_cfg: RewardConfig,
        seed: int = 42,
        mode: str = "train"
    ) -> None:
        self.cfg = gui_cfg
        self.robot_cfg = robot_cfg
        self.env_cfg = env_cfg
        self.rew_cfg = reward_cfg
        self.seed = seed

        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        self.mode: str = mode
        self.episode_count: int = 0
        self.done_pause_until: float = 0.0

        self.env: Environment = self._make_env(train=True)
        self.env.reset_episode(train=True)

        plot_w = self.cfg.window_size[0] - self.cfg.sim_width
        plot_h = self.cfg.window_size[1]
        self.plot_panel = PlotPanel((plot_w, plot_h), update_sec=self.cfg.plot_update_sec)

        self._control_dt = 1.0 / float(self.cfg.control_hz)
        self._accum = 0.0
        self._last_time = time.perf_counter()

    def _make_env(self, *, train: bool) -> Environment:
        model = Model(obs_dim=8, act_dim=2)
        if not train:
            model.set_eval_mode()
        env = Environment(
            env_cfg=self.env_cfg,
            reward_cfg=self.rew_cfg,
            robot_cfg=self.robot_cfg,
            model=model,
            seed=self.seed,
        )
        env.reset_episode(train=train)
        return env

    def _switch_to_test(self) -> None:
        # save training model
        self.env.model.save(self.cfg.model_path, include_optimizer=False, include_metrics=False)

        # build new env + model, load weights, set eval mode
        test_model = Model(obs_dim=8, act_dim=2)
        test_model.load(self.cfg.model_path, load_optimizer=False)
        test_model.set_eval_mode()

        self.env = Environment(
            env_cfg=self.env_cfg,
            reward_cfg=self.rew_cfg,
            robot_cfg=self.robot_cfg,
            model=test_model,
            seed=self.seed,
        )
        self.env.reset_episode(train=False)

        self.mode = "test"
        self.episode_count = 0

    def run(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(self.cfg.window_size)
        pygame.display.set_caption("RL Robot: Train/Test GUI")
        self.clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 20)

        running = True
        paused = False

        while running:
            assert self.screen is not None and self.clock is not None

            # events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused

            now = time.perf_counter()
            dt = now - self._last_time
            self._last_time = now
            self._accum += dt

            # control stepping (fixed rate)
            if not paused and now >= self.done_pause_until:
                while self._accum >= self._control_dt:
                    self._accum -= self._control_dt
                    obs, r, done, info = self.env.step()

                    if done and (not info.get("needs_reset", False)):
                        self.episode_count += 1

                        if self.cfg.pause_on_done_sec > 0:
                            self.done_pause_until = time.perf_counter() + self.cfg.pause_on_done_sec

                        # mode transitions
                        if self.mode == "train" and self.episode_count >= self.cfg.train_episodes:
                            self._switch_to_test()
                            break

                        if self.mode == "test" and self.episode_count >= self.cfg.test_episodes:
                            paused = True  # stop at the end
                            break

            # draw background
            self.screen.fill((245, 245, 245))

            # simulation panel (left)
            sim_rect = pygame.Rect(0, 0, self.cfg.sim_width, self.cfg.window_size[1])
            pygame.draw.rect(self.screen, (255, 255, 255), sim_rect)

            render = self.env.get_render_data()
            self._draw_robot(self.screen, render, offset=(0, 0))

            # separator
            pygame.draw.line(
                self.screen, (200, 200, 200),
                (self.cfg.sim_width, 0),
                (self.cfg.sim_width, self.cfg.window_size[1]),
                2
            )

            # plots panel (right)
            metrics = self.env.get_metrics()
            plot_surface = self.plot_panel.render(metrics, self.mode, now)
            self.screen.blit(plot_surface, (self.cfg.sim_width, 0))

            # HUD text
            hud = f"mode={self.mode}  ep={self.episode_count}  step={render['step']}  dist={render['distance']:.1f}  done={render['done']}  reason={render['reason']}"
            txt = font.render(hud, True, (20, 20, 20))
            self.screen.blit(txt, (10, 10))

            pygame.display.flip()
            self.clock.tick(self.cfg.fps)

        pygame.quit()

    def _draw_robot(self, screen: pygame.Surface, render: Dict[str, Any], offset: Tuple[int, int]) -> None:
        ox, oy = offset
        joints = render["joints"]
        target = render["target"]

        # lines
        pts = [(int(joints[i, 0] + ox), int(joints[i, 1] + oy)) for i in range(joints.shape[0])]
        pygame.draw.lines(screen, (50, 50, 50), False, pts, 6)

        # joints
        for p in pts:
            pygame.draw.circle(screen, (80, 80, 80), p, 8)

        # target
        tx, ty = int(target[0] + ox), int(target[1] + oy)
        pygame.draw.circle(screen, (220, 50, 50), (tx, ty), 6)
        pygame.draw.circle(screen, (220, 50, 50), (tx, ty), int(self.env_cfg.target_thresh), 1)


def main() -> None:
    # You can tweak these defaults to match your layout
    gui_cfg = GUIConfig(
        window_size=(1600, 800),
        sim_width=800,
        fps=60,
        control_hz=30,
        plot_update_sec=0.25,
        train_episodes=500,
        test_episodes=100,
        model_path="best_policy.pt",
        pause_on_done_sec=0.0,
    )

    # Ensure base is in the sim panel
    robot_cfg = RobotConfig(base_xy=(400.0, 400.0), link_lengths=(120, 160), wrap_angles=True, dtheta_max=0.03)

    env_cfg = EnvConfig(
        target_xy=(250.0, 200.0),
        target_thresh=15.0,
        max_steps=200,
        forbid_link_target_intersection=True,
        target_point_radius=2.0,
        use_abs_dist=True,
        normalize_dist=False,
        dist_scale=300.0,
        auto_reset=True,
    )

    rew_cfg = RewardConfig(
        progress_scale=1.0,
        step_penalty=0.01,
        goal_reward=10.0,
        fail_penalty=10.0,
        action_l2_scale=0.0,
        action_delta_scale=0.0,
    )

    app = GUIApp(gui_cfg, robot_cfg, env_cfg, rew_cfg, mode="train")
    app.run()


if __name__ == "__main__":
    main()
