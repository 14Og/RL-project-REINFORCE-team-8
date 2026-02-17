"""gui.py

Pygame "god GUI" that visualizes:
- simulation panel (robot arm + target) using pygame
- plots panel (matplotlib rendered offscreen, blitted into pygame)

Modes:
- --train: run training, save model, then automatically run test (no learning)
- --test: load model and run test only (no learning)

Important changes per your request:
- removed time-based control stepping / accumulators
- env.step() is called as fast as the main loop can run
- plot panel updates by frame count (not time)
- added argparse: --train / --test and --no-sim (train only)

This file intentionally keeps env/model/robot free of any pygame/matplotlib deps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pygame

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from config import RobotConfig, EnvConfig, RewardConfig, GUIConfig
from env import Environment
from model import Model


class PlotPanel:
    """Renders metrics into a pygame.Surface via offscreen matplotlib Agg."""

    def __init__(self, size: Tuple[int, int], update_every: int = 10) -> None:
        self.w, self.h = size
        self.update_every = max(1, int(update_every))
        self._frame = 0
        self.surface: Optional[pygame.Surface] = None
        self.win = 50
        self._max_plot_points = 500  # downsample arrays beyond this for speed

        # dpi=100 => figsize in inches = pixels / 100
        self.fig = Figure(figsize=(self.w / 100.0, self.h / 100.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)

        # 3 stacked plots
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        self.fig.tight_layout(pad=1.0)

    def render(self, metrics: Dict[str, Any], mode: str) -> pygame.Surface:
        self._frame += 1
        if self.surface is not None and (self._frame % self.update_every) != 0:
            return self.surface

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        if mode == "train":
            train = metrics["train"]
            r = np.asarray(train["total_reward"], dtype=np.float32)
            s = np.asarray(train["success"], dtype=np.float32)
            steps = np.asarray(train["steps"], dtype=np.float32)

            if r.size:
                self.ax1.plot(*self._downsample(r), alpha=0.3, linewidth=0.5)
                self.ax1.plot(*self._downsample(self._running_mean(r, 10)), linewidth=2)
            self.ax1.set_title("train: total_reward (ma=10)")
            self.ax1.grid(True, alpha=0.3)

            if s.size:
                # uniform moving average with expanding window for early episodes
                cs = np.cumsum(s)
                ma = np.empty_like(cs)
                ma[: self.win] = cs[: self.win] / np.arange(
                    1, min(self.win, s.size) + 1, dtype=np.float32
                )
                if s.size > self.win:
                    ma[self.win :] = (cs[self.win :] - cs[: -self.win]) / self.win
                self.ax2.plot(*self._downsample(ma), linewidth=2)

            self.ax2.set_ylim(-0.05, 1.05)
            self.ax2.set_title(f"train: success rate (window={self.win})")
            self.ax2.grid(True, alpha=0.3)

            if steps.size:
                self.ax3.plot(*self._downsample(steps), alpha=0.3, linewidth=0.5)
                self.ax3.plot(*self._downsample(self._running_mean(steps, 10)), linewidth=2)
            self.ax3.set_title("train: steps/episode (ma=10)")
            self.ax3.grid(True, alpha=0.3)

        else:  # test
            test = metrics["test"]
            s = np.asarray(test["success"], dtype=np.float32)
            dist = np.asarray(test["final_distance"], dtype=np.float32)
            steps = np.asarray(test["steps"], dtype=np.float32)

            if s.size:
                csr = np.cumsum(s) / np.arange(1, s.size + 1, dtype=np.float32)
                self.ax1.plot(csr, linewidth=2)
                self.ax1.set_ylim(-0.05, 1.05)
            self.ax1.set_title("test: cumulative success rate")
            self.ax1.grid(True, alpha=0.3)

            if dist.size:
                self.ax2.plot(dist)
                self.ax2.set_title(f"test: final_distance (avg={float(dist.mean()):.2f})")
            else:
                self.ax2.set_title("test: final_distance")
            self.ax2.grid(True, alpha=0.3)

            if steps.size:
                self.ax3.plot(steps)
                self.ax3.set_title(f"test: steps (avg={float(steps.mean()):.1f})")
            else:
                self.ax3.set_title("test: steps")
            self.ax3.grid(True, alpha=0.3)

        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()

        rgba = np.asarray(self.canvas.buffer_rgba())  # (h, w, 4)
        rgb = rgba[..., :3]
        surf = pygame.surfarray.make_surface(np.transpose(np.ascontiguousarray(rgb), (1, 0, 2)))
        self.surface = surf
        return surf

    def _downsample(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x, y) with at most _max_plot_points points, using strided sampling."""
        n = arr.shape[0]
        if n <= self._max_plot_points:
            return np.arange(n), arr
        idx = np.linspace(0, n - 1, self._max_plot_points, dtype=int)
        return idx, arr[idx]

    @staticmethod
    def _running_mean(arr: np.ndarray, win: int) -> np.ndarray:
        """Expanding-then-sliding running mean, same length as input."""
        cs = np.cumsum(arr)
        ma = np.empty_like(cs)
        ma[:win] = cs[:win] / np.arange(1, min(win, arr.size) + 1, dtype=np.float32)
        if arr.size > win:
            ma[win:] = (cs[win:] - cs[:-win]) / win
        return ma


class GUIApp:
    def __init__(
        self,
        *,
        gui_cfg: GUIConfig,
        robot_cfg: RobotConfig,
        env_cfg: EnvConfig,
        reward_cfg: RewardConfig,
        seed: int = 42,
        start_mode: str = "train",
        no_sim_train: bool = False,
    ) -> None:
        self.cfg = gui_cfg
        self.robot_cfg = robot_cfg
        self.env_cfg = env_cfg
        self.rew_cfg = reward_cfg
        self.seed = seed

        self.mode: str = start_mode
        self.no_sim_train = bool(no_sim_train)

        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        self.env: Environment = self._make_env(
            train=(self.mode == "train"), load_model=(self.mode == "test")
        )
        self.env.reset_episode(train=(self.mode == "train"))

        self.episode_count: int = 0
        self.pause_frames_left: int = 0

        # Plot panel width depends on whether we hide sim in train
        sim_w = 0 if (self.mode == "train" and self.no_sim_train) else self.cfg.sim_width
        plot_w = self.cfg.window_size[0] - sim_w
        plot_h = self.cfg.window_size[1]
        self.plot_panel = PlotPanel((plot_w, plot_h), update_every=self.cfg.plot_update_every)

    def _make_env(self, *, train: bool, load_model: bool) -> Environment:
        model = Model(obs_dim=8, act_dim=2, action_limit=self.robot_cfg.dtheta_max)
        if load_model:
            model.load(self.cfg.model_path, load_optimizer=False)
            model.set_eval_mode()
        elif not train:
            model.set_eval_mode()

        env = Environment(
            env_cfg=self.env_cfg,
            reward_cfg=self.rew_cfg,
            robot_cfg=self.robot_cfg,
            model=model,
            seed=self.seed,
        )
        return env

    def _switch_to_test(self) -> None:
        # save policy and create a new env with loaded weights
        self.env.model.save(self.cfg.model_path, include_optimizer=False, include_metrics=False)

        self.mode = "test"
        self.episode_count = 0
        self.pause_frames_left = 0

        self.env = self._make_env(train=False, load_model=True)
        self.env.reset_episode(train=False)

        # re-init plot panel width (simulation is always shown in test)
        sim_w = self.cfg.sim_width
        plot_w = self.cfg.window_size[0] - sim_w
        plot_h = self.cfg.window_size[1]
        self.plot_panel = PlotPanel((plot_w, plot_h), update_every=self.cfg.plot_update_every)

    def run(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(self.cfg.window_size)
        pygame.display.set_caption("RL Robot: Train/Test GUI")
        self.clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 20)

        running = True
        paused = False
        frame = 0

        while running:
            assert self.screen is not None and self.clock is not None
            frame += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused

            # run multiple env steps per rendered frame for speed
            if (not paused) and self.pause_frames_left <= 0:
                spf = (
                    self.cfg.steps_per_frame_no_sim
                    if (self.mode == "train" and self.no_sim_train)
                    else self.cfg.steps_per_frame
                )
                if self.mode == "test":
                    spf = 1  # always 1 step per frame in test for visibility

                for _ in range(spf):
                    # pump events periodically so the OS doesn't think we're frozen
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                            elif event.key == pygame.K_SPACE:
                                paused = not paused
                    if not running or paused:
                        break

                    obs, r, done, info = self.env.step()

                    if done and (not info.get("needs_reset", False)):
                        self.episode_count += 1

                        if self.cfg.pause_on_done_frames > 0:
                            self.pause_frames_left = int(self.cfg.pause_on_done_frames)
                            break

                        # mode transitions
                        if self.mode == "train" and self.episode_count >= self.cfg.train_episodes:
                            self._switch_to_test()
                            break

                        elif self.mode == "test" and self.episode_count >= self.cfg.test_episodes:
                            paused = True
                            break

            elif self.pause_frames_left > 0:
                self.pause_frames_left -= 1

            # draw
            self.screen.fill((245, 245, 245))

            # decide layout for this frame
            show_sim = not (self.mode == "train" and self.no_sim_train)
            sim_w = self.cfg.sim_width if show_sim else 0
            plot_x = sim_w

            # simulation panel (left)
            render = self.env.get_render_data()
            if show_sim:
                sim_rect = pygame.Rect(0, 0, sim_w, self.cfg.window_size[1])
                pygame.draw.rect(self.screen, (255, 255, 255), sim_rect)
                self._draw_robot(self.screen, render, offset=(0, 0))

                # separator
                pygame.draw.line(
                    self.screen, (200, 200, 200), (sim_w, 0), (sim_w, self.cfg.window_size[1]), 2
                )

            # plots panel (right or full screen)
            metrics = self.env.get_metrics()
            plot_surface = self.plot_panel.render(metrics, self.mode)
            self.screen.blit(plot_surface, (plot_x, 0))

            # HUD
            s = metrics["train"]["success"]
            win = min(self.plot_panel.win, len(s))
            s_rate = sum(s[-win:]) / win if s else 0
            hud = (
                f"mode={self.mode}  ep={self.episode_count}  step={render['step']}  dist={render['distance']:.1f}" \
                + (f"succcess={s_rate:.3f}"
                if self.mode == "train"
                else "")
            )
            txt = font.render(hud, True, (20, 20, 20))
            self.screen.blit(txt, (10, 10))

            pygame.display.flip()

            if self.mode == "test":
                self.clock.tick(20)

        pygame.quit()

    def _draw_robot(
        self, screen: pygame.Surface, render: Dict[str, Any], offset: Tuple[int, int]
    ) -> None:
        ox, oy = offset
        joints = render["joints"]
        target = render["target"]

        pts = [(int(joints[i, 0] + ox), int(joints[i, 1] + oy)) for i in range(joints.shape[0])]
        pygame.draw.lines(screen, (50, 50, 50), False, pts, 6)

        for p in pts:
            pygame.draw.circle(screen, (80, 80, 80), p, 8)

        tx, ty = int(target[0] + ox), int(target[1] + oy)
        pygame.draw.circle(screen, (220, 50, 50), (tx, ty), 6)
        pygame.draw.circle(screen, (220, 50, 50), (tx, ty), int(self.env_cfg.target_thresh), 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--train", action="store_true", help="Train policy, save, then run test.")
    g.add_argument(
        "--test", action="store_true", help="Run test only (loads model from --model-path)."
    )

    p.add_argument(
        "--no-sim", action="store_true", help="(train only) Hide simulation panel; show plots only."
    )
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--train-episodes", type=int, default=None)
    p.add_argument("--test-episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:

    args = parse_args()

    robot_cfg = RobotConfig()
    env_cfg = EnvConfig()
    rew_cfg = RewardConfig()
    gui_cfg = GUIConfig()

    gui_cfg.train_episodes = args.train_episodes or gui_cfg.train_episodes
    gui_cfg.test_episodes = args.test_episodes or gui_cfg.test_episodes
    gui_cfg.model_path = args.model_path or gui_cfg.model_path

    start_mode = "train" if args.train else "test"
    no_sim_train = bool(args.no_sim) if args.train else False

    app = GUIApp(
        gui_cfg=gui_cfg,
        robot_cfg=robot_cfg,
        env_cfg=env_cfg,
        reward_cfg=rew_cfg,
        seed=int(args.seed),
        start_mode=start_mode,
        no_sim_train=no_sim_train,
    )
    app.run()


if __name__ == "__main__":
    main()
