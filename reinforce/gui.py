from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import numpy as np
import pygame

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .config import RobotConfig, EnvConfig, RewardConfig, GUIConfig, ModelConfig
from .env import Environment
from .model import Model


class PlotPanel:
    def __init__(self, size: Tuple[int, int], update_every: int = 10) -> None:
        self.w, self.h = size
        self.update_every = max(1, int(update_every))
        self._frame = 0
        self.surface: Optional[pygame.Surface] = None
        self.win = 50
        self._max_plot_points = 500

        self.fig = Figure(figsize=(self.w / 100.0, self.h / 100.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self._create_axes()
        self.fig.tight_layout(pad=1.0)

    def _create_axes(self) -> None:
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)

    def render(self, metrics: Dict[str, Any], mode: str) -> pygame.Surface:
        self._frame += 1
        if self.surface is not None and (self._frame % self.update_every) != 0:
            return self.surface

        self._draw(metrics, mode)

        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()

        rgba = np.asarray(self.canvas.buffer_rgba())
        rgb = rgba[..., :3]
        surf = pygame.surfarray.make_surface(np.transpose(np.ascontiguousarray(rgb), (1, 0, 2)))
        self.surface = surf
        return surf

    def _draw(self, metrics: Dict[str, Any], mode: str) -> None:
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        if mode == "train":
            train = metrics["train"]
            r = np.asarray(train["total_reward"], dtype=np.float32)
            s = np.asarray(train["success"], dtype=np.float32)
            steps = np.asarray(train["steps"], dtype=np.float32)

            if r.size:
                self.ax1.plot(*self._downsample(r), alpha=0.3, linewidth=0.5, color="#2196F3")
                self.ax1.plot(*self._downsample(self._running_mean(r, 10)), linewidth=2, color="#1565C0")
            self.ax1.set_title("train: total_reward (ma=10)")
            self.ax1.grid(True, alpha=0.3)

            if s.size:
                cs = np.cumsum(s)
                ma = np.empty_like(cs)
                ma[: self.win] = cs[: self.win] / np.arange(
                    1, min(self.win, s.size) + 1, dtype=np.float32
                )
                if s.size > self.win:
                    ma[self.win :] = (cs[self.win :] - cs[: -self.win]) / self.win
                self.ax2.plot(*self._downsample(ma), linewidth=2, color="#4CAF50")

            self.ax2.set_ylim(-0.05, 1.05)
            self.ax2.set_title(f"train: success rate (window={self.win})")
            self.ax2.grid(True, alpha=0.3)

            if steps.size:
                self.ax3.plot(*self._downsample(steps), alpha=0.3, linewidth=0.5, color="#FF9800")
                self.ax3.plot(*self._downsample(self._running_mean(steps, 10)), linewidth=2, color="#E65100")
            self.ax3.set_title("train: steps/episode (ma=10)")
            self.ax3.grid(True, alpha=0.3)

        else:
            test = metrics["test"]
            s = np.asarray(test["success"], dtype=np.float32)
            dist = np.asarray(test["final_distance"], dtype=np.float32)
            steps = np.asarray(test["steps"], dtype=np.float32)

            if s.size:
                csr = np.cumsum(s) / np.arange(1, s.size + 1, dtype=np.float32)
                self.ax1.plot(csr, linewidth=2, color="#4CAF50")
                self.ax1.set_ylim(-0.05, 1.05)
            self.ax1.set_title("test: cumulative success rate")
            self.ax1.grid(True, alpha=0.3)

            if dist.size:
                self.ax2.plot(dist, color="#F44336")
                self.ax2.set_title(f"test: final_distance (avg={float(dist.mean()):.2f})")
            else:
                self.ax2.set_title("test: final_distance")
            self.ax2.grid(True, alpha=0.3)

            if steps.size:
                self.ax3.plot(steps, color="#FF9800")
                self.ax3.set_title(f"test: steps (avg={float(steps.mean()):.1f})")
            else:
                self.ax3.set_title("test: steps")
            self.ax3.grid(True, alpha=0.3)

    def _downsample(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = arr.shape[0]
        if n <= self._max_plot_points:
            return np.arange(n), arr
        idx = np.linspace(0, n - 1, self._max_plot_points, dtype=int)
        return idx, arr[idx]

    @staticmethod
    def _running_mean(arr: np.ndarray, win: int) -> np.ndarray:
        cs = np.cumsum(arr)
        ma = np.empty_like(cs)
        ma[:win] = cs[:win] / np.arange(1, min(win, arr.size) + 1, dtype=np.float32)
        if arr.size > win:
            ma[win:] = (cs[win:] - cs[:-win]) / win
        return ma


class ExtendedPlotPanel(PlotPanel):
    def _create_axes(self) -> None:
        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1.2])
        self.axes = [
            self.fig.add_subplot(gs[0, 0]),
            self.fig.add_subplot(gs[0, 1]),
            self.fig.add_subplot(gs[1, 0]),
            self.fig.add_subplot(gs[1, 1]),
            self.fig.add_subplot(gs[2, 0]),
            self.fig.add_subplot(gs[2, 1]),
            self.fig.add_subplot(gs[3, :]),
        ]

    def _draw(self, metrics: Dict[str, Any], mode: str) -> None:
        for ax in self.axes:
            ax.clear()

        if mode != "train":
            return

        train = metrics["train"]
        specs = [
            ("total_reward", "total reward (ma=10)", True, "#2196F3", "#1565C0"),
            ("steps", "steps/episode (ma=10)", True, "#FF9800", "#E65100"),
            ("final_distance", "final distance (ma=10)", True, "#F44336", "#B71C1C"),
            ("loss", "policy loss (ma=10)", True, "#9C27B0", "#6A1B9A"),
            ("baseline", "baseline", False, "#00838F", "#00838F"),
            ("grad_norm", "grad norm (ma=10)", True, "#795548", "#4E342E"),
            ("success", f"success rate (window)={self.win}", False, "#4CAF50", "#4CAF50"),
        ]

        for ax, (key, title, use_ma, c_raw, c_ma) in zip(self.axes, specs):
            arr = np.asarray(train[key], dtype=np.float32)
            if key == "success":
                if arr.size:
                    cs = np.cumsum(arr)
                    ma = np.empty_like(cs)
                    ma[: self.win] = cs[: self.win] / np.arange(
                        1, min(self.win, arr.size) + 1, dtype=np.float32
                    )
                    if arr.size > self.win:
                        ma[self.win :] = (cs[self.win :] - cs[: -self.win]) / self.win
                    ax.plot(*self._downsample(ma), linewidth=2, color=c_ma)
                ax.set_ylim(-0.05, 1.05)
            elif arr.size:
                if use_ma:
                    ax.plot(*self._downsample(arr), alpha=0.3, linewidth=0.5, color=c_raw)
                    ax.plot(*self._downsample(self._running_mean(arr, 10)), linewidth=2, color=c_ma)
                else:
                    ax.plot(*self._downsample(arr), linewidth=1.5, color=c_raw)
            ax.set_title(title, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)


class GUI:
    def __init__(
        self,
        *,
        gui_cfg: GUIConfig,
        robot_cfg: RobotConfig,
        env_cfg: EnvConfig,
        reward_cfg: RewardConfig,
        model_cfg: ModelConfig,
        seed: int = 42,
        start_mode: str = "train",
        no_sim_train: bool = False,
        extended: bool = False,
    ) -> None:
        self.cfg = gui_cfg
        self.robot_cfg = robot_cfg
        self.env_cfg = env_cfg
        self.rew_cfg = reward_cfg
        self.model_cfg = model_cfg
        self.seed = seed
        self.extended = bool(extended)

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

        sim_w = 0 if (self.mode == "train" and self.no_sim_train) else self.cfg.sim_width
        plot_w = self.cfg.window_size[0] - sim_w
        plot_h = self.cfg.window_size[1]
        self.plot_panel = self._make_plot_panel(plot_w, plot_h)

    def _make_plot_panel(self, w: int, h: int) -> PlotPanel:
        if self.extended and self.no_sim_train and self.mode == "train":
            return ExtendedPlotPanel((w, h), update_every=self.cfg.plot_update_every)
        return PlotPanel((w, h), update_every=self.cfg.plot_update_every)

    def _make_env(self, *, train: bool, load_model: bool) -> Environment:
        model = Model(
            obs_dim=8,
            act_dim=2,
            cfg=self.model_cfg,
            action_limit=self.robot_cfg.dtheta_max,
            train_episodes=self.cfg.train_episodes,
        )
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
        self.env.model.save(self.cfg.model_path, include_optimizer=False, include_metrics=False)

        self.mode = "test"
        self.episode_count = 0
        self.pause_frames_left = 0

        self.env = self._make_env(train=False, load_model=True)
        self.env.reset_episode(train=False)

        sim_w = self.cfg.sim_width
        plot_w = self.cfg.window_size[0] - sim_w
        plot_h = self.cfg.window_size[1]
        self.plot_panel = self._make_plot_panel(plot_w, plot_h)

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

            if (not paused) and self.pause_frames_left <= 0:
                spf = (
                    self.cfg.steps_per_frame_no_sim
                    if (self.mode == "train" and self.no_sim_train)
                    else self.cfg.steps_per_frame
                )

                for _ in range(spf):
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

                        if self.mode == "train" and self.episode_count >= self.cfg.train_episodes:
                            self._switch_to_test()
                            break

                        elif self.mode == "test" and self.episode_count >= self.cfg.test_episodes:
                            paused = True
                            break

            elif self.pause_frames_left > 0:
                self.pause_frames_left -= 1

            if not running:
                continue

            self.screen.fill((245, 245, 245))

            show_sim = not (self.mode == "train" and self.no_sim_train)
            sim_w = self.cfg.sim_width if show_sim else 0
            plot_x = sim_w

            render = self.env.get_render_data()
            if show_sim:
                sim_rect = pygame.Rect(0, 0, sim_w, self.cfg.window_size[1])
                pygame.draw.rect(self.screen, (255, 255, 255), sim_rect)
                self._draw_robot(self.screen, render, offset=(0, 0))

                pygame.draw.line(
                    self.screen, (200, 200, 200), (sim_w, 0), (sim_w, self.cfg.window_size[1]), 2
                )

            metrics = self.env.get_metrics()
            plot_surface = self.plot_panel.render(metrics, self.mode)
            self.screen.blit(plot_surface, (plot_x, 0))

            s = metrics["train"]["success"]
            win = min(self.plot_panel.win, len(s))
            s_rate = sum(s[-win:]) / win if s else 0
            hud = (
                f"mode={self.mode}  ep={self.episode_count}  step={render['step']}  dist={render['distance']:.1f}"
                + (f"succcess={s_rate:.3f}" if self.mode == "train" else "")
            )
            txt = font.render(hud, True, (20, 20, 20))
            self.screen.blit(txt, (10, 10))

            pygame.display.flip()

            if self.mode == "test":
                self.clock.tick(10)

        import matplotlib.pyplot as plt
        plt.close(self.plot_panel.fig)
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


if __name__ == "__main__":
    raise RuntimeError("Run main.py instead.")
