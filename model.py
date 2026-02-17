"""model.py

Minimal Model class for a pure REINFORCE exercise (baseline allowed).
Model outputs raw action u (unbounded). Robot applies its own constraints internally

Typical usage (driven by Environment loop):

  model.start_episode()
  s = env.reset()
  while not done:
      u = model.select_action(s, train=True)  # raw u
      s_next, r, done, info = env.step(u)
      model.observe(r)
      s = s_next
  model.finish_episode(success=info["success"], final_distance=info["final_distance"])

For testing (also driven externally), just record outcomes:
  model.record_test_episode(success=..., final_distance=..., steps=...)

State default:
- obs_dim=8 (Robot obs 6 + Env appends dx,dy to target).

Action default:
- act_dim=2 (raw u for 2 joints).
"""

from __future__ import annotations

from state import State

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union

class GaussianMLPPolicy(nn.Module):
    """Diagonal Gaussian policy with tanh-bounded mean and log-std sigma."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        action_limit: float = 0.1,
        log_std_min: float = -3.0,
        log_std_max: float = -0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, act_dim)
        self.log_std_head = nn.Linear(in_dim, act_dim)

        self.action_limit = float(action_limit)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.net(x)
        mu = torch.tanh(self.mu_head(z)) * self.action_limit
        log_std = torch.clamp(self.log_std_head(z), self.log_std_min, self.log_std_max)
        sigma = torch.exp(log_std) * self.action_limit
        return mu, sigma


class Model:
    """Pure REINFORCE + moving-average baseline. Stores metrics for train/test."""

    def __init__(
        self,
        obs_dim: int = 8,
        act_dim: int = 2,
        gamma: float = 0.99,
        lr: float = 1e-4,
        lr_min: float = 1e-5,
        total_episodes: int = 5000,
        baseline_window: int = 200,
        grad_clip_norm: float = 1.0,
        device: Optional[str] = None,
        policy: Optional[nn.Module] = None,
        # used only if policy is None
        hidden_sizes: Tuple[int, ...] = (128, 128),
        action_limit: float = 0.1,
        log_std_min: float = -3.0,
        log_std_max: float = -0.5,
    ) -> None:
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1].")

        self.gamma = float(gamma)
        self.grad_clip_norm = float(grad_clip_norm)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if policy is None:
            policy = GaussianMLPPolicy(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_sizes=hidden_sizes,
                action_limit=action_limit,
                log_std_min=log_std_min,
                log_std_max=log_std_max,
            )
        self.policy = policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(lr))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=int(total_episodes), eta_min=float(lr_min)
        )

        # moving-average baseline over recent episode returns (return-to-go from t=0)
        self.baseline_buffer: Deque[float] = deque(maxlen=int(baseline_window))

        # episode buffers
        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []
        self._steps: int = 0
        
        # data for plotting elsewhere (per-episode)
        self.train: Dict[str, List[float]] = {
            "total_reward": [],
            "success": [],
            "steps": [],
            "final_distance": [],
            "loss": [],
            "baseline": [],
            "grad_norm": [],
        }

        # test data (per-episode). compute rates/averages outside
        self.test: Dict[str, List[float]] = {
            "success": [],
            "final_distance": [],
            "steps": [],
        }

    # ---------------- episode api (called by env loop) ----------------

    def start_episode(self) -> None:
        self._log_probs.clear()
        self._rewards.clear()
        self._steps = 0

    def select_action(self, state: Union[np.ndarray, State], *, train: bool = True) -> np.ndarray:
        """Return raw action u.

        If train=True, samples u ~ N(mu, sigma) and records log_prob(u|s).
        If train=False, returns deterministic mu (no recording).
        """
        s = self._to_tensor(state)
        mu, sigma = self.policy(s)
        dist = Normal(mu, sigma)

        if train:
            u = dist.sample()
            logp = dist.log_prob(u).sum(dim=-1)
            self._log_probs.append(logp.squeeze(0))
            self._steps += 1
            return u.squeeze(0).detach().cpu().numpy()

        return mu.squeeze(0).detach().cpu().numpy()

    @torch.no_grad()
    def act(self, state: Union[np.ndarray, Sequence[float]], *, deterministic: bool = True) -> np.ndarray:
        """Raw action for rendering/animation (no recording)."""
        s = self._to_tensor(state)
        mu, sigma = self.policy(s)
        u = mu if deterministic else Normal(mu, sigma).sample()
        return u.squeeze(0).cpu().numpy()

    def observe(self, reward: float) -> None:
        """Append per-step reward produced by Environment."""
        self._rewards.append(float(reward))

    def finish_episode(self, *, success: bool, final_distance: Optional[float] = None) -> Dict[str, float]:
        """Run one REINFORCE update and append training metrics."""
        total_reward = float(sum(self._rewards))
        baseline = float(np.mean(self.baseline_buffer)) if self.baseline_buffer else 0.0

        # if no steps were taken, just log
        if not self._rewards or not self._log_probs:
            metrics = {
                "total_reward": total_reward,
                "success": float(bool(success)),
                "steps": float(self._steps),
                "final_distance": float(final_distance) if final_distance is not None else float("nan"),
                "loss": float("nan"),
                "baseline": baseline,
                "grad_norm": float("nan"),
            }
            self._append_train(metrics)
            return metrics

        returns = self._discounted_returns(self._rewards, self.gamma).to(self.device)  # (T,)
        episode_return = float(returns[0].item())

        advantages = returns - baseline  # baseline-only
        # normalise advantages for variance reduction
        # if advantages.numel() > 1:
        #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        logps = torch.stack(self._log_probs).to(self.device)  # (T,)

        loss = -(logps * advantages).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()

        # update baseline after update
        self.baseline_buffer.append(episode_return)

        metrics = {
            "total_reward": total_reward,
            "success": float(bool(success)),
            "steps": float(self._steps),
            "final_distance": float(final_distance) if final_distance is not None else float("nan"),
            "loss": float(loss.item()),
            "baseline": baseline,
            "grad_norm": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
        }
        self._append_train(metrics)
        return metrics

    # ---------------- test data recording (no loops) ----------------

    def record_test_episode(self, *, success: bool, final_distance: float, steps: int) -> None:
        """Call this from your external test/animation loop."""
        self.test["success"].append(float(bool(success)))
        self.test["final_distance"].append(float(final_distance))
        self.test["steps"].append(float(steps))

    # ---------------- data accessors ----------------

    def get_train_metrics(self) -> Dict[str, List[float]]:
        return self.train

    def get_test_metrics(self) -> Dict[str, List[float]]:
        return self.test

    # ---------------- internals ----------------

    def _append_train(self, m: Dict[str, float]) -> None:
        for k in self.train.keys():
            self.train[k].append(float(m[k]))

    def _to_tensor(self, state: Union[np.ndarray, State]) -> torch.Tensor:
        if not isinstance(state, np.ndarray):
            state = np.asarray(state, dtype=np.float32)
        return torch.from_numpy(state).float().unsqueeze(0).to(self.device)

    @staticmethod
    def _discounted_returns(rewards: List[float], gamma: float) -> torch.Tensor:
        out: List[float] = []
        R = 0.0
        for r in reversed(rewards):
            R = float(r) + float(gamma) * R
            out.append(R)
        out.reverse()
        return torch.tensor(out, dtype=torch.float32)

    # ---------------- persistence ----------------

    def save(self, path: str, *, include_optimizer: bool = False, include_metrics: bool = False) -> None:
        """Save policy (and optionally optimizer + metrics) to a .pt file."""
        ckpt = {
            "policy_state_dict": self.policy.state_dict(),
        }
        if include_optimizer:
            ckpt["optimizer_state_dict"] = self.optimizer.state_dict()
        if include_metrics:
            ckpt["train_metrics"] = self.train
            ckpt["test_metrics"] = self.test
            ckpt["baseline_buffer"] = list(self.baseline_buffer)
        torch.save(ckpt, path)

    def load(self, path: str, *, load_optimizer: bool = False, strict: bool = True) -> None:
        """Load policy weights from a .pt file."""
        ckpt = torch.load(path, map_location=self.device)
        if "policy_state_dict" not in ckpt:
            raise ValueError("Checkpoint missing 'policy_state_dict'.")
        self.policy.load_state_dict(ckpt["policy_state_dict"], strict=strict)
        if load_optimizer and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    def set_train_mode(self) -> None:
        """Put policy module into train() mode."""
        self.policy.train()

    def set_eval_mode(self) -> None:
        """Put policy module into eval() mode."""
        self.policy.eval()

