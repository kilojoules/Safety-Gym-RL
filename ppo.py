"""
Continuous-action PPO with Gaussian policy for Safety Gym.

GaussianActor outputs mean of a Normal distribution with learnable log_std.
Critic outputs scalar value estimate.
"""
import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class PPOConfig:
    obs_dim: int = 60
    act_dim: int = 2
    hidden: int = 64
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    clip_ratio: float = 0.2
    train_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    minibatch_size: int = 64


class GaussianActor(nn.Module):
    """MLP policy that outputs mean of a Gaussian; log_std is a free parameter."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = self.log_std.exp()
        return Normal(mean, std)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self.forward(obs)
        return dist.log_prob(action).sum(-1)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self.forward(obs)
        return dist.entropy().sum(-1)


class Critic(nn.Module):
    """MLP value function."""

    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class PPOAgent:
    """Wraps actor + critic + optimizers for PPO training."""

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.actor = GaussianActor(cfg.obs_dim, cfg.act_dim, cfg.hidden)
        self.critic = Critic(cfg.obs_dim, cfg.hidden)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Select action given observation. Returns (action, log_prob) as numpy."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_t, logp_t = self.actor.act(obs_t)
        action = action_t.squeeze(0).numpy()
        action = np.clip(action, -1.0, 1.0)
        return action, logp_t.item()

    def get_value(self, obs: np.ndarray) -> float:
        """Get value estimate for observation."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return self.critic(obs_t).item()

    def update(self, rollout) -> Dict[str, float]:
        """PPO update from a RolloutBuffer.

        Returns dict of loss metrics.
        """
        obs = torch.as_tensor(rollout.observations, dtype=torch.float32)
        acts = torch.as_tensor(rollout.actions, dtype=torch.float32)
        old_logps = torch.as_tensor(rollout.log_probs, dtype=torch.float32)
        advantages = torch.as_tensor(rollout.advantages, dtype=torch.float32)
        returns = torch.as_tensor(rollout.returns, dtype=torch.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.cfg.train_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.cfg.minibatch_size):
                end = min(start + self.cfg.minibatch_size, n)
                idx = indices[start:end]

                mb_obs = obs[idx]
                mb_acts = acts[idx]
                mb_old_logps = old_logps[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]

                # Policy loss
                new_logps = self.actor.log_prob(mb_obs, mb_acts)
                ratio = (new_logps - mb_old_logps).exp()
                clip_ratio = torch.clamp(ratio, 1 - self.cfg.clip_ratio,
                                         1 + self.cfg.clip_ratio)
                pg_loss = -torch.min(ratio * mb_adv, clip_ratio * mb_adv).mean()

                # Entropy bonus
                entropy = self.actor.entropy(mb_obs).mean()

                # Actor update
                actor_loss = pg_loss - self.cfg.entropy_coef * entropy
                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_optim.step()

                # Value loss
                values = self.critic(mb_obs)
                v_loss = (values - mb_ret).pow(2).mean()
                self.critic_optim.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_optim.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        return {
            "pg_loss": total_pg_loss / max(num_updates, 1),
            "v_loss": total_v_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def get_state(self) -> dict:
        """Return serializable state for zoo snapshots."""
        return {
            "actor": copy.deepcopy(self.actor.state_dict()),
            "critic": copy.deepcopy(self.critic.state_dict()),
        }

    def load_state(self, state: dict):
        """Load state from zoo snapshot."""
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
