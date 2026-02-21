"""
Buffered agent with trajectory replay buffer and importance-weighted PPO.

Stores rollouts in a FIFO buffer and trains from sampled batches with
importance sampling correction for off-policy data.
"""
import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from ppo import GaussianActor, Critic, PPOConfig


@dataclass
class BufferedConfig:
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
    buffer_size: int = 20000
    is_ratio_clip: float = 2.0


class TrajectoryReplayBuffer:
    """FIFO buffer storing flattened rollout data."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.advantages = []
        self.returns = []
        self._size = 0

    def add(self, rollout):
        """Add rollout data to buffer."""
        n = len(rollout.observations)
        self.observations.append(rollout.observations)
        self.actions.append(rollout.actions)
        self.log_probs.append(rollout.log_probs)
        self.advantages.append(rollout.advantages)
        self.returns.append(rollout.returns)
        self._size += n

        # Trim if over capacity
        while self._size > self.max_size and len(self.observations) > 1:
            removed = len(self.observations[0])
            self.observations.pop(0)
            self.actions.pop(0)
            self.log_probs.pop(0)
            self.advantages.pop(0)
            self.returns.pop(0)
            self._size -= removed

    def sample(self, batch_size: int) -> Tuple:
        """Sample a random batch from the buffer."""
        all_obs = np.concatenate(self.observations)
        all_acts = np.concatenate(self.actions)
        all_logps = np.concatenate(self.log_probs)
        all_adv = np.concatenate(self.advantages)
        all_ret = np.concatenate(self.returns)

        idx = np.random.choice(len(all_obs), size=min(batch_size, len(all_obs)),
                               replace=False)
        return (all_obs[idx], all_acts[idx], all_logps[idx],
                all_adv[idx], all_ret[idx])

    def get_all(self) -> Tuple:
        """Return all data in the buffer."""
        return (
            np.concatenate(self.observations),
            np.concatenate(self.actions),
            np.concatenate(self.log_probs),
            np.concatenate(self.advantages),
            np.concatenate(self.returns),
        )

    def __len__(self):
        return self._size


class BufferedAgent:
    """PPO agent with trajectory replay buffer and IS correction."""

    def __init__(self, cfg: BufferedConfig):
        self.cfg = cfg
        self.actor = GaussianActor(cfg.obs_dim, cfg.act_dim, cfg.hidden)
        self.critic = Critic(cfg.obs_dim, cfg.hidden)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.buffer = TrajectoryReplayBuffer(cfg.buffer_size)

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_t, logp_t = self.actor.act(obs_t)
        action = action_t.squeeze(0).numpy()
        action = np.clip(action, -1.0, 1.0)
        return action, logp_t.item()

    def get_value(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return self.critic(obs_t).item()

    def store(self, rollout):
        """Store a rollout in the replay buffer."""
        self.buffer.add(rollout)

    def update(self) -> Dict[str, float]:
        """Update from buffered data with importance sampling correction."""
        if len(self.buffer) < self.cfg.minibatch_size:
            return {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "is_ratio": 1.0}

        all_obs, all_acts, all_old_logps, all_adv, all_ret = self.buffer.get_all()

        obs = torch.as_tensor(all_obs, dtype=torch.float32)
        acts = torch.as_tensor(all_acts, dtype=torch.float32)
        old_logps = torch.as_tensor(all_old_logps, dtype=torch.float32)
        advantages = torch.as_tensor(all_adv, dtype=torch.float32)
        returns = torch.as_tensor(all_ret, dtype=torch.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        total_is_ratio = 0.0
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

                # Current policy log probs
                new_logps = self.actor.log_prob(mb_obs, mb_acts)

                # Importance sampling ratio
                is_ratio = (new_logps - mb_old_logps).exp()
                # Clip IS ratio
                is_ratio_clipped = torch.clamp(is_ratio, 0.0, self.cfg.is_ratio_clip)

                # PPO clipped objective with IS correction
                ratio = is_ratio_clipped
                clip_ratio = torch.clamp(ratio, 1 - self.cfg.clip_ratio,
                                         1 + self.cfg.clip_ratio)
                pg_loss = -torch.min(ratio * mb_adv, clip_ratio * mb_adv).mean()

                entropy = self.actor.entropy(mb_obs).mean()

                actor_loss = pg_loss - self.cfg.entropy_coef * entropy
                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_optim.step()

                # Value loss with IS weighting
                values = self.critic(mb_obs)
                v_loss = (is_ratio_clipped.detach() * (values - mb_ret).pow(2)).mean()
                self.critic_optim.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_optim.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy.item()
                total_is_ratio += is_ratio.mean().item()
                num_updates += 1

        return {
            "pg_loss": total_pg_loss / max(num_updates, 1),
            "v_loss": total_v_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "is_ratio": total_is_ratio / max(num_updates, 1),
        }

    def get_state(self) -> dict:
        return {
            "actor": copy.deepcopy(self.actor.state_dict()),
            "critic": copy.deepcopy(self.critic.state_dict()),
        }

    def load_state(self, state: dict):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
