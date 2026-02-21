"""
Safety Gym environment wrapper with adversarial action perturbation.

The adversary outputs δ ∈ [-1,1]², and the executed action is
clip(agent_action + ε·δ, -1, 1).
"""
import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class EnvConfig:
    env_id: str = "SafetyPointGoal1-v0"
    epsilon: float = 0.1
    max_episode_steps: int = 1000


@dataclass
class RolloutBuffer:
    """Stores a batch of rollout data for PPO updates."""
    observations: np.ndarray = field(default_factory=lambda: np.empty(0))
    actions: np.ndarray = field(default_factory=lambda: np.empty(0))
    log_probs: np.ndarray = field(default_factory=lambda: np.empty(0))
    rewards: np.ndarray = field(default_factory=lambda: np.empty(0))
    values: np.ndarray = field(default_factory=lambda: np.empty(0))
    advantages: np.ndarray = field(default_factory=lambda: np.empty(0))
    returns: np.ndarray = field(default_factory=lambda: np.empty(0))


class RunningNormalizer:
    """Welford's online mean/variance for observation normalization."""

    def __init__(self, shape: Tuple[int, ...], clip: float = 10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            (x - self.mean) / np.sqrt(self.var + 1e-8),
            -self.clip, self.clip
        ).astype(np.float32)


def _make_env(cfg: EnvConfig):
    """Create a Safety Gym environment."""
    import safety_gymnasium
    env = safety_gymnasium.make(cfg.env_id, max_episode_steps=cfg.max_episode_steps)
    return env


def collect_episodes(
    env_cfg: EnvConfig,
    protagonist_fn: Callable,
    adversary_fn: Optional[Callable],
    num_steps: int = 2048,
    normalizer: Optional[RunningNormalizer] = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    value_fn: Optional[Callable] = None,
    adv_value_fn: Optional[Callable] = None,
) -> Tuple[RolloutBuffer, Optional[RolloutBuffer], Dict]:
    """Collect rollout data from the environment.

    Args:
        env_cfg: Environment configuration.
        protagonist_fn: fn(obs) -> (action, log_prob).
        adversary_fn: fn(obs) -> (action, log_prob), or None for clean evaluation.
        num_steps: Number of environment steps to collect.
        normalizer: Optional observation normalizer (updated in-place).
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        value_fn: fn(obs) -> value estimate for protagonist.
        adv_value_fn: fn(obs) -> value estimate for adversary.

    Returns:
        (protagonist_buffer, adversary_buffer, episode_stats)
    """
    env = _make_env(env_cfg)
    obs, info = env.reset()
    obs_dim = obs.shape[0]
    act_dim = env.action_space.shape[0]
    adv_act_dim = act_dim  # adversary has same action dim

    # Storage
    all_obs = []
    all_acts = []
    all_logps = []
    all_rewards = []
    all_values = []
    all_dones = []

    all_adv_obs = []
    all_adv_acts = []
    all_adv_logps = []
    all_adv_rewards = []
    all_adv_values = []

    ep_rewards = []
    ep_costs = []
    current_ep_reward = 0.0
    current_ep_cost = 0.0

    for step in range(num_steps):
        if normalizer is not None:
            norm_obs = normalizer.normalize(obs[np.newaxis])[0]
        else:
            norm_obs = obs.astype(np.float32)

        # Protagonist action
        with torch.no_grad():
            action, log_prob = protagonist_fn(norm_obs)
        if value_fn is not None:
            with torch.no_grad():
                value = value_fn(norm_obs)
        else:
            value = 0.0

        # Adversary perturbation
        if adversary_fn is not None:
            with torch.no_grad():
                adv_action, adv_log_prob = adversary_fn(norm_obs)
            if adv_value_fn is not None:
                with torch.no_grad():
                    adv_value = adv_value_fn(norm_obs)
            else:
                adv_value = 0.0
            # Perturbed action
            executed_action = np.clip(
                action + env_cfg.epsilon * adv_action, -1.0, 1.0
            )
        else:
            executed_action = action
            adv_action = np.zeros(adv_act_dim, dtype=np.float32)
            adv_log_prob = 0.0
            adv_value = 0.0

        # Step environment (safety-gymnasium returns 6 values)
        next_obs, reward, cost, terminated, truncated, info = env.step(executed_action)
        done = terminated or truncated

        all_obs.append(norm_obs)
        all_acts.append(action)
        all_logps.append(log_prob)
        all_rewards.append(reward)
        all_values.append(value)
        all_dones.append(done)

        if adversary_fn is not None:
            all_adv_obs.append(norm_obs)
            all_adv_acts.append(adv_action)
            all_adv_logps.append(adv_log_prob)
            all_adv_rewards.append(-reward)  # zero-sum
            all_adv_values.append(adv_value)

        current_ep_reward += reward
        current_ep_cost += cost

        if done:
            ep_rewards.append(current_ep_reward)
            ep_costs.append(current_ep_cost)
            current_ep_reward = 0.0
            current_ep_cost = 0.0
            obs, info = env.reset()
        else:
            obs = next_obs

    # Update normalizer with collected observations
    raw_obs_for_norm = np.array(all_obs)
    if normalizer is not None:
        normalizer.update(raw_obs_for_norm)

    # Compute GAE for protagonist
    obs_arr = np.array(all_obs, dtype=np.float32)
    acts_arr = np.array(all_acts, dtype=np.float32)
    logps_arr = np.array(all_logps, dtype=np.float32)
    rewards_arr = np.array(all_rewards, dtype=np.float32)
    values_arr = np.array(all_values, dtype=np.float32)
    dones_arr = np.array(all_dones, dtype=bool)

    # Bootstrap value for last state
    if not dones_arr[-1] and value_fn is not None:
        if normalizer is not None:
            last_norm = normalizer.normalize(obs[np.newaxis])[0]
        else:
            last_norm = obs.astype(np.float32)
        with torch.no_grad():
            last_value = value_fn(last_norm)
    else:
        last_value = 0.0

    advantages, returns = _compute_gae(
        rewards_arr, values_arr, dones_arr, last_value, gamma, gae_lambda
    )

    prot_buffer = RolloutBuffer(
        observations=obs_arr,
        actions=acts_arr,
        log_probs=logps_arr,
        rewards=rewards_arr,
        values=values_arr,
        advantages=advantages,
        returns=returns,
    )

    # Compute GAE for adversary
    adv_buffer = None
    if adversary_fn is not None and all_adv_obs:
        adv_obs_arr = np.array(all_adv_obs, dtype=np.float32)
        adv_acts_arr = np.array(all_adv_acts, dtype=np.float32)
        adv_logps_arr = np.array(all_adv_logps, dtype=np.float32)
        adv_rewards_arr = np.array(all_adv_rewards, dtype=np.float32)
        adv_values_arr = np.array(all_adv_values, dtype=np.float32)

        if not dones_arr[-1] and adv_value_fn is not None:
            with torch.no_grad():
                adv_last_value = adv_value_fn(last_norm)
        else:
            adv_last_value = 0.0

        adv_advantages, adv_returns = _compute_gae(
            adv_rewards_arr, adv_values_arr, dones_arr, adv_last_value,
            gamma, gae_lambda
        )

        adv_buffer = RolloutBuffer(
            observations=adv_obs_arr,
            actions=adv_acts_arr,
            log_probs=adv_logps_arr,
            rewards=adv_rewards_arr,
            values=adv_values_arr,
            advantages=adv_advantages,
            returns=adv_returns,
        )

    stats = {
        "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
        "mean_cost": float(np.mean(ep_costs)) if ep_costs else 0.0,
        "num_episodes": len(ep_rewards),
        "total_steps": num_steps,
    }

    env.close()
    return prot_buffer, adv_buffer, stats


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(n)):
        if t == n - 1:
            next_value = last_value
            next_non_terminal = 1.0 - float(dones[t])
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - float(dones[t])

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def evaluate(
    env_cfg: EnvConfig,
    protagonist_fn: Callable,
    adversary_fn: Optional[Callable],
    num_episodes: int = 10,
    normalizer: Optional[RunningNormalizer] = None,
) -> Dict:
    """Evaluate protagonist (with or without adversary) over episodes.

    Returns dict with mean_reward, mean_cost, std_reward.
    """
    env = _make_env(env_cfg)
    ep_rewards = []
    ep_costs = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_cost = 0.0
        done = False

        while not done:
            if normalizer is not None:
                norm_obs = normalizer.normalize(obs[np.newaxis])[0]
            else:
                norm_obs = obs.astype(np.float32)

            with torch.no_grad():
                action, _ = protagonist_fn(norm_obs)

            if adversary_fn is not None:
                with torch.no_grad():
                    adv_action, _ = adversary_fn(norm_obs)
                executed_action = np.clip(
                    action + env_cfg.epsilon * adv_action, -1.0, 1.0
                )
            else:
                executed_action = action

            obs, reward, cost, terminated, truncated, info = env.step(executed_action)
            ep_reward += reward
            ep_cost += cost
            done = terminated or truncated

        ep_rewards.append(ep_reward)
        ep_costs.append(ep_cost)

    env.close()
    return {
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "mean_cost": float(np.mean(ep_costs)),
        "std_cost": float(np.std(ep_costs)),
    }
