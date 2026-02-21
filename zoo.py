"""
Adversary zoo: stores historical adversary checkpoints and samples from them.

Supports uniform random sampling or Thompson Sampling (Beta-Bernoulli).
Adapted from Kuhn-Poker-RL/zoo.py for continuous-action PyTorch policies.
"""
import math
import random
from typing import Any, Dict, List, Tuple

import numpy as np

from ppo import PPOAgent, PPOConfig


class AdversaryZoo:
    """Manages a zoo of adversary policy checkpoints."""

    def __init__(self, cfg: PPOConfig, max_size: int = 50,
                 sampling_strategy: str = "uniform",
                 competitiveness_threshold: float = 2.0):
        self.cfg = cfg
        self.max_size = max_size
        self.sampling_strategy = sampling_strategy
        self.competitiveness_threshold = competitiveness_threshold
        self.checkpoints: List[Dict[str, Any]] = []
        self.alphas: List[float] = []
        self.betas: List[float] = []

    def add(self, agent: PPOAgent, update: int):
        """Snapshot current agent params into the zoo."""
        self.checkpoints.append({
            "params": agent.get_state(),
            "update": update,
        })
        self.alphas.append(1.0)
        self.betas.append(1.0)
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)
            self.alphas.pop(0)
            self.betas.pop(0)

    def sample(self) -> Tuple[PPOAgent, int]:
        """Return (agent, index) from the zoo."""
        if not self.checkpoints:
            raise ValueError("Zoo is empty")

        if self.sampling_strategy in ("thompson", "thompson_loss") and len(self.checkpoints) > 1:
            thetas = [np.random.beta(a, b) for a, b in zip(self.alphas, self.betas)]
            idx = int(np.argmax(thetas))
        else:
            idx = random.randrange(len(self.checkpoints))

        ckpt = self.checkpoints[idx]
        agent = PPOAgent(self.cfg)
        agent.load_state(ckpt["params"])
        return agent, idx

    def update_outcome(self, idx: int, protagonist_reward: float):
        """Update Beta posterior based on protagonist's reward against this adversary.

        - "thompson": success = competitive match (|reward| < threshold).
          Prefers adversaries that produce close games.
        - "thompson_loss": success = adversary won (protagonist reward low).
          Prefers adversaries that beat the protagonist, providing corrective signal.
        """
        if idx < 0 or idx >= len(self.checkpoints):
            return
        if self.sampling_strategy == "thompson_loss":
            # Loss-seeking: success when protagonist does POORLY against this adversary
            if protagonist_reward < -self.competitiveness_threshold:
                self.alphas[idx] += 1.0
            else:
                self.betas[idx] += 1.0
        else:
            # Standard Thompson: success when match is competitive
            if abs(protagonist_reward) < self.competitiveness_threshold:
                self.alphas[idx] += 1.0
            else:
                self.betas[idx] += 1.0

    def ts_diagnostics(self) -> Dict[str, float]:
        if not self.alphas:
            return {}
        return {
            "ts_alpha_mean": float(np.mean(self.alphas)),
            "ts_beta_mean": float(np.mean(self.betas)),
            "ts_success_rate": float(
                np.mean([a / (a + b) for a, b in zip(self.alphas, self.betas)])
            ),
        }

    def __len__(self):
        return len(self.checkpoints)


def a_schedule(t: int, timesteps: int, schedule: str = "exponential",
               halflife: float = 0.25) -> float:
    """Map training progress to A value in [0, 1]."""
    if schedule.endswith("_down"):
        base = schedule[:-5]
        return 1.0 - a_schedule(t, timesteps, base, halflife)

    frac = t / timesteps
    h = halflife
    if schedule == "exponential":
        return 1.0 - math.exp(-math.log(2) * frac / h)
    elif schedule == "linear":
        return min(frac / (2 * h), 1.0)
    elif schedule == "sigmoid":
        k = math.log(99) / h
        return 1.0 / (1.0 + math.exp(-k * (frac - h)))
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
