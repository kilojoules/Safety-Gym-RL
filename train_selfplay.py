#!/usr/bin/env python3
"""
RARL baseline: co-train protagonist + adversary (A=0, no zoo).

The adversary perturbs protagonist actions by δ scaled by epsilon.
Both agents update every rollout.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from env_wrapper import EnvConfig, RunningNormalizer, collect_episodes, evaluate
from ppo import PPOAgent, PPOConfig


def train_selfplay(
    timesteps: int = 300_000,
    steps_per_rollout: int = 2048,
    epsilon: float = 0.1,
    log_interval: int = 5,
    eval_interval: int = 20,
    eval_episodes: int = 10,
    output_dir: str = "experiments/results/selfplay",
    seed: int = 0,
    hidden: int = 64,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_cfg = EnvConfig(epsilon=epsilon)

    # Probe env for dimensions
    import safety_gymnasium
    probe_env = safety_gymnasium.make(env_cfg.env_id, max_episode_steps=env_cfg.max_episode_steps)
    obs_dim = probe_env.observation_space.shape[0]
    act_dim = probe_env.action_space.shape[0]
    probe_env.close()

    ppo_cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden,
                        lr_actor=lr_actor, lr_critic=lr_critic)

    protagonist = PPOAgent(ppo_cfg)
    adversary = PPOAgent(ppo_cfg)
    normalizer = RunningNormalizer((obs_dim,))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "metrics.jsonl"

    total_steps = 0
    update_step = 0

    while total_steps < timesteps:
        # Collect rollout
        prot_buf, adv_buf, stats = collect_episodes(
            env_cfg=env_cfg,
            protagonist_fn=protagonist.act,
            adversary_fn=adversary.act,
            num_steps=steps_per_rollout,
            normalizer=normalizer,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            value_fn=protagonist.get_value,
            adv_value_fn=adversary.get_value,
        )

        # Update both agents
        prot_metrics = protagonist.update(prot_buf)
        adv_metrics = adversary.update(adv_buf)

        total_steps += steps_per_rollout
        update_step += 1

        if update_step % log_interval == 0:
            metrics = {
                "update": update_step,
                "timesteps": total_steps,
                "algorithm": "selfplay",
                "adv_reward": stats["mean_reward"],
                "adv_cost": stats["mean_cost"],
                "num_episodes": stats["num_episodes"],
                "prot_pg_loss": prot_metrics["pg_loss"],
                "prot_entropy": prot_metrics["entropy"],
                "adv_pg_loss": adv_metrics["pg_loss"],
            }

            # Periodic evaluation
            if update_step % eval_interval == 0:
                clean_eval = evaluate(env_cfg, protagonist.act, None,
                                      eval_episodes, normalizer)
                adv_eval = evaluate(env_cfg, protagonist.act, adversary.act,
                                    eval_episodes, normalizer)
                metrics["clean_reward"] = clean_eval["mean_reward"]
                metrics["clean_cost"] = clean_eval["mean_cost"]
                metrics["eval_adv_reward"] = adv_eval["mean_reward"]
                metrics["eval_adv_cost"] = adv_eval["mean_cost"]
                if clean_eval["mean_reward"] != 0:
                    metrics["reward_drop"] = (
                        (clean_eval["mean_reward"] - adv_eval["mean_reward"])
                        / abs(clean_eval["mean_reward"])
                    )
                else:
                    metrics["reward_drop"] = 0.0

            with open(log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

            clean_str = ""
            if "clean_reward" in metrics:
                clean_str = (f" clean={metrics['clean_reward']:.1f}"
                             f" drop={metrics.get('reward_drop', 0):.3f}")
            print(
                f"[{total_steps:>8d}] adv_rew={stats['mean_reward']:.1f}"
                f" cost={stats['mean_cost']:.1f}"
                f" ent={prot_metrics['entropy']:.3f}{clean_str}"
            )

    print(f"\nDone. Metrics saved to {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser(description="Safety Gym RARL self-play (A=0)")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--steps-per-rollout", type=int, default=2048)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="experiments/results/selfplay")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    args = parser.parse_args()

    train_selfplay(**vars(args))


if __name__ == "__main__":
    main()
