#!/usr/bin/env python3
"""
Zoo-based adversarial training for Safety Gym (PPO protagonist).

Samples adversary from zoo with probability A, trains protagonist every rollout,
trains latest adversary only when selected.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from env_wrapper import EnvConfig, RunningNormalizer, collect_episodes, evaluate
from ppo import PPOAgent, PPOConfig
from zoo import AdversaryZoo, a_schedule


def train_zoo(
    a_value: float = 0.1,
    timesteps: int = 300_000,
    steps_per_rollout: int = 2048,
    epsilon: float = 0.1,
    zoo_update_interval: int = 10,
    zoo_max_size: int = 50,
    log_interval: int = 5,
    eval_interval: int = 20,
    eval_episodes: int = 10,
    output_dir: str = "experiments/results/zoo",
    seed: int = 0,
    hidden: int = 64,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
    sampling_strategy: str = "uniform",
    competitiveness_threshold: float = 2.0,
    a_schedule_type: str = "constant",
    a_halflife: float = 0.25,
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
    latest_adversary = PPOAgent(ppo_cfg)
    normalizer = RunningNormalizer((obs_dim,))

    adversary_zoo = AdversaryZoo(
        ppo_cfg, max_size=zoo_max_size,
        sampling_strategy=sampling_strategy,
        competitiveness_threshold=competitiveness_threshold,
    )
    adversary_zoo.add(latest_adversary, update=0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "metrics.jsonl"

    total_steps = 0
    update_step = 0

    while total_steps < timesteps:
        # Current A value
        if a_schedule_type != "constant":
            current_a = a_schedule(total_steps, timesteps, a_schedule_type, a_halflife)
        else:
            current_a = a_value

        # Pick adversary
        zoo_idx = None
        if len(adversary_zoo) > 0 and np.random.random() < current_a:
            current_adversary, zoo_idx = adversary_zoo.sample()
        else:
            current_adversary = latest_adversary

        # Collect rollout
        prot_buf, adv_buf, stats = collect_episodes(
            env_cfg=env_cfg,
            protagonist_fn=protagonist.act,
            adversary_fn=current_adversary.act,
            num_steps=steps_per_rollout,
            normalizer=normalizer,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            value_fn=protagonist.get_value,
            adv_value_fn=current_adversary.get_value,
        )

        # Thompson update
        if zoo_idx is not None:
            adversary_zoo.update_outcome(zoo_idx, stats["mean_reward"])

        # Update protagonist always
        prot_metrics = protagonist.update(prot_buf)

        # Update latest adversary only when selected
        adv_metrics = {"pg_loss": 0.0, "entropy": 0.0}
        if current_adversary is latest_adversary and adv_buf is not None:
            adv_metrics = latest_adversary.update(adv_buf)

        total_steps += steps_per_rollout
        update_step += 1

        # Add to zoo periodically
        if update_step % zoo_update_interval == 0:
            adversary_zoo.add(latest_adversary, update=update_step)

        if update_step % log_interval == 0:
            metrics = {
                "update": update_step,
                "timesteps": total_steps,
                "algorithm": "ppo",
                "a_value": current_a,
                "a_schedule_type": a_schedule_type,
                "sampling_strategy": sampling_strategy,
                "zoo_size": len(adversary_zoo),
                "adv_reward": stats["mean_reward"],
                "adv_cost": stats["mean_cost"],
                "num_episodes": stats["num_episodes"],
                "prot_pg_loss": prot_metrics["pg_loss"],
                "prot_entropy": prot_metrics["entropy"],
            }

            if sampling_strategy in ("thompson", "thompson_loss"):
                metrics.update(adversary_zoo.ts_diagnostics())

            # Periodic evaluation
            if update_step % eval_interval == 0:
                clean_eval = evaluate(env_cfg, protagonist.act, None,
                                      eval_episodes, normalizer)
                adv_eval = evaluate(env_cfg, protagonist.act, latest_adversary.act,
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
                f"[{total_steps:>8d}] A={current_a:.3f} zoo={len(adversary_zoo):>3d}"
                f" adv_rew={stats['mean_reward']:.1f}"
                f" ent={prot_metrics['entropy']:.3f}{clean_str}"
            )

    print(f"\nDone. Metrics saved to {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser(description="Safety Gym zoo training (PPO)")
    parser.add_argument("--a-value", "-A", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--steps-per-rollout", type=int, default=2048)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--zoo-update-interval", type=int, default=10)
    parser.add_argument("--zoo-max-size", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="experiments/results/zoo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--sampling-strategy", type=str, default="uniform",
                        choices=["uniform", "thompson", "thompson_loss"])
    parser.add_argument("--competitiveness-threshold", type=float, default=2.0)
    parser.add_argument("--a-schedule", type=str, default="constant",
                        choices=["constant", "exponential", "linear", "sigmoid",
                                 "exponential_down", "linear_down", "sigmoid_down"])
    parser.add_argument("--a-halflife", type=float, default=0.25)
    args = parser.parse_args()

    train_zoo(
        a_value=args.a_value,
        timesteps=args.timesteps,
        steps_per_rollout=args.steps_per_rollout,
        epsilon=args.epsilon,
        zoo_update_interval=args.zoo_update_interval,
        zoo_max_size=args.zoo_max_size,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        seed=args.seed,
        hidden=args.hidden,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        sampling_strategy=args.sampling_strategy,
        competitiveness_threshold=args.competitiveness_threshold,
        a_schedule_type=args.a_schedule,
        a_halflife=args.a_halflife,
    )


if __name__ == "__main__":
    main()
