# Safety-Gym-RL: Does Adversary Diversity Help or Hurt Robust RL?

## Motivation

A standard way to train robust RL agents is **adversarial training**: pair a protagonist with an adversary that tries to make it fail, and co-train both ([RARL, Pinto et al. 2017](https://arxiv.org/abs/1703.02702)). But a single co-evolving adversary can overspecialize — both players adapt to each other's quirks rather than learning general strategies.

A natural fix is to maintain a **zoo** of historical adversary checkpoints and occasionally train the protagonist against a randomly sampled old adversary instead of the current one. The intuition is that diverse opponents should produce a more robust protagonist — the same logic behind population-based training and league play.

**This project tests whether that intuition holds in continuous control with safety constraints.** The answer is no: zoo-based adversary diversity fails to improve robustness and dramatically increases safety violations.

## Problem Setup

**Environment**: `SafetyPointGoal1-v0` from [Safety Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium). A Point robot (2D continuous action, 60D observation) navigates to goal locations while avoiding hazard regions. The environment returns both a task reward and a safety cost (number of hazard violations).

**Adversary model**: The adversary observes the same state as the protagonist and outputs a perturbation δ ∈ [-1, 1]². The action actually executed in the environment is:

```
executed_action = clip(protagonist_action + ε · δ,  -1, 1)
```

with ε = 0.1 (the adversary can nudge actions by up to 10%). The adversary's reward is the negative of the protagonist's reward (zero-sum game). This setup follows the Robust Adversarial RL (RARL) framework.

**The zoo mechanism**: We maintain a collection of past adversary checkpoints. A parameter **A ∈ [0, 1]** controls the probability that, on each training rollout, the protagonist faces a randomly sampled historical adversary instead of the current (latest) adversary. When A = 0, this reduces to standard RARL self-play; when A = 1, the protagonist always trains against a zoo adversary.

**Zoo sampling strategies**: When sampling from the zoo, we compare three methods:
- **Uniform**: each historical adversary is equally likely
- **Thompson**: Beta-Bernoulli Thompson sampling that prefers adversaries producing competitive (close) matches
- **Thompson (loss-seeking)**: Thompson sampling that prefers adversaries that *beat* the protagonist, prioritizing corrective training signal

**Protagonist algorithms**: We compare two variants:
- **PPO**: standard Proximal Policy Optimization with Gaussian policy (memoryless — each update uses only the latest rollout)
- **Buffered**: PPO augmented with a trajectory replay buffer and importance-weighted corrections, allowing learning from older experience

**Robustness metric**: We periodically evaluate the protagonist with and without the adversary, and compute:

```
reward_drop = (clean_reward - adversarial_reward) / |clean_reward|
```

A reward drop near 0 means the protagonist is robust to adversarial perturbation. Higher values mean the adversary successfully degrades performance.

## Results

### Zoo diversity does not improve robustness

![Reward Drop vs A](experiments/results/a_curve.png)

Self-play (A = 0, dashed gray) achieves a reward drop of **-0.15** — the adversary actually *helps* slightly, likely by adding beneficial exploration noise. Across all tested values of A, zoo-trained agents show **high variance and no consistent improvement**. The PPO agent at A = 0.2 exhibits a pathological -30x reward drop in one seed. High values of A (0.7–0.9) produce erratic results for both algorithms.

### Thompson sampling reduces variance but doesn't solve the problem

![Thompson Comparison](experiments/results/thompson_comparison.png)

Smarter zoo sampling (Thompson and loss-seeking Thompson) produces tighter error bars than uniform sampling — the training is more stable. But neither variant consistently outperforms simple self-play. Loss-seeking Thompson for the Buffered agent at A = 0.5 is the only condition that marginally beats the baseline.

### Safety violations spike dramatically under zoo training

![Cost Comparison](experiments/results/cost_comparison.png)

This is the most striking finding. Self-play agents accumulate ~4 safety violations per episode. Zoo-trained agents accumulate **50–120 violations** — a 10–30x increase — both under adversarial attack and in clean (no-adversary) evaluation. Zoo diversity doesn't just fail to improve robustness; it actively produces agents that ignore safety constraints.

The likely mechanism: when facing diverse historical adversaries of varying strength, the protagonist learns aggressive, reward-seeking policies that exploit weaker opponents. These strategies incidentally violate hazard boundaries because the training signal from diverse weak adversaries doesn't penalize boundary-crossing as consistently as co-adapted self-play does.

### Training dynamics show progressive degradation

![Training Timeseries](experiments/results/timeseries.png)

High-A training (A = 0.9, red) shows reward drop *increasing* over time — the agent becomes progressively less robust. Low-A (A = 0.05, blue) tracks close to self-play. This suggests zoo diversity actively interferes with the protagonist-adversary co-adaptation that makes RARL work.

## Summary Table

| Condition | Reward Drop | Clean Reward | Adv Cost |
|-----------|------------|-------------|----------|
| Self-play (A=0) | -0.15 ± 0.05 | -14.3 ± 10.3 | 4.2 ± 0.2 |
| PPO uniform A=0.05 | -0.17 ± 0.07 | 2.5 | 35.7 |
| PPO uniform A=0.50 | -0.48 ± 0.32 | 5.4 | 68.2 |
| PPO uniform A=0.90 | 1.34 ± 1.01 | -7.7 | 15.4 |
| Buffered uniform A=0.05 | 0.11 ± 0.24 | -5.6 | 71.7 |
| Buffered uniform A=0.50 | 10.97 ± 9.97 | -0.9 | 25.9 |
| Buffered uniform A=0.90 | -0.56 ± 1.04 | -3.8 | 39.5 |

## Why Does This Happen?

RARL works because the protagonist and adversary co-adapt: the adversary finds the protagonist's current weaknesses, and the protagonist patches them. This tight feedback loop is disrupted when the protagonist frequently faces stale adversaries from the zoo. The old adversaries exploit weaknesses the protagonist has already fixed, providing a misleading training signal and preventing the adversary from discovering *current* vulnerabilities.

In games with well-defined Nash equilibria (like Rock-Paper-Scissors), zoo diversity can still help because the equilibrium strategy is also the best response to a uniform mixture of opponents. But in continuous control — where there is no such equilibrium and robustness depends on ongoing co-adaptation — diversity is counterproductive.

This connects to a broader pattern we observe across domains:

| Domain | Has Nash = Best Response to Mixture? | Zoo Helps? |
|--------|-------------------------------------|------------|
| [Rock-Paper-Scissors](https://github.com/kilojoules/RPS_RL) | Yes | Yes |
| [Kuhn Poker](https://github.com/kilojoules/Kuhn-Poker-RL) | No | No |
| Safety Gym (this project) | No fixed equilibrium | No |

## Architecture

| File | Description |
|------|-------------|
| `env_wrapper.py` | SafetyPointGoal1 wrapper, adversarial perturbation, GAE rollout collection |
| `ppo.py` | Gaussian PPO (2-layer MLP, learnable log_std) |
| `buffered_agent.py` | PPO + trajectory replay buffer with importance sampling correction |
| `zoo.py` | Adversary zoo with uniform/Thompson/Thompson-loss sampling |
| `train_selfplay.py` | RARL baseline (A=0) |
| `train_zoo.py` | Zoo training (PPO protagonist) |
| `train_zoo_buffered.py` | Zoo training (Buffered protagonist) |
| `run_sweep.py` | Parallel experiment orchestration |
| `analyze.py` | Plots and summary tables |

## Usage

```bash
# Install dependencies
pixi install

# Run self-play baseline
pixi run python train_selfplay.py --timesteps 100000 --seed 0

# Run a zoo experiment (A=0.3, loss-seeking Thompson sampling)
pixi run python train_zoo.py -A 0.3 --timesteps 100000 --sampling-strategy thompson_loss

# Run full sweep (62 experiments, ~2.5 hours with 8 parallel)
pixi run python run_sweep.py --timesteps 100000 --seeds 2 --parallel 8 \
    --a-values "0.05,0.2,0.5,0.7,0.9"

# Generate plots from results
pixi run python analyze.py experiments/results/
```

## Sweep Configuration

| Parameter | Value |
|-----------|-------|
| A values | 0.05, 0.2, 0.5, 0.7, 0.9 |
| Algorithms | PPO, Buffered |
| Zoo sampling | uniform, thompson, thompson_loss |
| Seeds | 2 |
| Timesteps | 100k (~49 PPO updates at 2048 steps/rollout) |
| Adversary ε | 0.1 |
| Parallel workers | 8 |
