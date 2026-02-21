#!/usr/bin/env python3
"""Analyze sweep results and generate plots for Safety Gym experiments."""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)


def load_metrics(results_dir: Path) -> dict:
    """Load all experiment metrics from directory tree."""
    data = {}
    for metrics_file in sorted(results_dir.rglob("metrics.jsonl")):
        rel = metrics_file.relative_to(results_dir)
        parts = list(rel.parts)

        if len(parts) >= 2:
            exp_name = parts[0]
            seed_dir = parts[1]
        else:
            exp_name = "unknown"
            seed_dir = parts[0] if parts else "seed0"

        seed_match = re.search(r"seed(\d+)", seed_dir)
        seed = int(seed_match.group(1)) if seed_match else 0

        lines = []
        with open(metrics_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))

        if not lines:
            continue

        if exp_name not in data:
            data[exp_name] = {}
        data[exp_name][seed] = lines

    return data


def get_final_metric(runs: dict, metric: str) -> list:
    """Get final value of a metric from each seed's run."""
    values = []
    for seed, lines in runs.items():
        # Walk backwards to find last entry with this metric
        for entry in reversed(lines):
            if metric in entry:
                values.append(entry[metric])
                break
    return values


def parse_experiment_name(name: str):
    """Parse algo, strategy, A value from experiment directory name.

    Format: {algo}_{strategy}_A{value}
    e.g. ppo_uniform_A0.50, buffered_thompson_loss_A0.90
    """
    m = re.match(r"^(ppo|buffered)_(.+)_A([\d.]+)$", name)
    if m:
        return m.group(1), m.group(2), float(m.group(3))
    return None, None, None


def plot_a_curve(data: dict, output_dir: Path):
    """Plot reward_drop vs A for PPO and Buffered (uniform)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, color, label in [("ppo", "tab:blue", "PPO (memoryless)"),
                                ("buffered", "tab:red", "Buffered (replay buffer)")]:
        a_vals = []
        means = []
        stds = []
        for name, runs in sorted(data.items()):
            a, strat, a_val = parse_experiment_name(name)
            if a == algo and strat == "uniform" and a_val is not None:
                vals = get_final_metric(runs, "reward_drop")
                if vals:
                    a_vals.append(a_val)
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))

        if a_vals:
            ax.errorbar(a_vals, means, yerr=stds, marker="o", label=label,
                       color=color, capsize=3)

    # Add self-play baseline
    if "selfplay" in data:
        sp_vals = get_final_metric(data["selfplay"], "reward_drop")
        if sp_vals:
            ax.axhline(np.mean(sp_vals), color="gray", linestyle="--",
                       label=f"Self-play (A=0): {np.mean(sp_vals):.3f}")

    ax.set_xlabel("A (zoo sampling probability)")
    ax.set_ylabel("Reward Drop (robustness metric)")
    ax.set_title("Safety Gym: Reward Drop vs Zoo Sampling (A)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "a_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'a_curve.png'}")


def plot_thompson_comparison(data: dict, output_dir: Path):
    """Plot uniform vs Thompson vs Thompson-Loss at each A value."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    strat_configs = [
        ("uniform", "tab:blue", "o", "Uniform"),
        ("thompson", "tab:orange", "s", "Thompson"),
        ("thompson_loss", "tab:green", "^", "Thompson (loss-seeking)"),
    ]

    for ax, algo, title in [(axes[0], "ppo", "PPO"),
                             (axes[1], "buffered", "Buffered")]:
        for strat, color, marker, label in strat_configs:
            a_vals, means, stds = [], [], []
            for name, runs in sorted(data.items()):
                a, s, a_val = parse_experiment_name(name)
                if a == algo and s == strat and a_val is not None:
                    vals = get_final_metric(runs, "reward_drop")
                    if vals:
                        a_vals.append(a_val)
                        means.append(np.mean(vals))
                        stds.append(np.std(vals))
            if a_vals:
                ax.errorbar(a_vals, means, yerr=stds, marker=marker,
                           label=label, color=color, capsize=3)

        ax.set_xlabel("A")
        ax.set_ylabel("Reward Drop")
        ax.set_title(f"{title}: Sampling Strategy Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "thompson_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'thompson_comparison.png'}")


def plot_timeseries(data: dict, output_dir: Path):
    """Plot reward_drop over training for select conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    targets = [
        ("selfplay", "Self-play (A=0)", "gray"),
        ("ppo_uniform_A0.05", "PPO A=0.05", "tab:blue"),
        ("ppo_uniform_A0.50", "PPO A=0.50", "tab:green"),
        ("ppo_uniform_A0.90", "PPO A=0.90", "tab:red"),
        ("ppo_thompson_A0.50", "PPO Thompson A=0.50", "tab:orange"),
        ("ppo_thompson_loss_A0.50", "PPO Thompson-Loss A=0.50", "tab:purple"),
    ]

    for name, label, color in targets:
        if name not in data:
            continue
        all_ts = {}
        for seed, lines in data[name].items():
            for entry in lines:
                if "reward_drop" not in entry:
                    continue
                t = entry["timesteps"]
                if t not in all_ts:
                    all_ts[t] = []
                all_ts[t].append(entry["reward_drop"])

        if all_ts:
            ts = sorted(all_ts.keys())
            means = [np.mean(all_ts[t]) for t in ts]
            ax.plot(ts, means, label=label, color=color, alpha=0.8)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Reward Drop")
    ax.set_title("Safety Gym: Reward Drop Over Training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "timeseries.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'timeseries.png'}")


def plot_cost_comparison(data: dict, output_dir: Path):
    """Plot safety violations (cost) under attack vs A value."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, ylabel, title in [
        (axes[0], "eval_adv_cost", "Adversarial Cost", "Safety Violations Under Attack"),
        (axes[1], "clean_cost", "Clean Cost", "Safety Violations (No Attack)"),
    ]:
        for algo, color, label in [("ppo", "tab:blue", "PPO"),
                                    ("buffered", "tab:red", "Buffered")]:
            a_vals, means, stds = [], [], []
            for name, runs in sorted(data.items()):
                a, strat, a_val = parse_experiment_name(name)
                if a == algo and strat == "uniform" and a_val is not None:
                    vals = get_final_metric(runs, metric)
                    if vals:
                        a_vals.append(a_val)
                        means.append(np.mean(vals))
                        stds.append(np.std(vals))
            if a_vals:
                ax.errorbar(a_vals, means, yerr=stds, marker="o",
                           label=label, color=color, capsize=3)

        if "selfplay" in data:
            sp_vals = get_final_metric(data["selfplay"], metric)
            if sp_vals:
                ax.axhline(np.mean(sp_vals), color="gray", linestyle="--",
                           label=f"Self-play: {np.mean(sp_vals):.1f}")

        ax.set_xlabel("A")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "cost_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'cost_comparison.png'}")


def print_summary_tables(data: dict):
    """Print summary tables to stdout."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLES")
    print("=" * 80)

    # Self-play baseline
    if "selfplay" in data:
        rd_vals = get_final_metric(data["selfplay"], "reward_drop")
        cr_vals = get_final_metric(data["selfplay"], "clean_reward")
        ac_vals = get_final_metric(data["selfplay"], "eval_adv_cost")
        if rd_vals:
            print(f"\nSelf-play (A=0):")
            print(f"  Reward drop: {np.mean(rd_vals):.4f} +/- {np.std(rd_vals):.4f}")
            if cr_vals:
                print(f"  Clean reward: {np.mean(cr_vals):.1f} +/- {np.std(cr_vals):.1f}")
            if ac_vals:
                print(f"  Adv cost: {np.mean(ac_vals):.1f} +/- {np.std(ac_vals):.1f}")
            print(f"  ({len(rd_vals)} seeds)")

    # PPO uniform table
    print("\nPPO (uniform sampling):")
    print(f"{'A':>6s} | {'Reward Drop':>20s} | {'Clean Reward':>16s} | {'Adv Cost':>12s} | {'Seeds':>5s}")
    print("-" * 70)
    for name in sorted(data.keys()):
        a, strat, a_val = parse_experiment_name(name)
        if a == "ppo" and strat == "uniform" and a_val is not None:
            rd = get_final_metric(data[name], "reward_drop")
            cr = get_final_metric(data[name], "clean_reward")
            ac = get_final_metric(data[name], "eval_adv_cost")
            if rd:
                cr_str = f"{np.mean(cr):.1f}" if cr else "N/A"
                ac_str = f"{np.mean(ac):.1f}" if ac else "N/A"
                print(f"{a_val:>6.2f} | {np.mean(rd):>8.4f} +/- {np.std(rd):<8.4f} | "
                      f"{cr_str:>16s} | {ac_str:>12s} | {len(rd):>5d}")

    # Buffered uniform table
    print("\nBuffered (uniform sampling):")
    print(f"{'A':>6s} | {'Reward Drop':>20s} | {'Clean Reward':>16s} | {'Adv Cost':>12s} | {'Seeds':>5s}")
    print("-" * 70)
    for name in sorted(data.keys()):
        a, strat, a_val = parse_experiment_name(name)
        if a == "buffered" and strat == "uniform" and a_val is not None:
            rd = get_final_metric(data[name], "reward_drop")
            cr = get_final_metric(data[name], "clean_reward")
            ac = get_final_metric(data[name], "eval_adv_cost")
            if rd:
                cr_str = f"{np.mean(cr):.1f}" if cr else "N/A"
                ac_str = f"{np.mean(ac):.1f}" if ac else "N/A"
                print(f"{a_val:>6.2f} | {np.mean(rd):>8.4f} +/- {np.std(rd):<8.4f} | "
                      f"{cr_str:>16s} | {ac_str:>12s} | {len(rd):>5d}")

    # 3-way Thompson comparison (PPO)
    print("\nSampling Strategy Comparison (PPO) - Reward Drop:")
    print(f"{'A':>6s} | {'Uniform':>20s} | {'Thompson':>20s} | {'Thompson-Loss':>20s}")
    print("-" * 80)
    for a_val in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        parts = []
        for strat in ["uniform", "thompson", "thompson_loss"]:
            name = f"ppo_{strat}_A{a_val:.2f}"
            if name in data:
                vals = get_final_metric(data[name], "reward_drop")
                if vals:
                    parts.append(f"{np.mean(vals):>8.4f} +/- {np.std(vals):<8.4f}")
                else:
                    parts.append("       N/A          ")
            else:
                parts.append("       N/A          ")
        print(f"{a_val:>6.2f} | {parts[0]} | {parts[1]} | {parts[2]}")

    # 3-way Thompson comparison (Buffered)
    print("\nSampling Strategy Comparison (Buffered) - Reward Drop:")
    print(f"{'A':>6s} | {'Uniform':>20s} | {'Thompson':>20s} | {'Thompson-Loss':>20s}")
    print("-" * 80)
    for a_val in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        parts = []
        for strat in ["uniform", "thompson", "thompson_loss"]:
            name = f"buffered_{strat}_A{a_val:.2f}"
            if name in data:
                vals = get_final_metric(data[name], "reward_drop")
                if vals:
                    parts.append(f"{np.mean(vals):>8.4f} +/- {np.std(vals):<8.4f}")
                else:
                    parts.append("       N/A          ")
            else:
                parts.append("       N/A          ")
        print(f"{a_val:>6.2f} | {parts[0]} | {parts[1]} | {parts[2]}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Safety Gym experiments")
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Loading results from {results_dir}...")
    data = load_metrics(results_dir)
    print(f"Found {len(data)} experiment conditions")

    if not data:
        print("No data found!")
        sys.exit(1)

    print_summary_tables(data)
    plot_a_curve(data, results_dir)
    plot_thompson_comparison(data, results_dir)
    plot_timeseries(data, results_dir)
    plot_cost_comparison(data, results_dir)


if __name__ == "__main__":
    main()
