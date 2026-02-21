#!/usr/bin/env python3
"""Sweep orchestration for Safety Gym A-parameter experiments."""
import argparse
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


def run_experiment(args_dict: dict) -> subprocess.Popen:
    """Launch a single experiment as a subprocess."""
    script = args_dict.pop("script")
    cmd = [sys.executable, script]
    for k, v in args_dict.items():
        cmd.extend([f"--{k}", str(v)])
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser(description="Safety Gym experiment sweep")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--a-values", type=str, default="0.05,0.1,0.2,0.3,0.5,0.7,0.9")
    parser.add_argument("--sampling-strategy", type=str, default="all",
                        choices=["uniform", "thompson", "thompson_loss", "both", "all"])
    parser.add_argument("--algorithms", type=str, default="both",
                        choices=["ppo", "buffered", "both"])
    parser.add_argument("--include-selfplay", action="store_true", default=True)
    parser.add_argument("--output-base", type=str, default="experiments/results")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--steps-per-rollout", type=int, default=2048)
    args = parser.parse_args()

    a_values = [float(x) for x in args.a_values.split(",")]
    seeds = list(range(args.seeds))

    if args.sampling_strategy == "all":
        strategies = ["uniform", "thompson", "thompson_loss"]
    elif args.sampling_strategy == "both":
        strategies = ["uniform", "thompson"]
    else:
        strategies = [args.sampling_strategy]

    if args.algorithms == "both":
        algorithms = ["ppo", "buffered"]
    else:
        algorithms = [args.algorithms]

    experiments = []

    # Self-play baselines
    if args.include_selfplay:
        for seed in seeds:
            experiments.append({
                "script": "train_selfplay.py",
                "timesteps": args.timesteps,
                "steps-per-rollout": args.steps_per_rollout,
                "epsilon": args.epsilon,
                "seed": seed,
                "output-dir": f"{args.output_base}/selfplay/seed{seed}",
                "log-interval": 5,
                "eval-interval": 20,
            })

    # Zoo experiments
    for algo, strategy, a_val, seed in product(algorithms, strategies, a_values, seeds):
        if algo == "ppo":
            script = "train_zoo.py"
        else:
            script = "train_zoo_buffered.py"

        exp_name = f"{algo}_{strategy}_A{a_val:.2f}"
        experiments.append({
            "script": script,
            "a-value": a_val,
            "timesteps": args.timesteps,
            "steps-per-rollout": args.steps_per_rollout,
            "epsilon": args.epsilon,
            "seed": seed,
            "sampling-strategy": strategy,
            "output-dir": f"{args.output_base}/{exp_name}/seed{seed}",
            "log-interval": 5,
            "eval-interval": 20,
        })

    total = len(experiments)
    print(f"Running {total} experiments ({args.parallel} parallel)...")
    print(f"  A values: {a_values}")
    print(f"  Algorithms: {algorithms}")
    print(f"  Strategies: {strategies}")
    print(f"  Seeds: {len(seeds)}")

    running = []
    completed = 0
    failed = 0
    start_time = time.time()

    for i, exp in enumerate(experiments):
        # Wait for a slot
        while len(running) >= args.parallel:
            for proc, name in list(running):
                if proc.poll() is not None:
                    if proc.returncode != 0:
                        stderr = proc.stderr.read().decode()
                        print(f"  FAILED: {name}")
                        if stderr:
                            print(f"    {stderr[:300]}")
                        failed += 1
                    else:
                        completed += 1
                    running.remove((proc, name))
            if len(running) >= args.parallel:
                time.sleep(1.0)

        name = exp.get("output-dir", "unknown")
        proc = run_experiment(exp.copy())
        running.append((proc, name))

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            print(f"  Launched {i+1}/{total}, completed {completed}, "
                  f"failed {failed}, elapsed {elapsed:.0f}s")

    # Wait for remaining
    for proc, name in running:
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode()
            print(f"  FAILED: {name}")
            if stderr:
                print(f"    {stderr[:300]}")
            failed += 1
        else:
            completed += 1

    elapsed = time.time() - start_time
    print(f"\nDone! {completed} succeeded, {failed} failed, {elapsed:.1f}s total")


if __name__ == "__main__":
    main()
