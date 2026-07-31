from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from nsga3drl.data.generator import generate_instance
from nsga3drl.data.io import load_dataset
from nsga3drl.rl.environment import OperatorSelectionEnv
from nsga3drl.rl.ppo import PPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO to choose the NSGA-III mutation operator.")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--vehicles", type=int, default=3)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", help="Directory of generated instance JSON files. Overrides --tasks and --vehicles.")
    parser.add_argument("--checkpoint", default="checkpoints/ppo_operator_selector.pt")
    parser.add_argument("--resume", help="Resume PPO model and optimizer state from this checkpoint.")
    parser.add_argument("--metrics", help="JSON report path; defaults to <checkpoint>.train.json")
    args = parser.parse_args()
    trainer = PPOTrainer.load(args.resume) if args.resume else PPOTrainer(
        OperatorSelectionEnv.observation_size, OperatorSelectionEnv.action_count, seed=args.seed
    )
    dataset = load_dataset(args.dataset) if args.dataset else None
    if dataset and any(instance.events for instance in dataset):
        print("note: dynamic events will be inserted at their configured arrival_generation")
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    update_records = []
    for update in range(args.updates):
        update_started = time.perf_counter()
        # Sampling dataset instances per update prevents PPO from memorising one layout.
        selected = (
            [dataset[int(index)] for index in rng.integers(0, len(dataset), size=args.episodes_per_update)]
            if dataset
            else [
                generate_instance(args.tasks, args.vehicles, args.seed + update * args.episodes_per_update + index)
                for index in range(args.episodes_per_update)
            ]
        )
        environments = [
            OperatorSelectionEnv(
                instance,
                population_size=args.population,
                generations=args.generations,
                seed=args.seed + update * args.episodes_per_update + index,
            )
            for index, instance in enumerate(selected)
        ]
        rollout = trainer.collect(environments)
        metrics = trainer.update(rollout)
        # Save after every update, so an interrupted run still has a reusable model.
        trainer.save(args.checkpoint)
        update_elapsed = time.perf_counter() - update_started
        elapsed = time.perf_counter() - started
        projected_total = elapsed / (update + 1) * args.updates
        metrics.update({
            "update": update + 1,
            "elapsed_seconds": update_elapsed,
            "rollout_steps": int(len(rollout["rewards"])),
            "projected_total_seconds": projected_total,
        })
        update_records.append(metrics)
        print(
            f"update={update + 1} loss={metrics['loss']:.4f} "
            f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f} "
            f"entropy={metrics['entropy']:.4f} mean_reward={metrics['mean_reward']:.5f} "
            f"elapsed={update_elapsed:.3f}s projected_total={projected_total:.3f}s"
        )
    total_elapsed = time.perf_counter() - started
    metrics_path = Path(args.metrics) if args.metrics else Path(args.checkpoint).with_suffix(".train.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({
        "mode": "train",
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "updates_requested": args.updates,
        "updates_completed": len(update_records),
        "episodes_per_update": args.episodes_per_update,
        "generations": args.generations,
        "population": args.population,
        "elapsed_seconds": total_elapsed,
        "projected_total_seconds_last": update_records[-1]["projected_total_seconds"] if update_records else 0.0,
        "updates": update_records,
    }, indent=2), encoding="utf-8")
    print(f"saved checkpoint to {args.checkpoint}")
    print(f"training elapsed={total_elapsed:.3f}s; metrics={metrics_path}")


if __name__ == "__main__":
    main()
