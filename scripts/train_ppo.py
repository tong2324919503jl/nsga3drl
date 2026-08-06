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


class EpochSampler:
    """Without-replacement dataset sampler; every instance is visited per epoch."""

    def __init__(self, instances, rng: np.random.Generator):
        self.instances = instances
        self.rng = rng
        self.order = rng.permutation(len(instances))
        self.position = 0
        self.completed_epochs = 0

    def take(self, count: int):
        selected = []
        for _ in range(count):
            if self.position == len(self.order):
                self.order = self.rng.permutation(len(self.instances))
                self.position = 0
                self.completed_epochs += 1
            selected.append(self.instances[int(self.order[self.position])])
            self.position += 1
        return selected


def evaluate_validation(
    trainer: PPOTrainer,
    instances,
    population: int,
    generations: int,
    seed: int,
    preference: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Run deterministic, no-gradient validation without changing PPO weights."""
    rewards = []
    elapsed_started = time.perf_counter()
    for index, instance in enumerate(instances):
        environment = OperatorSelectionEnv(
            instance, population_size=population, generations=generations,
            seed=seed + index, preference=preference,
        )
        observation = environment.reset()
        done = False
        reward_sum = 0.0
        while not done:
            observation, reward, done, _ = environment.step(trainer.greedy_action(observation))
            reward_sum += reward
        rewards.append(reward_sum)
    return {
        "mean_reward_sum": float(np.mean(rewards)),
        "elapsed_seconds": time.perf_counter() - elapsed_started,
        "instances": len(instances),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO to choose the NSGA-III mutation operator.")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--vehicles", type=int, default=3)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preference", type=float, nargs=2, default=None, metavar=("EFFICIENCY", "FAIRNESS"), help="Override dataset preference for every training instance.")
    parser.add_argument("--dataset", help="Directory of generated instance JSON files. Overrides --tasks and --vehicles.")
    parser.add_argument("--checkpoint", default="checkpoints/ppo_operator_selector.pt")
    parser.add_argument("--resume", help="Resume PPO model and optimizer state from this checkpoint.")
    parser.add_argument("--metrics", help="JSON report path; defaults to <checkpoint>.train.json")
    parser.add_argument("--sampling", choices=("epoch", "random"), default="epoch")
    parser.add_argument("--validation-dataset", help="Independent dataset used to select the best checkpoint.")
    parser.add_argument("--validation-every", type=int, default=0, help="Validate every N updates; 0 disables validation.")
    parser.add_argument("--validation-instances", type=int, default=10, help="Maximum validation instances per validation pass; 0 uses all.")
    args = parser.parse_args()
    cli_preference = tuple(args.preference) if args.preference is not None else None
    trainer = PPOTrainer.load(args.resume) if args.resume else PPOTrainer(
        OperatorSelectionEnv.observation_size, OperatorSelectionEnv.action_count, seed=args.seed
    )
    dataset = load_dataset(args.dataset) if args.dataset else None
    if dataset and any(instance.events for instance in dataset):
        print("note: dynamic events are excluded from PPO training and handled after the initial plan")
    rng = np.random.default_rng(args.seed)
    epoch_sampler = EpochSampler(dataset, rng) if dataset and args.sampling == "epoch" else None
    validation_set = load_dataset(args.validation_dataset) if args.validation_dataset else []
    if args.validation_instances > 0:
        validation_set = validation_set[:args.validation_instances]
    if args.validation_every and not validation_set:
        raise ValueError("--validation-every requires --validation-dataset")
    best_validation_reward = float("-inf")
    best_checkpoint = Path(args.checkpoint).with_name(Path(args.checkpoint).stem + ".best.pt")
    started = time.perf_counter()
    update_records = []
    for update in range(args.updates):
        update_started = time.perf_counter()
        # Epoch sampling visits every instance once before starting a reshuffled pass.
        selected = (
            epoch_sampler.take(args.episodes_per_update) if epoch_sampler
            else [dataset[int(index)] for index in rng.integers(0, len(dataset), size=args.episodes_per_update)]
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
                preference=cli_preference,
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
            "dataset_epochs_completed": epoch_sampler.completed_epochs if epoch_sampler else None,
        })
        if args.validation_every and (update + 1) % args.validation_every == 0:
            validation = evaluate_validation(
                trainer,
                validation_set,
                args.population,
                args.generations,
                args.seed + 10_000,
                cli_preference,
            )
            metrics["validation"] = validation
            if validation["mean_reward_sum"] > best_validation_reward:
                best_validation_reward = validation["mean_reward_sum"]
                trainer.save(best_checkpoint)
                metrics["best_checkpoint_updated"] = True
            print(
                f"validation update={update + 1} mean_reward_sum={validation['mean_reward_sum']:.5f} "
                f"elapsed={validation['elapsed_seconds']:.3f}s best={best_validation_reward:.5f}"
            )
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
        "sampling": args.sampling,
        "dataset_epochs_completed": epoch_sampler.completed_epochs if epoch_sampler else None,
        "validation_dataset": args.validation_dataset,
        "validation_every": args.validation_every,
        "best_validation_reward": best_validation_reward if validation_set else None,
        "best_checkpoint": str(best_checkpoint) if validation_set else None,
        "elapsed_seconds": total_elapsed,
        "projected_total_seconds_last": update_records[-1]["projected_total_seconds"] if update_records else 0.0,
        "updates": update_records,
    }, indent=2), encoding="utf-8")
    print(f"saved checkpoint to {args.checkpoint}")
    print(f"training elapsed={total_elapsed:.3f}s; metrics={metrics_path}")


if __name__ == "__main__":
    main()
