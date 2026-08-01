from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from nsga3drl.data.io import load_dataset
from nsga3drl.algorithms.nsga3 import NSGA3Result
from nsga3drl.rl.environment import OperatorSelectionEnv
from nsga3drl.rl.ppo import PPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO operator selector on an independent dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics", help="JSON report path; defaults to <checkpoint>.eval.json")
    args = parser.parse_args()
    trainer = PPOTrainer.load(args.checkpoint)
    started = time.perf_counter()
    instances = load_dataset(args.dataset)
    results = []
    for index, instance in enumerate(instances):
        instance_started = time.perf_counter()
        environment = OperatorSelectionEnv(
            instance,
            population_size=args.population,
            generations=args.generations,
            seed=args.seed + index,
        )
        observation = environment.reset()
        done = False
        reward_sum = 0.0
        events = 0
        while not done:
            observation, reward, done, info = environment.step(trainer.greedy_action(observation))
            reward_sum += reward
            events += info["events_injected"]
        instance_elapsed = time.perf_counter() - instance_started
        algorithm = environment.algorithm
        result = NSGA3Result(algorithm.population, algorithm.objectives, algorithm.history)
        preferred_index = result.preferred_index(instance.preference)
        preferred_genome = result.population[preferred_index]
        preferred_objective = result.objectives[preferred_index]
        vehicle_route_lengths = algorithm.problem.route_lengths(preferred_genome).tolist()
        preferred = {
            "preference": list(instance.preference),
            "total_distance": float(preferred_objective[0]),
            "variance": float(preferred_objective[1]),
            "vehicle_route_lengths": vehicle_route_lengths,
        }
        results.append({
            "instance_id": instance.instance_id,
            "final_hv": float(info["hypervolume"]),
            "reward_sum": float(reward_sum),
            "events": events,
            "active_tasks": algorithm.problem.task_count,
            "elapsed_seconds": instance_elapsed,
            "preferred_solution": preferred,
        })
        elapsed = time.perf_counter() - started
        projected_total = elapsed / (index + 1) * len(instances)
        print(
            f"instance={instance.instance_id} final_hv={info['hypervolume']:.4f} "
            f"reward_sum={reward_sum:.5f} events={events} active_tasks={environment.algorithm.problem.task_count} "
            f"elapsed={instance_elapsed:.3f}s projected_total={projected_total:.3f}s "
            f"preferred_distance={preferred['total_distance']:.3f} preferred_variance={preferred['variance']:.3f} "
            f"preference={preferred['preference']}"
        )
    values = np.asarray([[item["final_hv"], item["reward_sum"], item["events"], item["active_tasks"], item["elapsed_seconds"]] for item in results], dtype=float)
    total_elapsed = time.perf_counter() - started
    metrics_path = Path(args.metrics) if args.metrics else Path(args.checkpoint).with_suffix(".eval.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({
        "mode": "evaluation",
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "instances": len(instances),
        "generations": args.generations,
        "population": args.population,
        "elapsed_seconds": total_elapsed,
        "projected_total_seconds_last": (total_elapsed / len(instances) * len(instances)) if instances else 0.0,
        "mean_instance_seconds": float(values[:, 4].mean()) if len(values) else 0.0,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(
        f"summary instances={len(results)} mean_final_hv={values[:, 0].mean():.4f} "
        f"mean_reward_sum={values[:, 1].mean():.5f} mean_events={values[:, 2].mean():.2f} "
        f"elapsed={total_elapsed:.3f}s metrics={metrics_path}"
    )


if __name__ == "__main__":
    main()
