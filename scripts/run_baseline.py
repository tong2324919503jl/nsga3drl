from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from nsga3drl.algorithms.nsga3 import NSGA3
from nsga3drl.data.generator import generate_instance
from nsga3drl.data.io import load_dataset
from nsga3drl.data.schema import MTSPInstance
from nsga3drl.dynamic.execution import execute_routes
from nsga3drl.problem.encoding import decode, encode
from nsga3drl.problem.open_mtsp import OpenMTSPProblem


def evaluate_instance(instance: MTSPInstance, population: int, generations: int, seed: int, preference_override):
    started = time.perf_counter()
    algorithm = NSGA3(OpenMTSPProblem(instance), population, seed=seed)
    result = algorithm.run(generations)
    preference = preference_override if preference_override is not None else instance.preference
    if preference is None:
        raise ValueError(f"{instance.instance_id} has no preference; supply --preference EFFICIENCY FAIRNESS")
    preferred = result.preferred_index(preference)
    preferred_genome = result.population[preferred]
    initial_problem = OpenMTSPProblem(instance.without_events())
    initial_objective = result.objectives[preferred]
    initial_lengths = initial_problem.route_lengths(preferred_genome).tolist()
    execution = execute_routes(instance, decode(preferred_genome, instance.vehicle_count))
    final_problem = OpenMTSPProblem(execution.instance)
    final_genome = encode(execution.routes, instance.vehicle_count)
    final_objective = final_problem.evaluate(final_genome)
    elapsed = time.perf_counter() - started
    preferred_solution = {
        "preference": list(preference),
        "initial_total_distance": float(initial_objective[0]),
        "initial_variance": float(initial_objective[1]),
        "initial_vehicle_route_lengths": initial_lengths,
        "total_distance": float(final_objective[0]),
        "variance": float(final_objective[1]),
        "vehicle_route_lengths": final_problem.route_lengths(final_genome).tolist(),
    }
    return {
        "instance_id": instance.instance_id,
        "final_hv": float(result.history[-1]["hypervolume"]),
        "reward_sum": 0.0,
        "events": len(execution.event_log),
        "active_tasks": final_problem.task_count,
        "dynamic_time_seconds": execution.elapsed_seconds,
        "elapsed_seconds": elapsed,
        "preferred_solution": preferred_solution,
        "pareto_solutions": len(result.pareto_indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NSGA-III baseline on one instance or a dataset.")
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--vehicles", type=int, default=3)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preference", type=float, nargs=2, default=None, metavar=("EFFICIENCY", "FAIRNESS"))
    parser.add_argument("--instance", help="Path to one instance JSON.")
    parser.add_argument("--dataset", help="Directory of instance JSON files; evaluates every instance.")
    parser.add_argument("--metrics", help="JSON report path for dataset or single-instance results.")
    args = parser.parse_args()
    if args.instance and args.dataset:
        parser.error("use only one of --instance and --dataset")
    if args.dataset:
        instances = load_dataset(args.dataset)
    elif args.instance:
        instances = [MTSPInstance.load(args.instance)]
    else:
        instances = [generate_instance(args.tasks, args.vehicles, args.seed)]

    started = time.perf_counter()
    results = []
    for index, instance in enumerate(instances):
        item = evaluate_instance(instance, args.population, args.generations, args.seed + index, tuple(args.preference) if args.preference else None)
        results.append(item)
        elapsed = time.perf_counter() - started
        projected = elapsed / (index + 1) * len(instances)
        preferred = item["preferred_solution"]
        print(
            f"instance={instance.instance_id} final_hv={item['final_hv']:.4f} reward_sum=0.00000 "
            f"events={item['events']} active_tasks={item['active_tasks']} elapsed={item['elapsed_seconds']:.3f}s "
            f"projected_total={projected:.3f}s preferred_distance={preferred['total_distance']:.3f} "
            f"preferred_variance={preferred['variance']:.3f} initial_distance={preferred['initial_total_distance']:.3f} "
            f"initial_variance={preferred['initial_variance']:.3f} preference={preferred['preference']}"
        )
    total_elapsed = time.perf_counter() - started
    metrics_path = Path(args.metrics) if args.metrics else None
    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        mean_elapsed = sum(item["elapsed_seconds"] for item in results) / len(results)
        metrics_path.write_text(json.dumps({
            "mode": "baseline",
            "dataset": args.dataset,
            "instances": len(results),
            "generations": args.generations,
            "population": args.population,
            "elapsed_seconds": total_elapsed,
            "mean_instance_seconds": mean_elapsed,
            "results": results,
        }, indent=2), encoding="utf-8")
    print(f"summary instances={len(results)} mean_elapsed={total_elapsed / len(results):.3f}s elapsed={total_elapsed:.3f}s metrics={metrics_path}")


if __name__ == "__main__":
    main()
