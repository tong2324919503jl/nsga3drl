from __future__ import annotations

import argparse
import time

from nsga3drl.algorithms.nsga3 import NSGA3
from nsga3drl.data.generator import generate_instance
from nsga3drl.data.schema import MTSPInstance
from nsga3drl.dynamic.execution import execute_routes
from nsga3drl.problem.encoding import decode, encode
from nsga3drl.problem.open_mtsp import OpenMTSPProblem


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NSGA-III baseline on one generated instance.")
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--vehicles", type=int, default=3)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preference", type=float, nargs=2, default=None, metavar=("EFFICIENCY", "FAIRNESS"), help="Override dataset preference.")
    parser.add_argument("--instance", help="Path to one generated instance JSON. Overrides --tasks and --vehicles.")
    args = parser.parse_args()
    instance = MTSPInstance.load(args.instance) if args.instance else generate_instance(args.tasks, args.vehicles, args.seed)
    started = time.perf_counter()
    algorithm = NSGA3(OpenMTSPProblem(instance), args.population, seed=args.seed)
    result = algorithm.run(args.generations)
    preference = tuple(args.preference) if args.preference is not None else instance.preference
    if preference is None:
        raise ValueError("dataset has no preference; supply --preference EFFICIENCY FAIRNESS")
    preferred = result.preferred_index(preference)
    execution = execute_routes(instance, decode(result.population[preferred], instance.vehicle_count))
    final_problem = OpenMTSPProblem(execution.instance)
    final_genome = encode(execution.routes, instance.vehicle_count)
    final_objectives = final_problem.evaluate(final_genome)
    elapsed = time.perf_counter() - started
    pareto = result.pareto_indices
    if instance.events:
        print(
            f"dynamic events processed={len(execution.event_log)}/{len(instance.events)}; "
            f"final active tasks={final_problem.task_count}; "
            f"dynamic_time={execution.elapsed_seconds:.3f}s"
        )
    print(f"final HV={result.history[-1]['hypervolume']:.4f}; pareto solutions={len(pareto)}; elapsed={elapsed:.3f}s")
    print(
        f"preference={list(preference)}; recommended "
        f"initial_distance={result.objectives[preferred, 0]:.3f}, initial_variance={result.objectives[preferred, 1]:.3f}; "
        f"executed_distance={final_objectives[0]:.3f}, executed_variance={final_objectives[1]:.3f}"
    )
    for objective in result.objectives[pareto[:5]]:
        print(f"total_distance={objective[0]:.3f}, variance={objective[1]:.3f}")


if __name__ == "__main__":
    main()
