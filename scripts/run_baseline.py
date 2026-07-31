from __future__ import annotations

import argparse
import time

from nsga3drl.algorithms.nsga3 import NSGA3
from nsga3drl.data.generator import generate_instance
from nsga3drl.data.schema import MTSPInstance
from nsga3drl.problem.open_mtsp import OpenMTSPProblem


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NSGA-III baseline on one generated instance.")
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--vehicles", type=int, default=3)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instance", help="Path to one generated instance JSON. Overrides --tasks and --vehicles.")
    args = parser.parse_args()
    instance = MTSPInstance.load(args.instance) if args.instance else generate_instance(args.tasks, args.vehicles, args.seed)
    started = time.perf_counter()
    algorithm = NSGA3(OpenMTSPProblem(instance), args.population, seed=args.seed)
    result = algorithm.run(args.generations)
    elapsed = time.perf_counter() - started
    pareto = result.pareto_indices
    if instance.events:
        print(
            f"dynamic events processed={len(algorithm.event_log)}/{len(instance.events)}; "
            f"final active tasks={algorithm.problem.task_count}; "
            f"dynamic_time={sum(item['elapsed_seconds'] for item in algorithm.event_log):.3f}s"
        )
    print(f"final HV={result.history[-1]['hypervolume']:.4f}; pareto solutions={len(pareto)}; elapsed={elapsed:.3f}s")
    for objective in result.objectives[pareto[:5]]:
        print(f"total_distance={objective[0]:.3f}, variance={objective[1]:.3f}")


if __name__ == "__main__":
    main()
