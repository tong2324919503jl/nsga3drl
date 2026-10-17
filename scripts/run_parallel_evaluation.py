from __future__ import annotations

import argparse
import os
from pathlib import Path

# Prevent nested math-library thread pools from oversubscribing the worker pool.
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

from nsga3drl.data.io import load_dataset
from nsga3drl.experiments.batch import parse_checkpoints, parse_preferences, run_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel, resumable evaluation across instances, preferences, models and dynamic strategies."
    )
    parser.add_argument("--dataset", required=True, help="Directory containing instance JSON files")
    parser.add_argument(
        "--checkpoint", action="append", metavar="LABEL=PATH",
        help="Repeat for PPO model versions; a bare path uses its filename as the label",
    )
    parser.add_argument("--method", action="append", choices=("ppo", "baseline"),
                        help="Repeat to select methods; defaults to PPO and baseline")
    parser.add_argument("--dynamic-strategy", action="append",
                        choices=("greedy_score_then_tsp", "tsp_scored"),
                        help="Repeat to select strategies; defaults to both")
    parser.add_argument("--preference", action="append", nargs=2, type=float,
                        metavar=("EFFICIENCY", "FAIRNESS"),
                        help="Repeat for a preference grid; weights must sum to 1")
    parser.add_argument("--generations", type=int, default=240)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--output-dir", default="results/parallel_evaluation")
    parser.add_argument("--state-db", help="SQLite resume database; defaults to <output-dir>/resume.sqlite3")
    args = parser.parse_args()

    args.method = args.method or ["ppo", "baseline"]
    args.dynamic_strategy = args.dynamic_strategy or ["greedy_score_then_tsp", "tsp_scored"]
    if args.workers < 1 or args.generations < 1 or args.population < 4:
        parser.error("workers/generations must be positive and population must be at least 4")

    try:
        preferences = parse_preferences(args.preference)
        checkpoints = parse_checkpoints(args.checkpoint)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if "ppo" in args.method and not checkpoints:
        parser.error("at least one --checkpoint is required when method ppo is selected")

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        parser.error(f"dataset directory does not exist: {dataset_dir}")
    dataset_files = sorted(path for path in dataset_dir.glob("*.json") if path.name != "manifest.json")
    if not dataset_files:
        parser.error(f"no instance JSON files found in {dataset_dir}")
    instances = load_dataset(dataset_dir)
    args.expected_instances = len(instances)
    run_batch(args, dataset_files, instances, preferences, checkpoints)


if __name__ == "__main__":
    main()
