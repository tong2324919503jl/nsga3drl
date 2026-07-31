from __future__ import annotations

import argparse

from nsga3drl.data.generator import generate_dataset
from nsga3drl.data.validation import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible Open MTSP instances.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--vehicles", type=int, default=3)
    parser.add_argument("--distribution", choices=("uniform", "clustered", "hotspot"), default="uniform")
    parser.add_argument("--dynamic-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    paths = generate_dataset(args.output, args.instances, args.tasks, args.vehicles, args.seed, args.distribution, args.dynamic_rate)
    reports = validate_dataset(args.output)
    failures = [report for report in reports if not report.valid]
    if failures:
        raise SystemExit("dataset validation failed: " + "; ".join(error for report in failures for error in report.errors))
    print(f"Generated and validated {len(paths)} instances in {args.output}")


if __name__ == "__main__":
    main()
