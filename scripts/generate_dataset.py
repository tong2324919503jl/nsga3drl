from __future__ import annotations

import argparse

from nsga3drl.data.generator import BALANCED_PREFERENCES, generate_dataset
from nsga3drl.data.validation import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible Open MTSP instances.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--vehicles", type=int, default=3)
    parser.add_argument("--distribution", choices=("uniform", "clustered", "hotspot", "mixed"), default="uniform")
    parser.add_argument("--dynamic-rate", type=float, default=0.1)
    parser.add_argument("--preference-mode", choices=("fixed", "balanced5", "extremes"), default="fixed")
    parser.add_argument("--preference", type=float, nargs=2, default=(0.5, 0.5), metavar=("EFFICIENCY", "FAIRNESS"))
    parser.add_argument("--paired-preferences", action="store_true", help="Create identical spatial instances for every configured preference.")
    parser.add_argument("--omit-preference", action="store_true", help="Do not write preference into instance JSON; supply it at runtime.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.preference_mode == "balanced5":
        preferences = BALANCED_PREFERENCES
    elif args.preference_mode == "extremes":
        # Keep the two endpoint preferences balanced for direct efficiency
        # versus fairness comparisons.
        preferences = ((1.0, 0.0), (0.0, 1.0))
    else:
        preferences = (tuple(args.preference),)
    distributions = ("uniform", "clustered", "hotspot") if args.distribution == "mixed" else (args.distribution,)
    paths = generate_dataset(
        args.output, args.instances, args.tasks, args.vehicles, args.seed,
        args.distribution, args.dynamic_rate, preferences, distributions, args.paired_preferences, not args.omit_preference,
    )
    reports = validate_dataset(args.output)
    failures = [report for report in reports if not report.valid]
    if failures:
        raise SystemExit("dataset validation failed: " + "; ".join(error for report in failures for error in report.errors))
    print(f"Generated and validated {len(paths)} instances in {args.output}")


if __name__ == "__main__":
    main()
