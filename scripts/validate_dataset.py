from __future__ import annotations

import argparse

from nsga3drl.data.validation import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MTSP dataset files and their manifest.")
    parser.add_argument("directory")
    parser.add_argument("--coordinate-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    reports = validate_dataset(args.directory, args.coordinate_tolerance)
    invalid = [report for report in reports if not report.valid]
    for report in reports:
        status = "OK" if report.valid else "ERROR: " + ", ".join(report.errors)
        print(f"{status} tasks={report.task_count} vehicles={report.vehicle_count} events={report.event_count}")
        for warning in report.warnings:
            print(f"WARNING: {warning}")
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
