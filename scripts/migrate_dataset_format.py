"""Convert legacy dataset event fields to the current schema.

Only the event trigger key is renamed. Coordinates, task IDs, vehicle starts,
task order, preferences, and event order are copied without regeneration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nsga3drl.data.validation import validate_dataset


def migrate_dataset(source: str | Path, destination: str | Path) -> int:
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir():
        raise FileNotFoundError(f"source dataset does not exist: {source}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in source.glob("*.json") if path.name != "manifest.json")
    if not files:
        raise ValueError(f"no instance JSON files found in: {source}")
    manifest_entries = []
    for index, path in enumerate(files):
        raw = json.loads(path.read_text(encoding="utf-8"))
        converted_events = []
        for event in raw.get("events", []):
            converted = dict(event)
            # Legacy datasets use arrival_generation for the same 10-task
            # completion schedule. Do not alter coordinates or event order.
            if "arrival_completed_tasks" not in converted:
                if "arrival_generation" not in converted:
                    raise ValueError(f"event in {path} has no arrival trigger field")
                converted["arrival_completed_tasks"] = converted.pop("arrival_generation")
            else:
                converted.pop("arrival_generation", None)
            converted_events.append(converted)
        raw["events"] = converted_events
        output_path = destination / path.name
        output_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        manifest_entries.append({
            "index": index,
            "path": path.name,
            "seed": raw.get("seed"),
            "events": len(converted_events),
        })
    (destination / "manifest.json").write_text(json.dumps(manifest_entries, indent=2), encoding="utf-8")
    reports = validate_dataset(destination)
    failures = [error for report in reports if not report.valid for error in report.errors]
    if failures:
        raise ValueError("migrated dataset failed validation: " + "; ".join(failures))
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy MTSP JSON files without changing coordinates.")
    parser.add_argument("--input", required=True, help="Legacy dataset directory")
    parser.add_argument("--output", required=True, help="New dataset directory (must be empty or absent)")
    args = parser.parse_args()
    count = migrate_dataset(args.input, args.output)
    print(f"Migrated and validated {count} instances: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
