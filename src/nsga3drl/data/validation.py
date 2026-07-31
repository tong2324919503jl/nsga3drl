from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .schema import MTSPInstance


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: list[str]
    warnings: list[str]
    task_count: int
    vehicle_count: int
    event_count: int


def _overlapping_pairs(coordinates: np.ndarray, tolerance: float) -> list[tuple[int, int]]:
    """Return index pairs whose Euclidean separation is within tolerance."""
    if len(coordinates) < 2:
        return []
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    rows, columns = np.where(np.triu(distances <= tolerance, k=1))
    return list(zip(rows.tolist(), columns.tolist()))


def validate_instance(instance: MTSPInstance, coordinate_tolerance: float = 1e-6) -> ValidationReport:
    if coordinate_tolerance < 0:
        raise ValueError("coordinate_tolerance must be non-negative")
    errors: list[str] = []
    warnings: list[str] = []
    if not np.isfinite(instance.vehicle_starts).all() or not np.isfinite(instance.task_coordinates).all():
        errors.append("coordinates must be finite")
    if len(set(instance.task_ids.tolist())) != instance.task_count:
        errors.append("task ids are not unique")
    event_ids = [event.task_id for event in instance.events]
    if len(event_ids) != len(set(event_ids)):
        errors.append("dynamic event task ids are not unique")
    if set(event_ids).intersection(instance.task_ids.tolist()):
        errors.append("dynamic event task ids overlap initial task ids")
    if any(event.arrival_generation < 0 for event in instance.events):
        errors.append("arrival_generation must be non-negative")
    # Dynamic points are not in task_coordinates yet, but must still be distinct
    # from every initial and future task point before an experiment begins.
    all_task_ids = np.asarray(instance.task_ids.tolist() + event_ids, dtype=int)
    event_coordinates = np.asarray([event.coordinate for event in instance.events], dtype=float)
    if len(event_coordinates) and not np.isfinite(event_coordinates).all():
        errors.append("dynamic event coordinates must be finite")
    all_task_coordinates = (
        np.vstack([instance.task_coordinates, event_coordinates]) if len(event_coordinates) else instance.task_coordinates
    )
    for left, right in _overlapping_pairs(all_task_coordinates, coordinate_tolerance):
        errors.append(
            f"task coordinates overlap within tolerance {coordinate_tolerance:g}: "
            f"task_id {all_task_ids[left]} and task_id {all_task_ids[right]}"
        )
    # A vehicle may legitimately begin at a task location, so retain this as a warning.
    start_distances = np.linalg.norm(instance.vehicle_starts[:, None, :] - all_task_coordinates[None, :, :], axis=2)
    for vehicle, task_index in zip(*np.where(start_distances <= coordinate_tolerance)):
        warnings.append(
            f"vehicle {vehicle} starts at task_id {all_task_ids[task_index]} within tolerance {coordinate_tolerance:g}"
        )
    return ValidationReport(not errors, errors, warnings, instance.task_count, instance.vehicle_count, len(instance.events))


def validate_dataset(directory: str | Path, coordinate_tolerance: float = 1e-6) -> list[ValidationReport]:
    directory = Path(directory)
    reports = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "manifest.json":
            continue
        reports.append(validate_instance(MTSPInstance.load(path), coordinate_tolerance))
    manifest = directory / "manifest.json"
    if manifest.exists():
        listed = {entry["path"] for entry in json.loads(manifest.read_text(encoding="utf-8"))}
        actual = {path.name for path in directory.glob("*.json") if path.name != "manifest.json"}
        if listed != actual:
            reports.append(ValidationReport(False, ["manifest does not match instance files"], [], 0, 0, 0))
    return reports
