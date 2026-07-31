from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .schema import DynamicEvent, MTSPInstance


def _coordinates(rng: np.random.Generator, count: int, distribution: str, width: float) -> np.ndarray:
    if distribution == "uniform":
        return rng.uniform(0, width, size=(count, 2))
    if distribution == "clustered":
        centers = rng.uniform(width * 0.15, width * 0.85, size=(3, 2))
        labels = rng.integers(0, len(centers), size=count)
        return np.clip(centers[labels] + rng.normal(0, width * 0.08, size=(count, 2)), 0, width)
    if distribution == "hotspot":
        center = rng.uniform(width * 0.25, width * 0.75, size=2)
        return np.clip(center + rng.normal(0, width * 0.13, size=(count, 2)), 0, width)
    raise ValueError("distribution must be uniform, clustered, or hotspot")


def generate_instance(
    task_count: int,
    vehicle_count: int,
    seed: int,
    distribution: str = "uniform",
    dynamic_rate: float = 0.0,
    preference: tuple[float, float] = (0.5, 0.5),
    width: float = 100.0,
) -> MTSPInstance:
    if task_count < 2 or vehicle_count < 1:
        raise ValueError("task_count must be >= 2 and vehicle_count must be >= 1")
    rng = np.random.default_rng(seed)
    dynamic_count = int(round(task_count * dynamic_rate))
    coordinates = _coordinates(rng, task_count + dynamic_count, distribution, width)
    events = [
        # arrival_generation means "at the start of this generation"; see DATASET_FORMAT.md.
        DynamicEvent(arrival_generation=10 * (index + 1), task_id=task_count + index + 1, coordinate=tuple(coordinates[task_count + index]))
        for index in range(dynamic_count)
    ]
    return MTSPInstance(
        instance_id=f"{distribution}_v{vehicle_count}_n{task_count}_{seed:06d}",
        vehicle_starts=_coordinates(rng, vehicle_count, "uniform", width),
        task_coordinates=coordinates[:task_count],
        task_ids=np.arange(1, task_count + 1),
        preference=preference,
        events=events,
        seed=seed,
    )


def generate_dataset(
    output: str | Path,
    instances: int,
    task_count: int,
    vehicle_count: int,
    seed: int = 0,
    distribution: str = "uniform",
    dynamic_rate: float = 0.0,
) -> list[Path]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    manifest = []
    for index, instance_seed in enumerate(rng.integers(0, 2**31 - 1, size=instances)):
        instance = generate_instance(task_count, vehicle_count, int(instance_seed), distribution, dynamic_rate)
        path = output / f"{instance.instance_id}.json"
        instance.save(path)
        paths.append(path)
        manifest.append({"index": index, "path": path.name, "seed": int(instance_seed), "events": len(instance.events)})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths
