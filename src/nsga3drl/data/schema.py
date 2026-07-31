from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DynamicEvent:
    arrival_generation: int
    task_id: int
    coordinate: tuple[float, float]


@dataclass
class MTSPInstance:
    instance_id: str
    vehicle_starts: np.ndarray
    task_coordinates: np.ndarray
    task_ids: np.ndarray | None = None
    preference: tuple[float, float] = (0.5, 0.5)
    events: list[DynamicEvent] = field(default_factory=list)
    seed: int | None = None

    def __post_init__(self) -> None:
        self.vehicle_starts = np.asarray(self.vehicle_starts, dtype=float)
        self.task_coordinates = np.asarray(self.task_coordinates, dtype=float)
        if self.vehicle_starts.ndim != 2 or self.vehicle_starts.shape[1] != 2:
            raise ValueError("vehicle_starts must have shape (vehicles, 2)")
        if self.task_coordinates.ndim != 2 or self.task_coordinates.shape[1] != 2:
            raise ValueError("task_coordinates must have shape (tasks, 2)")
        if self.task_ids is None:
            self.task_ids = np.arange(1, len(self.task_coordinates) + 1, dtype=int)
        else:
            self.task_ids = np.asarray(self.task_ids, dtype=int)
        if len(self.task_ids) != len(self.task_coordinates) or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique and match task_coordinates")
        if min(self.preference) < 0 or not np.isclose(sum(self.preference), 1.0):
            raise ValueError("preference must be non-negative and sum to 1")

    @property
    def vehicle_count(self) -> int:
        return len(self.vehicle_starts)

    @property
    def task_count(self) -> int:
        return len(self.task_ids)

    def with_event_task(self, event: DynamicEvent) -> "MTSPInstance":
        if event.task_id in self.task_ids:
            raise ValueError(f"duplicate task id {event.task_id}")
        return MTSPInstance(
            instance_id=self.instance_id,
            vehicle_starts=self.vehicle_starts.copy(),
            task_coordinates=np.vstack([self.task_coordinates, event.coordinate]),
            task_ids=np.append(self.task_ids, event.task_id),
            preference=self.preference,
            events=self.events,
            seed=self.seed,
        )

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "vehicle_starts": self.vehicle_starts.tolist(),
            "task_coordinates": self.task_coordinates.tolist(),
            "task_ids": self.task_ids.tolist(),
            "preference": list(self.preference),
            "events": [asdict(event) for event in self.events],
            "seed": self.seed,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MTSPInstance":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["events"] = [DynamicEvent(**event) for event in raw.get("events", [])]
        return cls(**raw)
