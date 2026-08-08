from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DynamicEvent:
    task_id: int
    coordinate: tuple[float, float]
    arrival_completed_tasks: int | None = None
    # Kept for loading old datasets; new datasets use arrival_completed_tasks.
    arrival_generation: int | None = None

    def __post_init__(self) -> None:
        completed = self.arrival_completed_tasks
        if completed is None:
            completed = self.arrival_generation
            object.__setattr__(self, "arrival_completed_tasks", completed)
        if completed is None or completed < 0:
            raise ValueError("arrival_completed_tasks must be non-negative")


@dataclass
class MTSPInstance:
    instance_id: str
    vehicle_starts: np.ndarray
    task_coordinates: np.ndarray
    task_ids: np.ndarray | None = None
    # ``None`` is intentional: a JSON instance without a preference must stay
    # preference-free so callers can provide weights explicitly at runtime.
    preference: tuple[float, float] | None = None
    events: list[DynamicEvent] = field(default_factory=list)
    seed: int | None = None

    def __post_init__(self) -> None:
        self.vehicle_starts = np.asarray(self.vehicle_starts, dtype=float)
        self.task_coordinates = np.asarray(self.task_coordinates, dtype=float)
        # JSON arrays load as lists; keep the public preference representation
        # consistent with the declared tuple type across generated and loaded data.
        if self.preference is not None:
            self.preference = tuple(float(value) for value in self.preference)
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
        if self.preference is not None and (len(self.preference) != 2 or min(self.preference) < 0 or not np.isclose(sum(self.preference), 1.0)):
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

    def with_preference(self, preference: tuple[float, float]) -> "MTSPInstance":
        return MTSPInstance(
            instance_id=self.instance_id,
            vehicle_starts=self.vehicle_starts.copy(),
            task_coordinates=self.task_coordinates.copy(),
            task_ids=self.task_ids.copy(),
            preference=preference,
            events=list(self.events),
            seed=self.seed,
        )

    def without_events(self) -> "MTSPInstance":
        return MTSPInstance(
            instance_id=self.instance_id,
            vehicle_starts=self.vehicle_starts.copy(),
            task_coordinates=self.task_coordinates.copy(),
            task_ids=self.task_ids.copy(),
            preference=self.preference,
            events=[],
            seed=self.seed,
        )

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "vehicle_starts": self.vehicle_starts.tolist(),
            "task_coordinates": self.task_coordinates.tolist(),
            "task_ids": self.task_ids.tolist(),
            **({"preference": list(self.preference)} if self.preference is not None else {}),
            "events": [
                {"task_id": event.task_id, "coordinate": list(event.coordinate), "arrival_completed_tasks": event.arrival_completed_tasks}
                for event in self.events
            ],
            "seed": self.seed,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MTSPInstance":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["events"] = [DynamicEvent(**event) for event in raw.get("events", [])]
        return cls(**raw)
