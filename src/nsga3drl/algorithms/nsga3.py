from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import time

import numpy as np

from ..data.schema import DynamicEvent
from ..dynamic.insertion import DynamicInsertionPlanner
from ..problem.encoding import decode, encode, random_genome
from ..problem.open_mtsp import OpenMTSPProblem


def dominates(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all(left <= right) and np.any(left < right))


def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    domination_sets = [[] for _ in objectives]
    dominated_by = np.zeros(len(objectives), dtype=int)
    fronts: list[list[int]] = [[]]
    for p in range(len(objectives)):
        for q in range(len(objectives)):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                domination_sets[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                dominated_by[p] += 1
        if dominated_by[p] == 0:
            fronts[0].append(p)
    index = 0
    while index < len(fronts) and fronts[index]:
        next_front: list[int] = []
        for p in fronts[index]:
            for q in domination_sets[p]:
                dominated_by[q] -= 1
                if dominated_by[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        index += 1
    return fronts


def reference_directions(partitions: int) -> np.ndarray:
    return np.asarray([(i / partitions, 1.0 - i / partitions) for i in range(partitions + 1)], dtype=float)


def hypervolume_2d(objectives: np.ndarray, reference: np.ndarray | None = None) -> float:
    """Exact two-objective minimisation HV after removing dominated points."""
    if len(objectives) == 0:
        return 0.0
    if reference is None:
        reference = np.max(objectives, axis=0) * 1.1 + 1e-9
    points = objectives[np.all(objectives < reference, axis=1)]
    if not len(points):
        return 0.0
    front = points[non_dominated_sort(points)[0]]
    front = front[np.argsort(front[:, 0])]
    area = 0.0
    current_y = float(reference[1])
    for x, y in front:
        if y < current_y:
            area += (reference[0] - x) * (current_y - y)
            current_y = y
    return float(max(area, 0.0))


def _normalize(objectives: np.ndarray) -> np.ndarray:
    low = objectives.min(axis=0)
    high = objectives.max(axis=0)
    return (objectives - low) / np.maximum(high - low, 1e-12)


def _perpendicular_distances(values: np.ndarray, directions: np.ndarray) -> np.ndarray:
    direction_norms = np.linalg.norm(directions, axis=1)
    unit = directions / np.maximum(direction_norms[:, None], 1e-12)
    projection = values @ unit.T
    return np.linalg.norm(values[:, None, :] - projection[:, :, None] * unit[None, :, :], axis=2)


@dataclass
class NSGA3Result:
    population: np.ndarray
    objectives: np.ndarray
    history: list[dict]

    @property
    def pareto_indices(self) -> np.ndarray:
        return np.asarray(non_dominated_sort(self.objectives)[0], dtype=int)

    def preferred_index(self, preference: tuple[float, float]) -> int:
        """Select one final Pareto solution after normalising both objectives."""
        if len(preference) != 2 or min(preference) < 0 or not np.isclose(sum(preference), 1.0):
            raise ValueError("preference must contain two non-negative weights summing to 1")
        indices = self.pareto_indices
        front = self.objectives[indices]
        normalized = _normalize(front)
        return int(indices[np.argmin(normalized @ np.asarray(preference, dtype=float))])


class NSGA3:
    """Small, dependency-light NSGA-III implementation for the two-objective MVP."""

    ACTIONS = ("swap", "inversion", "relocate", "separator_shift")

    def __init__(
        self,
        problem: OpenMTSPProblem,
        population_size: int = 64,
        partitions: int = 31,
        seed: int = 0,
        dynamic_events: bool = False,
    ):
        if population_size < 4:
            raise ValueError("population_size must be at least 4")
        self.problem = problem
        self.population_size = population_size
        self.directions = reference_directions(partitions)
        self.rng = np.random.default_rng(seed)
        self.population: np.ndarray | None = None
        self.objectives: np.ndarray | None = None
        self.generation = 0
        self.history: list[dict] = []
        self.hv_reference: np.ndarray | None = None
        self.dynamic_events = dynamic_events
        self._events = sorted(problem.instance.events, key=lambda event: (event.arrival_completed_tasks, event.task_id))
        self._processed_event_ids: set[int] = set()
        self.event_log: list[dict] = []
        self.last_hv_delta = 0.0

    def initialize(self) -> None:
        self.population = np.vstack([
            random_genome(self.problem.task_count, self.problem.vehicle_count, self.rng)
            for _ in range(self.population_size)
        ])
        self.objectives = self.problem.evaluate_population(self.population)
        self.generation = 0
        self.hv_reference = self.objectives.max(axis=0) * 1.5 + 1.0
        initial_events = self.process_events(0) if self.dynamic_events else 0
        self._record(events_injected=initial_events)

    def current_hypervolume(self) -> float:
        assert self.objectives is not None and self.hv_reference is not None
        return hypervolume_2d(self.objectives, self.hv_reference)

    def _record(self, events_injected: int = 0) -> None:
        assert self.objectives is not None and self.hv_reference is not None
        fronts = non_dominated_sort(self.objectives)
        self.history.append({
            "generation": self.generation,
            "hypervolume": self.current_hypervolume(),
            "front_size": len(fronts[0]),
            "objective_mean": self.objectives.mean(axis=0).tolist(),
            "objective_std": self.objectives.std(axis=0).tolist(),
            "events_injected": events_injected,
        })

    def process_events(self, generation: int) -> int:
        """Apply events scheduled at this generation before offspring are created."""
        if self.population is None or self.objectives is None:
            raise RuntimeError("initialize before processing dynamic events")
        scheduled = [
            event for event in self._events
            if event.arrival_completed_tasks == generation and event.task_id not in self._processed_event_ids
        ]
        for event in scheduled:
            event_started = time.perf_counter()
            updated = []
            decisions = []
            for genome in self.population:
                # The planner keeps an internal coordinate index, so every
                # population member needs an independent insertion transaction.
                planner = DynamicInsertionPlanner(self.problem.instance)
                routes = decode(genome, self.problem.vehicle_count)
                routes, decision = planner.insert(routes, event.task_id, event.coordinate)
                updated.append(encode(routes, self.problem.vehicle_count))
                decisions.append(decision)
            self.problem = OpenMTSPProblem(self.problem.instance.with_event_task(event))
            self.population = np.vstack(updated)
            self.objectives = self.problem.evaluate_population(self.population)
            # The objective space expands after a new task; reset the HV reference
            # so rewards compare only states from the same dynamic problem segment.
            self.hv_reference = self.objectives.max(axis=0) * 1.5 + 1.0
            self._processed_event_ids.add(event.task_id)
            self.event_log.append({
                "generation": generation,
                "task_id": event.task_id,
                "mean_distance_delta": float(np.mean([item.distance_delta for item in decisions])),
                "local_search_improvements": int(sum(item.local_search_improved for item in decisions)),
                "elapsed_seconds": time.perf_counter() - event_started,
            })
        if scheduled:
            self.last_hv_delta = 0.0
        return len(scheduled)

    def _tournament(self) -> np.ndarray:
        assert self.population is not None and self.objectives is not None
        candidates = self.rng.choice(len(self.population), size=2, replace=False)
        fronts = non_dominated_sort(self.objectives)
        rank = {item: index for index, front in enumerate(fronts) for item in front}
        winner = min(candidates, key=lambda item: rank[int(item)])
        return self.population[winner]

    def _ox(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Order crossover for task order, retaining left parent's route sizes."""
        task_count = self.problem.task_count
        left_tasks = left[left != 0].tolist()
        right_tasks = right[right != 0].tolist()
        start, end = sorted(self.rng.choice(task_count, size=2, replace=False))
        child = [-1] * task_count
        child[start : end + 1] = left_tasks[start : end + 1]
        fill = [item for item in right_tasks if item not in child]
        cursor = 0
        for index in range(task_count):
            if child[index] == -1:
                child[index] = fill[cursor]
                cursor += 1
        separator_positions = np.flatnonzero(left == 0).tolist()
        values: list[int] = []
        task_cursor = 0
        for index in range(task_count + self.problem.vehicle_count - 1):
            if index in separator_positions:
                values.append(0)
            else:
                values.append(child[task_cursor])
                task_cursor += 1
        return np.asarray(values, dtype=int)

    def _mutate(self, genome: np.ndarray, action: int) -> np.ndarray:
        child = genome.copy()
        task_positions = np.flatnonzero(child != 0)
        if action == 0 and len(task_positions) >= 2:
            a, b = self.rng.choice(task_positions, size=2, replace=False)
            child[a], child[b] = child[b], child[a]
        elif action == 1 and len(task_positions) >= 2:
            a, b = sorted(self.rng.choice(task_positions, size=2, replace=False))
            child[a : b + 1] = child[a : b + 1][::-1]
        elif action == 2 and len(task_positions) >= 2:
            source, destination = self.rng.choice(task_positions, size=2, replace=False)
            value = child[source]
            child = np.delete(child, source)
            child = np.insert(child, destination, value)
        elif action == 3 and self.problem.vehicle_count > 1:
            zero = int(self.rng.choice(np.flatnonzero(child == 0)))
            move = int(self.rng.choice([-1, 1]))
            target = max(0, min(len(child) - 1, zero + move))
            child[zero], child[target] = child[target], child[zero]
        return child

    def _offspring(self, action: int | np.ndarray | None = None) -> np.ndarray:
        """Create one child per population member.

        A scalar keeps the original behaviour: every child in this generation
        uses the same mutation.  An action vector lets PPO allocate different
        mutation operators to different children without changing baseline
        behaviour.
        """
        if action is None:
            actions = np.full(self.population_size, int(self.rng.integers(len(self.ACTIONS))), dtype=int)
        elif np.isscalar(action):
            if not 0 <= int(action) < len(self.ACTIONS):
                raise ValueError("invalid mutation action")
            actions = np.full(self.population_size, int(action), dtype=int)
        else:
            actions = np.asarray(action, dtype=int)
            if actions.shape != (self.population_size,) or np.any((actions < 0) | (actions >= len(self.ACTIONS))):
                raise ValueError("action vector must contain one valid action per child")
        children = []
        while len(children) < self.population_size:
            child = self._ox(self._tournament(), self._tournament())
            children.append(self._mutate(child, int(actions[len(children)])))
        return np.vstack(children)

    def _select(self, combined_population: np.ndarray, combined_objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected: list[int] = []
        fronts = non_dominated_sort(combined_objectives)
        split_front: list[int] = []
        for front in fronts:
            if len(selected) + len(front) <= self.population_size:
                selected.extend(front)
            else:
                split_front = front
                break
        remaining = self.population_size - len(selected)
        if remaining:
            normalized = _normalize(combined_objectives)
            distances = _perpendicular_distances(normalized, self.directions)
            niches = np.argmin(distances, axis=1)
            counts = np.bincount(niches[selected], minlength=len(self.directions))
            candidates = set(split_front)
            while remaining and candidates:
                niche = int(np.argmin(np.where(np.isin(np.arange(len(counts)), [niches[i] for i in candidates]), counts, np.inf)))
                in_niche = [item for item in candidates if niches[item] == niche]
                if not in_niche:
                    counts[niche] = np.iinfo(np.int32).max
                    continue
                choice = min(in_niche, key=lambda item: distances[item, niche]) if counts[niche] == 0 else int(self.rng.choice(in_niche))
                selected.append(choice)
                candidates.remove(choice)
                counts[niche] += 1
                remaining -= 1
        return combined_population[selected], combined_objectives[selected]

    def step(self, action: int | np.ndarray | None = None, process_dynamic_events: bool = True) -> dict:
        if self.population is None:
            self.initialize()
        assert self.population is not None and self.objectives is not None
        events_injected = self.process_events(self.generation + 1) if self.dynamic_events and process_dynamic_events else 0
        before_hv = self.current_hypervolume()
        offspring = self._offspring(action)
        offspring_objectives = self.problem.evaluate_population(offspring)
        self.population, self.objectives = self._select(
            np.vstack([self.population, offspring]), np.vstack([self.objectives, offspring_objectives])
        )
        self.generation += 1
        self.last_hv_delta = self.current_hypervolume() - before_hv
        self._record(events_injected=events_injected)
        return self.history[-1]

    def run(self, generations: int, action_selector: Callable[["NSGA3"], int] | None = None) -> NSGA3Result:
        if self.population is None:
            self.initialize()
        for _ in range(generations):
            self.step(None if action_selector is None else action_selector(self))
        assert self.population is not None and self.objectives is not None
        return NSGA3Result(self.population.copy(), self.objectives.copy(), list(self.history))
