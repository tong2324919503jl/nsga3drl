from __future__ import annotations

import numpy as np


def decode(genome: np.ndarray | list[int], vehicles: int) -> list[list[int]]:
    routes = [[] for _ in range(vehicles)]
    vehicle = 0
    for value in np.asarray(genome, dtype=int):
        if value == 0:
            vehicle += 1
        elif vehicle < vehicles:
            routes[vehicle].append(int(value))
        else:
            raise ValueError("too many separators")
    if vehicle != vehicles - 1:
        raise ValueError("wrong separator count")
    return routes


def encode(routes: list[list[int]], vehicles: int | None = None) -> np.ndarray:
    if vehicles is None:
        vehicles = len(routes)
    if len(routes) != vehicles:
        raise ValueError("route count does not match vehicles")
    values: list[int] = []
    for index, route in enumerate(routes):
        values.extend(route)
        if index < vehicles - 1:
            values.append(0)
    return np.asarray(values, dtype=int)


def validate(genome: np.ndarray | list[int], task_count: int, vehicles: int) -> bool:
    values = np.asarray(genome, dtype=int)
    return (
        len(values) == task_count + vehicles - 1
        and int(np.count_nonzero(values == 0)) == vehicles - 1
        and sorted(values[values != 0].tolist()) == list(range(1, task_count + 1))
    )


def repair(genome: np.ndarray | list[int], task_count: int, vehicles: int, rng: np.random.Generator) -> np.ndarray:
    """Return a legal permutation with exactly vehicles - 1 separators."""
    values = [int(value) for value in np.asarray(genome, dtype=int) if value != 0]
    seen: set[int] = set()
    tasks = []
    for value in values:
        if 1 <= value <= task_count and value not in seen:
            tasks.append(value)
            seen.add(value)
    tasks.extend(task for task in range(1, task_count + 1) if task not in seen)
    rng.shuffle(tasks)
    positions = np.sort(rng.choice(task_count + vehicles - 1, size=vehicles - 1, replace=False))
    result: list[int] = []
    task_index = 0
    separator_positions = set(positions.tolist())
    for index in range(task_count + vehicles - 1):
        if index in separator_positions:
            result.append(0)
        else:
            result.append(tasks[task_index])
            task_index += 1
    return np.asarray(result, dtype=int)


def random_genome(task_count: int, vehicles: int, rng: np.random.Generator) -> np.ndarray:
    tasks = rng.permutation(np.arange(1, task_count + 1))
    separator_positions = set(rng.choice(task_count + vehicles - 1, size=vehicles - 1, replace=False).tolist())
    genome: list[int] = []
    task_index = 0
    for position in range(task_count + vehicles - 1):
        if position in separator_positions:
            genome.append(0)
        else:
            genome.append(int(tasks[task_index]))
            task_index += 1
    return np.asarray(genome, dtype=int)
