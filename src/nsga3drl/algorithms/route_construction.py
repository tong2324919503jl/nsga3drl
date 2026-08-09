"""Heuristic route seeds and lightweight local search for open MTSP."""

from __future__ import annotations

import numpy as np

from ..data.schema import MTSPInstance
from ..problem.encoding import decode, encode, random_genome


def _route_lengths(routes: list[list[int]], instance: MTSPInstance) -> np.ndarray:
    coordinates = {int(task_id): point for task_id, point in zip(instance.task_ids, instance.task_coordinates)}
    lengths = np.zeros(instance.vehicle_count, dtype=float)
    for vehicle, route in enumerate(routes):
        current = instance.vehicle_starts[vehicle]
        for task_id in route:
            target = coordinates[int(task_id)]
            lengths[vehicle] += float(np.linalg.norm(target - current))
            current = target
    return lengths


def _objective(routes: list[list[int]], instance: MTSPInstance) -> np.ndarray:
    lengths = _route_lengths(routes, instance)
    return np.asarray([lengths.sum(), lengths.var()], dtype=float)


def _score(objective: np.ndarray, reference: np.ndarray, preference: tuple[float, float]) -> float:
    # Relative scaling keeps distance and variance comparable during local search.
    return float(
        preference[0] * objective[0] / max(reference[0], 1e-9)
        + preference[1] * objective[1] / max(reference[1], 1e-9)
    )


def _nearest_neighbor(routes: list[list[int]], instance: MTSPInstance) -> list[list[int]]:
    coordinates = {int(task_id): point for task_id, point in zip(instance.task_ids, instance.task_coordinates)}
    result = []
    for vehicle, assigned in enumerate(routes):
        pending = set(assigned)
        current = instance.vehicle_starts[vehicle]
        ordered = []
        while pending:
            task_id = min(pending, key=lambda item: float(np.linalg.norm(coordinates[item] - current)))
            ordered.append(int(task_id))
            pending.remove(task_id)
            current = coordinates[task_id]
        result.append(ordered)
    return result


def _capacities(task_count: int, vehicle_count: int) -> list[int]:
    base, remainder = divmod(task_count, vehicle_count)
    return [base + (index < remainder) for index in range(vehicle_count)]


def nearest_start_routes(instance: MTSPInstance) -> list[list[int]]:
    """Assign nearby tasks to vehicles with balanced capacity, then order them greedily."""
    capacities = _capacities(instance.task_count, instance.vehicle_count)
    remaining = set(int(task_id) for task_id in instance.task_ids)
    coordinates = {int(task_id): point for task_id, point in zip(instance.task_ids, instance.task_coordinates)}
    assignments = [[] for _ in range(instance.vehicle_count)]
    distance_order = {
        task_id: np.argsort([np.linalg.norm(coordinates[task_id] - start) for start in instance.vehicle_starts])
        for task_id in remaining
    }
    # Allocate the most isolated tasks first so they are not stranded on a full route.
    tasks = sorted(remaining, key=lambda task: np.partition(
        [np.linalg.norm(coordinates[task] - start) for start in instance.vehicle_starts], 1
    )[1] - np.partition(
        [np.linalg.norm(coordinates[task] - start) for start in instance.vehicle_starts], 1
    )[0], reverse=True)
    for task_id in tasks:
        for vehicle in distance_order[task_id]:
            if len(assignments[int(vehicle)]) < capacities[int(vehicle)]:
                assignments[int(vehicle)].append(task_id)
                break
    return _nearest_neighbor(assignments, instance)


def sweep_routes(instance: MTSPInstance) -> list[list[int]]:
    center = instance.task_coordinates.mean(axis=0)
    angles = np.arctan2(instance.task_coordinates[:, 1] - center[1], instance.task_coordinates[:, 0] - center[0])
    ordered_ids = instance.task_ids[np.argsort(angles)].tolist()
    capacities = _capacities(instance.task_count, instance.vehicle_count)
    routes = []
    cursor = 0
    for capacity in capacities:
        routes.append([int(task_id) for task_id in ordered_ids[cursor:cursor + capacity]])
        cursor += capacity
    return _nearest_neighbor(routes, instance)


def improve_routes(
    routes: list[list[int]],
    instance: MTSPInstance,
    preference: tuple[float, float] | None = None,
    max_passes: int = 1,
    max_evaluations: int = 250,
) -> list[list[int]]:
    """Apply bounded first-improvement 2-opt, relocate, swap and boundary moves."""
    preference = preference or instance.preference or (0.5, 0.5)
    current = [list(route) for route in routes]
    current_objective = _objective(current, instance)
    evaluations = 0

    def consider(candidate: list[list[int]]) -> bool:
        nonlocal current, current_objective, evaluations
        if evaluations >= max_evaluations:
            return False
        evaluations += 1
        objective = _objective(candidate, instance)
        if _score(objective, current_objective, preference) + 1e-12 < _score(current_objective, current_objective, preference):
            current, current_objective = candidate, objective
            return True
        return False

    for _ in range(max_passes):
        changed = False
        # Intra-route 2-opt.
        for vehicle, route in enumerate(current):
            for start in range(len(route) - 1):
                for end in range(start + 2, len(route) + 1):
                    candidate = [list(item) for item in current]
                    candidate[vehicle][start:end] = reversed(candidate[vehicle][start:end])
                    if consider(candidate):
                        changed = True
                        break
                if changed or evaluations >= max_evaluations:
                    break
            if changed or evaluations >= max_evaluations:
                break
        if evaluations >= max_evaluations:
            break
        # Cross-vehicle relocate.
        for source, source_route in enumerate(current):
            for position, task_id in enumerate(source_route):
                for target, target_route in enumerate(current):
                    if source == target:
                        continue
                    for insert_at in range(len(target_route) + 1):
                        candidate = [list(item) for item in current]
                        candidate[source].pop(position)
                        candidate[target].insert(insert_at, task_id)
                        if consider(candidate):
                            changed = True
                            break
                    if changed or evaluations >= max_evaluations:
                        break
                if changed or evaluations >= max_evaluations:
                    break
            if changed or evaluations >= max_evaluations:
                break
        if evaluations >= max_evaluations:
            break
        # Cross-vehicle swap.
        for left in range(len(current)):
            for right in range(left + 1, len(current)):
                for left_pos in range(len(current[left])):
                    for right_pos in range(len(current[right])):
                        candidate = [list(item) for item in current]
                        candidate[left][left_pos], candidate[right][right_pos] = candidate[right][right_pos], candidate[left][left_pos]
                        if consider(candidate):
                            changed = True
                            break
                    if changed or evaluations >= max_evaluations:
                        break
                if changed or evaluations >= max_evaluations:
                    break
            if changed or evaluations >= max_evaluations:
                break
        if not changed or evaluations >= max_evaluations:
            break
    return current


def initial_population(instance: MTSPInstance, population_size: int, rng: np.random.Generator) -> np.ndarray:
    """Mix constructive seeds, local improvements, and random diversity."""
    genomes = []
    heuristic_count = int(round(population_size * 0.85))
    for index in range(heuristic_count):
        routes = nearest_start_routes(instance) if index % 3 else sweep_routes(instance)
        # Preserve a useful seed while injecting enough diversity for NSGA-III.
        for _ in range(1 + index % 3):
            nonempty = [vehicle for vehicle, route in enumerate(routes) if route]
            source = int(rng.choice(nonempty))
            if rng.random() < 0.5 or len(nonempty) == 1:
                if len(routes[source]) >= 2:
                    left, right = rng.choice(len(routes[source]), size=2, replace=False)
                    routes[source][int(left)], routes[source][int(right)] = routes[source][int(right)], routes[source][int(left)]
            else:
                target = int(rng.choice(nonempty))
                task = routes[source].pop(int(rng.integers(len(routes[source]))))
                routes[target].insert(int(rng.integers(len(routes[target]) + 1)), task)
        routes = improve_routes(routes, instance, max_passes=1, max_evaluations=120)
        genomes.append(encode(routes, instance.vehicle_count))
    while len(genomes) < population_size:
        genomes.append(random_genome(instance.task_count, instance.vehicle_count, rng))
    return np.vstack(genomes)
