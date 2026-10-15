"""Heuristic route seeds and lightweight local search for open MTSP."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

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


def _two_opt_local(routes: list[list[int]], instance: MTSPInstance) -> list[list[int]]:
    """Bring every open vehicle route to a 2-opt local optimum efficiently."""
    coordinates = {int(task_id): point for task_id, point in zip(instance.task_ids, instance.task_coordinates)}
    result = [list(route) for route in routes]
    for vehicle, route in enumerate(result):
        improved = True
        while improved:
            improved = False
            for start in range(len(route) - 1):
                before = instance.vehicle_starts[vehicle] if start == 0 else coordinates[route[start - 1]]
                for end in range(start + 1, len(route)):
                    after = None if end == len(route) - 1 else coordinates[route[end + 1]]
                    old = float(np.linalg.norm(before - coordinates[route[start]]))
                    new = float(np.linalg.norm(before - coordinates[route[end]]))
                    if after is not None:
                        old += float(np.linalg.norm(coordinates[route[end]] - after))
                        new += float(np.linalg.norm(coordinates[route[start]] - after))
                    if new + 1e-12 < old:
                        route[start:end + 1] = reversed(route[start:end + 1])
                        improved = True
                        break
                if improved:
                    break
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


def sweep_routes(instance: MTSPInstance, angle_offset: float = 0.0) -> list[list[int]]:
    center = instance.task_coordinates.mean(axis=0)
    angles = (np.arctan2(instance.task_coordinates[:, 1] - center[1], instance.task_coordinates[:, 0] - center[0]) + angle_offset) % (2 * np.pi)
    ordered_ids = instance.task_ids[np.argsort(angles)].tolist()
    capacities = _capacities(instance.task_count, instance.vehicle_count)
    routes = []
    cursor = 0
    for capacity in capacities:
        routes.append([int(task_id) for task_id in ordered_ids[cursor:cursor + capacity]])
        cursor += capacity
    routes = _nearest_neighbor(routes, instance)
    # Route segments are spatially fixed; assign them to the closest starts.
    segment_centers = np.asarray([
        np.mean([instance.task_coordinates[np.where(instance.task_ids == task_id)[0][0]] for task_id in route], axis=0)
        for route in routes
    ])
    cost = np.linalg.norm(instance.vehicle_starts[:, None, :] - segment_centers[None, :, :], axis=2)
    vehicles, segments = linear_sum_assignment(cost)
    assigned = [[] for _ in range(instance.vehicle_count)]
    for vehicle, segment in zip(vehicles, segments):
        assigned[int(vehicle)] = routes[int(segment)]
    return _nearest_neighbor(assigned, instance)


def clustered_routes(instance: MTSPInstance, rng: np.random.Generator, iterations: int = 12) -> list[list[int]]:
    """Multi-start k-means partition followed by start-aware cluster assignment."""
    coordinates = instance.task_coordinates
    cluster_count = instance.vehicle_count
    centers = coordinates[rng.choice(len(coordinates), size=cluster_count, replace=False)].copy()
    labels = np.zeros(len(coordinates), dtype=int)
    for _ in range(iterations):
        distances = np.linalg.norm(coordinates[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        for cluster in range(cluster_count):
            members = coordinates[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
            else:
                centers[cluster] = coordinates[int(rng.integers(len(coordinates)))]
    routes = [[int(task_id) for task_id in instance.task_ids[labels == cluster]] for cluster in range(cluster_count)]
    # Repair rare empty clusters before assigning every cluster to one vehicle.
    for cluster, route in enumerate(routes):
        if not route:
            donor = max(range(cluster_count), key=lambda item: len(routes[item]))
            route.append(routes[donor].pop())
            centers[cluster] = coordinates[np.where(instance.task_ids == route[0])[0][0]]
    cost = np.linalg.norm(instance.vehicle_starts[:, None, :] - centers[None, :, :], axis=2)
    vehicles, clusters = linear_sum_assignment(cost)
    assigned = [[] for _ in range(instance.vehicle_count)]
    for vehicle, cluster in zip(vehicles, clusters):
        assigned[int(vehicle)] = routes[int(cluster)]
    return _nearest_neighbor(assigned, instance)


def ensure_nonempty_routes(routes: list[list[int]]) -> list[list[int]]:
    """Move one task from the largest route to each empty vehicle route."""
    repaired = [list(route) for route in routes]
    for vehicle, route in enumerate(repaired):
        if not route:
            donor = max(range(len(repaired)), key=lambda index: len(repaired[index]))
            if len(repaired[donor]) <= 1:
                raise ValueError("cannot create a non-empty route for every vehicle")
            route.append(repaired[donor].pop())
    return repaired


def improve_routes(
    routes: list[list[int]],
    instance: MTSPInstance,
    preference: tuple[float, float] | None = None,
    max_passes: int = 1,
    max_evaluations: int = 250,
) -> list[list[int]]:
    """Apply bounded first-improvement 2-opt, relocate, swap and boundary moves."""
    preference = preference or instance.preference or (0.5, 0.5)
    current = _two_opt_local(ensure_nonempty_routes(routes), instance)
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
            if len(source_route) <= 1:
                continue
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
        if index % 3 == 0:
            routes = clustered_routes(instance, rng)
        elif index % 3 == 1:
            routes = sweep_routes(instance, angle_offset=float(rng.uniform(0, 2 * np.pi)))
        else:
            routes = nearest_start_routes(instance)
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
        routes = improve_routes(routes, instance, max_passes=1, max_evaluations=180)
        genomes.append(encode(routes, instance.vehicle_count))
    while len(genomes) < population_size:
        routes = decode(random_genome(instance.task_count, instance.vehicle_count, rng), instance.vehicle_count)
        genomes.append(encode(ensure_nonempty_routes(routes), instance.vehicle_count))
    return np.vstack(genomes)
