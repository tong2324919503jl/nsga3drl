from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from ..data.schema import DynamicEvent, MTSPInstance
from ..problem.encoding import encode
from .insertion import DynamicInsertionPlanner


@dataclass
class DynamicExecutionResult:
    routes: list[list[int]]
    instance: MTSPInstance
    completed_initial_tasks: int
    event_log: list[dict]
    elapsed_seconds: float
    simulated_time_seconds: float


def execute_routes(
    instance: MTSPInstance,
    initial_routes: list[list[int]],
    k_neighbors: int = 8,
) -> DynamicExecutionResult:
    """Execute a fixed initial plan; dynamic points never call NSGA-III/PPO."""
    started = time.perf_counter()
    routes = [list(route) for route in initial_routes]
    completed_routes = [[] for _ in routes]
    positions = instance.vehicle_starts.copy()
    clocks = np.zeros(len(routes), dtype=float)
    visited: set[int] = set()
    expected_initial = set(int(task_id) for task_id in instance.task_ids)
    supplied_initial = [task_id for route in routes for task_id in route]
    if set(supplied_initial) != expected_initial or len(supplied_initial) != len(expected_initial):
        raise ValueError("initial_routes must contain every initial task exactly once")
    current_instance = instance.without_events()
    coordinates = {int(task_id): coord for task_id, coord in zip(current_instance.task_ids, current_instance.task_coordinates)}
    events = sorted(instance.events, key=lambda event: (event.arrival_completed_tasks, event.task_id))
    event_index = 0
    completed = 0
    simulated_time = 0.0
    event_log: list[dict] = []

    while any(routes) or event_index < len(events):
        # A batch arrives after the global completion counter reaches 10, 20, ...
        while event_index < len(events) and completed >= events[event_index].arrival_completed_tasks:
            event = events[event_index]
            event_started = time.perf_counter()
            planner = DynamicInsertionPlanner(current_instance, k_neighbors, positions)
            routes, decision = planner.insert(routes, event.task_id, event.coordinate)
            current_instance = current_instance.with_event_task(event)
            coordinates[event.task_id] = np.asarray(event.coordinate, dtype=float)
            event_log.append({
                "completed_tasks": completed,
                "simulated_time_seconds": simulated_time,
                "task_id": event.task_id,
                "vehicle": decision.vehicle,
                "position": decision.position,
                "distance_delta": decision.distance_delta,
                "local_search_improved": decision.local_search_improved,
                "elapsed_seconds": time.perf_counter() - event_started,
            })
            event_index += 1

        next_arrivals = []
        for vehicle, route in enumerate(routes):
            if route:
                task_id = route[0]
                travel_time = float(np.linalg.norm(coordinates[task_id] - positions[vehicle]))
                next_arrivals.append((clocks[vehicle] + travel_time, vehicle, task_id))
        if not next_arrivals:
            if event_index < len(events):
                raise ValueError(
                    "dynamic event arrival_completed_tasks exceeds the number of executable initial tasks"
                )
            break
        # Equal-speed vehicles run concurrently. The next global completion is
        # the smallest projected arrival time; vehicle id resolves exact ties.
        arrival_time, vehicle, task_id = min(next_arrivals, key=lambda item: (item[0], item[1]))
        if task_id in visited:
            raise RuntimeError(f"task {task_id} would be visited more than once")
        simulated_time = max(simulated_time, float(arrival_time))
        # Keep each vehicle's own clock.  Vehicles move in parallel, so a
        # completion by another vehicle must not artificially delay this one.
        clocks[vehicle] = float(arrival_time)
        routes[vehicle].pop(0)
        positions[vehicle] = coordinates[task_id]
        completed_routes[vehicle].append(task_id)
        visited.add(task_id)
        completed += 1

    # Keep the complete executed order for final objective calculation; the
    # mutable `routes` list only contains work that was still pending.
    final_routes = [completed_routes[index] + routes[index] for index in range(len(routes))]
    return DynamicExecutionResult(
        final_routes,
        current_instance,
        min(completed, instance.task_count),
        event_log,
        time.perf_counter() - started,
        simulated_time,
    )
