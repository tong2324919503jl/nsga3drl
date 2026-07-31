from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

from ..data.schema import MTSPInstance


@dataclass(frozen=True)
class InsertionDecision:
    vehicle: int
    position: int
    detour_scores: np.ndarray
    balance_scores: np.ndarray
    assignment_scores: np.ndarray
    distance_delta: float
    local_search_improved: bool


class DynamicInsertionPlanner:
    def __init__(self, instance: MTSPInstance, k_neighbors: int = 8):
        self.instance = instance
        self.k_neighbors = k_neighbors
        self._coords = {int(task_id): coord for task_id, coord in zip(instance.task_ids, instance.task_coordinates)}

    def _route_length(self, vehicle: int, route: list[int]) -> float:
        current = self.instance.vehicle_starts[vehicle]
        length = 0.0
        for task_id in route:
            target = self._coords[task_id]
            length += float(np.linalg.norm(target - current))
            current = target
        return length

    def _two_opt(self, vehicle: int, route: list[int]) -> tuple[list[int], bool]:
        """Use first-improvement 2-opt on only the route changed by insertion."""
        best = list(route)
        best_length = self._route_length(vehicle, best)
        for start in range(len(best) - 1):
            for end in range(start + 2, len(best) + 1):
                candidate = best[:start] + best[start:end][::-1] + best[end:]
                candidate_length = self._route_length(vehicle, candidate)
                if candidate_length + 1e-12 < best_length:
                    return candidate, True
        return best, False

    def insert(self, routes: list[list[int]], new_task_id: int, coordinate: tuple[float, float]) -> tuple[list[list[int]], InsertionDecision]:
        if new_task_id in self._coords:
            raise ValueError(f"duplicate task id {new_task_id}")
        if len(routes) != self.instance.vehicle_count:
            raise ValueError("route count does not match vehicles")
        assigned = [task_id for route in routes for task_id in route]
        if len(assigned) != len(set(assigned)) or any(task_id not in self._coords for task_id in assigned):
            raise ValueError("routes must contain unique known task ids")
        self._coords[new_task_id] = np.asarray(coordinate, dtype=float)
        owner = {task_id: vehicle for vehicle, route in enumerate(routes) for task_id in route}
        pending = list(owner)
        detour = np.zeros(len(routes), dtype=float)
        if pending:
            tree = KDTree(np.vstack([self._coords[task_id] for task_id in pending]))
            count = min(self.k_neighbors, len(pending))
            distances, indices = tree.query(self._coords[new_task_id], k=count)
            for distance, index in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
                detour[owner[pending[int(index)]]] += 1.0 / (1.0 + float(distance))
        else:
            detour.fill(1.0)
        detour /= max(detour.max(), 1e-12)
        lengths = np.asarray([self._route_length(vehicle, route) for vehicle, route in enumerate(routes)])
        balance = 1.0 - (lengths - lengths.min()) / max(lengths.max() - lengths.min(), 1e-12)
        scores = detour * balance
        candidates = np.flatnonzero(np.isclose(scores, scores.max()))
        vehicle = int(candidates[np.argmin(lengths[candidates])])
        route = routes[vehicle]
        original_length = self._route_length(vehicle, route)
        best_position, best_delta = 0, float("inf")
        for position in range(len(route) + 1):
            before = self.instance.vehicle_starts[vehicle] if position == 0 else self._coords[route[position - 1]]
            after = None if position == len(route) else self._coords[route[position]]
            delta = float(np.linalg.norm(before - self._coords[new_task_id]))
            if after is not None:
                delta += float(np.linalg.norm(self._coords[new_task_id] - after) - np.linalg.norm(before - after))
            if delta < best_delta:
                best_position, best_delta = position, delta
        updated = [list(route_) for route_ in routes]
        updated[vehicle].insert(best_position, new_task_id)
        updated[vehicle], locally_improved = self._two_opt(vehicle, updated[vehicle])
        final_delta = self._route_length(vehicle, updated[vehicle]) - original_length
        return updated, InsertionDecision(
            vehicle, best_position, detour, balance, scores, final_delta, locally_improved
        )
