from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from ..data.schema import MTSPInstance


HELD_KARP_MAX_TASKS = 18
BRANCH_AND_BOUND_MAX_TASKS = 30
BRANCH_AND_BOUND_NODE_LIMIT = 50_000


@dataclass(frozen=True)
class InsertionDecision:
    vehicle: int
    position: int
    detour_scores: np.ndarray
    balance_scores: np.ndarray
    assignment_scores: np.ndarray
    distance_delta: float
    local_search_improved: bool
    candidate_planning_seconds: float
    vehicle_planning_seconds: np.ndarray
    solver_method: str
    candidate_task_counts: np.ndarray
    branch_bound_nodes: np.ndarray
    fallback_reason: str | None
    strategy: str
    scoring_distance_deltas: np.ndarray
    greedy_insertion_positions: np.ndarray
    vehicle_scoring_seconds: np.ndarray
    selected_tsp_seconds: float


class DynamicInsertionPlanner:
    """Assign a new task by comparing per-vehicle open-TSP route expansions."""

    def __init__(
        self,
        instance: MTSPInstance,
        vehicle_positions: np.ndarray | None = None,
        preference: tuple[float, float] | None = None,
    ):
        self.instance = instance
        self.vehicle_positions = (
            np.asarray(vehicle_positions, dtype=float)
            if vehicle_positions is not None else instance.vehicle_starts
        )
        if self.vehicle_positions.shape != instance.vehicle_starts.shape:
            raise ValueError("vehicle_positions must have shape (vehicle_count, 2)")
        self.preference = preference if preference is not None else instance.preference
        if self.preference is None:
            self.preference = (0.5, 0.5)
        if (
            len(self.preference) != 2
            or min(self.preference) < 0
            or not np.isclose(sum(self.preference), 1.0)
        ):
            raise ValueError("preference must contain two non-negative weights summing to 1")
        self._coords = {
            int(task_id): np.asarray(coord, dtype=float)
            for task_id, coord in zip(instance.task_ids, instance.task_coordinates)
        }

    def _route_length(self, vehicle: int, route: list[int]) -> float:
        current = self.vehicle_positions[vehicle]
        length = 0.0
        for task_id in route:
            target = self._coords[task_id]
            length += float(np.linalg.norm(target - current))
            current = target
        return length

    def _distance_data(self, vehicle: int, tasks: list[int]) -> tuple[np.ndarray, np.ndarray]:
        coordinates = np.vstack([self._coords[task_id] for task_id in tasks])
        start = self.vehicle_positions[vehicle]
        start_distances = np.linalg.norm(coordinates - start, axis=1)
        distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
        return start_distances, distances

    def _held_karp(self, vehicle: int, tasks: list[int]) -> tuple[list[int], float]:
        """Exact Held-Karp solution for a small open TSP instance."""
        count = len(tasks)
        start_distances, distances = self._distance_data(vehicle, tasks)
        state_count = 1 << count
        costs = np.full((state_count, count), np.inf, dtype=float)
        parents = np.full((state_count, count), -1, dtype=np.int16)
        for last in range(count):
            costs[1 << last, last] = start_distances[last]

        for mask in range(1, state_count):
            for last in range(count):
                bit = 1 << last
                if not mask & bit or mask == bit:
                    continue
                previous_mask = mask ^ bit
                previous_costs = costs[previous_mask] + distances[:, last]
                previous_last = int(np.argmin(previous_costs))
                costs[mask, last] = previous_costs[previous_last]
                parents[mask, last] = previous_last

        full_mask = state_count - 1
        last = int(np.argmin(costs[full_mask]))
        best_length = float(costs[full_mask, last])
        mask = full_mask
        reverse_order: list[int] = []
        while mask:
            reverse_order.append(last)
            previous_last = int(parents[mask, last])
            mask ^= 1 << last
            if mask:
                last = previous_last
        return [tasks[index] for index in reversed(reverse_order)], best_length

    def _heuristic(self, vehicle: int, tasks: list[int], current_route: list[int]) -> tuple[list[int], bool]:
        """Insertion + nearest-neighbor seeds, each improved by best-improvement 2-opt."""
        best_position = 0
        best_insertion_length = float("inf")
        for position in range(len(current_route) + 1):
            candidate = current_route[:position] + [tasks[-1]] + current_route[position:]
            length = self._route_length(vehicle, candidate)
            if length < best_insertion_length:
                best_position, best_insertion_length = position, length
        insertion_seed = current_route[:best_position] + [tasks[-1]] + current_route[best_position:]

        remaining = set(tasks)
        nearest_seed: list[int] = []
        current = self.vehicle_positions[vehicle]
        while remaining:
            task_id = min(
                remaining,
                key=lambda item: (float(np.linalg.norm(self._coords[item] - current)), item),
            )
            nearest_seed.append(task_id)
            remaining.remove(task_id)
            current = self._coords[task_id]

        best_route: list[int] | None = None
        best_length = float("inf")
        improved_any = False
        for seed in (insertion_seed, nearest_seed):
            candidate = list(seed)
            while len(candidate) > 1:
                base_length = self._route_length(vehicle, candidate)
                move: tuple[int, int] | None = None
                move_length = base_length
                for left in range(len(candidate) - 1):
                    for right in range(left + 1, len(candidate)):
                        changed = candidate[:left] + candidate[left:right + 1][::-1] + candidate[right + 1:]
                        length = self._route_length(vehicle, changed)
                        if length + 1e-12 < move_length:
                            move, move_length = (left, right), length
                if move is None:
                    break
                left, right = move
                candidate[left:right + 1] = reversed(candidate[left:right + 1])
                improved_any = True
            length = self._route_length(vehicle, candidate)
            if length < best_length:
                best_route, best_length = candidate, length
        assert best_route is not None
        return best_route, improved_any

    @staticmethod
    def _mst_cost(mask: int, distances: np.ndarray, cache: dict[int, float]) -> float:
        """MST lower bound over the nodes represented by a bit mask."""
        if mask in cache:
            return cache[mask]
        nodes: list[int] = []
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            nodes.append(bit.bit_length() - 1)
            remaining ^= bit
        if len(nodes) < 2:
            cache[mask] = 0.0
            return 0.0
        in_tree = {nodes[0]}
        nearest = {node: float(distances[nodes[0], node]) for node in nodes[1:]}
        total = 0.0
        while len(in_tree) < len(nodes):
            node = min(nearest, key=nearest.__getitem__)
            total += nearest.pop(node)
            in_tree.add(node)
            for other in nearest:
                nearest[other] = min(nearest[other], float(distances[node, other]))
        cache[mask] = total
        return total

    def _branch_and_bound(
        self,
        vehicle: int,
        tasks: list[int],
        incumbent_route: list[int],
        node_limit: int = BRANCH_AND_BOUND_NODE_LIMIT,
    ) -> tuple[list[int], float, int, bool]:
        """Exact open TSP search with an MST lower bound and a node budget."""
        start_distances, distances = self._distance_data(vehicle, tasks)
        count = len(tasks)
        full_mask = (1 << count) - 1
        index_of = {task_id: index for index, task_id in enumerate(tasks)}
        incumbent_indices = [index_of[task_id] for task_id in incumbent_route]
        best_length = self._route_length(vehicle, incumbent_route)
        best_order = list(incumbent_indices)
        path: list[int] = []
        mst_cache: dict[int, float] = {0: 0.0}
        expanded = 0
        exhausted = True

        def search(mask: int, last: int, distance_so_far: float) -> None:
            nonlocal best_length, best_order, expanded, exhausted
            if not exhausted:
                return
            expanded += 1
            if expanded > node_limit:
                exhausted = False
                return
            if mask == full_mask:
                if distance_so_far + 1e-12 < best_length:
                    best_length = distance_so_far
                    best_order = list(path)
                return

            remaining_mask = full_mask ^ mask
            if last < 0:
                connection = min(float(start_distances[node]) for node in range(count) if remaining_mask & (1 << node))
            else:
                connection = min(float(distances[last, node]) for node in range(count) if remaining_mask & (1 << node))
            lower_bound = distance_so_far + connection + self._mst_cost(remaining_mask, distances, mst_cache)
            if lower_bound >= best_length - 1e-12:
                return

            options = [node for node in range(count) if remaining_mask & (1 << node)]
            if last < 0:
                options.sort(key=lambda node: (float(start_distances[node]), tasks[node]))
            else:
                options.sort(key=lambda node: (float(distances[last, node]), tasks[node]))
            for node in options:
                edge = float(start_distances[node]) if last < 0 else float(distances[last, node])
                path.append(node)
                search(mask | (1 << node), node, distance_so_far + edge)
                path.pop()
                if not exhausted:
                    return

        search(0, -1, 0.0)
        return [tasks[index] for index in best_order], best_length, expanded, exhausted

    def _replan_vehicle(
        self,
        vehicle: int,
        route: list[int],
        new_task_id: int,
        solver: str,
    ) -> tuple[list[int], float, int, bool, int, bool]:
        tasks = [*route, new_task_id]
        original_length = self._route_length(vehicle, route)
        if solver == "held_karp":
            best_route, best_length = self._held_karp(vehicle, tasks)
            nodes, exact = 0, True
            best_insertion_length = min(
                self._route_length(vehicle, route[:position] + [new_task_id] + route[position:])
                for position in range(len(route) + 1)
            )
            improved = best_length + 1e-12 < best_insertion_length
        elif solver == "branch_and_bound":
            incumbent, improved = self._heuristic(vehicle, tasks, route)
            best_route, best_length, nodes, exact = self._branch_and_bound(vehicle, tasks, incumbent)
        else:
            best_route, improved = self._heuristic(vehicle, tasks, route)
            best_length = self._route_length(vehicle, best_route)
            nodes, exact = 0, False

        return (
            best_route,
            best_length - original_length,
            best_route.index(new_task_id),
            improved,
            nodes,
            exact,
        )

    def insert(
        self,
        routes: list[list[int]],
        new_task_id: int,
        coordinate: tuple[float, float],
    ) -> tuple[list[list[int]], InsertionDecision]:
        if new_task_id in self._coords:
            raise ValueError(f"duplicate task id {new_task_id}")
        if len(routes) != self.instance.vehicle_count:
            raise ValueError("route count does not match vehicles")
        assigned = [task_id for route in routes for task_id in route]
        if len(assigned) != len(set(assigned)) or any(task_id not in self._coords for task_id in assigned):
            raise ValueError("routes must contain unique known task ids")

        self._coords[new_task_id] = np.asarray(coordinate, dtype=float)
        candidate_task_counts = np.asarray([len(route) + 1 for route in routes], dtype=int)
        largest_candidate = int(candidate_task_counts.max(initial=0))
        if largest_candidate <= HELD_KARP_MAX_TASKS:
            solver = "held_karp"
        elif largest_candidate <= BRANCH_AND_BOUND_MAX_TASKS:
            solver = "branch_and_bound"
        else:
            solver = "heuristic"

        candidate_routes: list[list[int]] = []
        deltas = np.zeros(len(routes), dtype=float)
        local_improvements = np.zeros(len(routes), dtype=bool)
        positions = np.zeros(len(routes), dtype=int)
        vehicle_times = np.zeros(len(routes), dtype=float)
        branch_bound_nodes = np.zeros(len(routes), dtype=int)
        exact_flags = np.zeros(len(routes), dtype=bool)
        planning_started = time.perf_counter()
        for vehicle, route in enumerate(routes):
            vehicle_started = time.perf_counter()
            (
                replanned,
                deltas[vehicle],
                positions[vehicle],
                local_improvements[vehicle],
                branch_bound_nodes[vehicle],
                exact_flags[vehicle],
            ) = self._replan_vehicle(
                vehicle, route, new_task_id, solver
            )
            candidate_routes.append(replanned)
            vehicle_times[vehicle] = time.perf_counter() - vehicle_started

        fallback_reason = None
        if solver == "branch_and_bound" and not np.all(exact_flags):
            fallback_reason = (
                f"at least one vehicle reached the {BRANCH_AND_BOUND_NODE_LIMIT}-node branch-and-bound budget; "
                "all vehicle candidates were rescored with the heuristic for consistency"
            )
            solver = "heuristic_fallback"
            for vehicle, route in enumerate(routes):
                vehicle_started = time.perf_counter()
                (
                    candidate_routes[vehicle],
                    local_improvements[vehicle],
                ) = self._heuristic(vehicle, [*route, new_task_id], route)
                vehicle_times[vehicle] += time.perf_counter() - vehicle_started
                deltas[vehicle] = self._route_length(vehicle, candidate_routes[vehicle]) - self._route_length(vehicle, route)
                positions[vehicle] = candidate_routes[vehicle].index(new_task_id)

        # Lower marginal distance means a better detour score; negative deltas remain valid.
        delta_span = float(deltas.max() - deltas.min())
        detour = np.ones(len(routes), dtype=float) if delta_span <= 1e-12 else (deltas.max() - deltas) / delta_span

        lengths = np.asarray([self._route_length(vehicle, route) for vehicle, route in enumerate(routes)])
        balance = 1.0 - (lengths - lengths.min()) / max(lengths.max() - lengths.min(), 1e-12)
        efficiency_weight, fairness_weight = self.preference
        scores = detour * efficiency_weight + balance * fairness_weight
        candidates = np.flatnonzero(np.isclose(scores, scores.max()))
        # Deterministic tie-break: less added distance, then shorter current route, then vehicle id.
        vehicle = int(min(candidates, key=lambda item: (deltas[item], lengths[item], int(item))))
        updated = [list(route) for route in routes]
        updated[vehicle] = candidate_routes[vehicle]
        return updated, InsertionDecision(
            vehicle=vehicle,
            position=int(positions[vehicle]),
            detour_scores=detour,
            balance_scores=balance,
            assignment_scores=scores,
            distance_delta=float(deltas[vehicle]),
            local_search_improved=bool(local_improvements[vehicle]),
            candidate_planning_seconds=time.perf_counter() - planning_started,
            vehicle_planning_seconds=vehicle_times,
            solver_method=solver,
            candidate_task_counts=candidate_task_counts,
            branch_bound_nodes=branch_bound_nodes,
            fallback_reason=fallback_reason,
            strategy="tsp_scored",
            scoring_distance_deltas=deltas.copy(),
            greedy_insertion_positions=np.full(len(routes), -1, dtype=int),
            vehicle_scoring_seconds=np.zeros(len(routes), dtype=float),
            selected_tsp_seconds=float(vehicle_times[vehicle]),
        )


class GreedyScoreDynamicInsertionPlanner(DynamicInsertionPlanner):
    """Score all vehicles by cheapest insertion, then TSP-replan only the winner."""

    def insert(
        self,
        routes: list[list[int]],
        new_task_id: int,
        coordinate: tuple[float, float],
    ) -> tuple[list[list[int]], InsertionDecision]:
        if new_task_id in self._coords:
            raise ValueError(f"duplicate task id {new_task_id}")
        if len(routes) != self.instance.vehicle_count:
            raise ValueError("route count does not match vehicles")
        assigned = [task_id for route in routes for task_id in route]
        if len(assigned) != len(set(assigned)) or any(task_id not in self._coords for task_id in assigned):
            raise ValueError("routes must contain unique known task ids")

        self._coords[new_task_id] = np.asarray(coordinate, dtype=float)
        total_started = time.perf_counter()
        lengths = np.asarray([
            self._route_length(vehicle, route)
            for vehicle, route in enumerate(routes)
        ])
        scoring_deltas = np.zeros(len(routes), dtype=float)
        insertion_positions = np.zeros(len(routes), dtype=int)
        vehicle_scoring_times = np.zeros(len(routes), dtype=float)

        # Greedily test each insertion position without solving a TSP per vehicle.
        for vehicle, route in enumerate(routes):
            vehicle_started = time.perf_counter()
            best_delta = float("inf")
            best_position = 0
            for position in range(len(route) + 1):
                candidate = route[:position] + [new_task_id] + route[position:]
                delta = self._route_length(vehicle, candidate) - lengths[vehicle]
                if delta < best_delta:
                    best_delta, best_position = delta, position
            scoring_deltas[vehicle] = best_delta
            insertion_positions[vehicle] = best_position
            vehicle_scoring_times[vehicle] = time.perf_counter() - vehicle_started
        # Normalize the greedy marginal distance across vehicles; lower is better.
        delta_span = float(scoring_deltas.max() - scoring_deltas.min())
        detour = (
            np.ones(len(routes), dtype=float)
            if delta_span <= 1e-12
            else (scoring_deltas.max() - scoring_deltas) / delta_span
        )
        balance = 1.0 - (lengths - lengths.min()) / max(lengths.max() - lengths.min(), 1e-12)
        efficiency_weight, fairness_weight = self.preference
        scores = detour * efficiency_weight + balance * fairness_weight
        candidates = np.flatnonzero(np.isclose(scores, scores.max()))
        vehicle = int(min(candidates, key=lambda item: (
            scoring_deltas[item], lengths[item], int(item)
        )))

        # Only the selected vehicle gets the more expensive three-tier TSP replan.
        candidate_task_counts = np.asarray([len(route) + 1 for route in routes], dtype=int)
        selected_count = int(candidate_task_counts[vehicle])
        if selected_count <= HELD_KARP_MAX_TASKS:
            solver = "held_karp"
        elif selected_count <= BRANCH_AND_BOUND_MAX_TASKS:
            solver = "branch_and_bound"
        else:
            solver = "heuristic"

        replan_started = time.perf_counter()
        (
            selected_route,
            selected_delta,
            selected_position,
            locally_improved,
            branch_nodes,
            exact,
        ) = self._replan_vehicle(vehicle, routes[vehicle], new_task_id, solver)
        selected_tsp_seconds = time.perf_counter() - replan_started
        branch_nodes_per_vehicle = np.zeros(len(routes), dtype=int)
        branch_nodes_per_vehicle[vehicle] = branch_nodes
        vehicle_planning_times = np.zeros(len(routes), dtype=float)
        vehicle_planning_times[vehicle] = selected_tsp_seconds

        fallback_reason = None
        if solver == "branch_and_bound" and not exact:
            fallback_reason = (
                f"selected vehicle reached the {BRANCH_AND_BOUND_NODE_LIMIT}-node branch-and-bound budget; "
                "its route was replanned with the heuristic"
            )
            solver = "heuristic_fallback"

        updated = [list(route) for route in routes]
        updated[vehicle] = selected_route
        return updated, InsertionDecision(
            vehicle=vehicle,
            position=int(selected_position),
            detour_scores=detour,
            balance_scores=balance,
            assignment_scores=scores,
            distance_delta=float(selected_delta),
            local_search_improved=bool(locally_improved),
            candidate_planning_seconds=time.perf_counter() - total_started,
            vehicle_planning_seconds=vehicle_planning_times,
            solver_method=solver,
            candidate_task_counts=candidate_task_counts,
            branch_bound_nodes=branch_nodes_per_vehicle,
            fallback_reason=fallback_reason,
            strategy="greedy_score_then_tsp",
            scoring_distance_deltas=scoring_deltas,
            greedy_insertion_positions=insertion_positions,
            vehicle_scoring_seconds=vehicle_scoring_times,
            selected_tsp_seconds=selected_tsp_seconds,
        )
