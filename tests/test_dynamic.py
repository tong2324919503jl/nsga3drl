from itertools import permutations

import numpy as np

from nsga3drl.data.schema import DynamicEvent, MTSPInstance
from nsga3drl.dynamic.execution import execute_routes
from nsga3drl.dynamic.insertion import (
    DynamicInsertionPlanner,
    GreedyScoreDynamicInsertionPlanner,
)


def test_new_task_is_inserted_once() -> None:
    instance = MTSPInstance("dynamic", [[0, 0], [10, 0]], [[1, 0], [8, 0]])
    routes, decision = DynamicInsertionPlanner(instance).insert([[1], [2]], 3, (1.5, 0.0))
    assert sum(route.count(3) for route in routes) == 1
    assert decision.vehicle == 0
    assert sorted(routes[0]) == [1, 3]
    assert decision.vehicle_planning_seconds.shape == (2,)
    assert decision.candidate_planning_seconds >= 0


def test_preference_weight_changes_dynamic_vehicle_assignment() -> None:
    instance = MTSPInstance(
        "weights", [[0, 0], [100, 0]], [[1, 0], [2, 0], [101, 0]],
    )
    routes = [[1, 2], [3]]
    _, efficiency_decision = DynamicInsertionPlanner(instance, preference=(1.0, 0.0)).insert(
        routes, 4, (1.5, 0.0)
    )
    _, fairness_decision = DynamicInsertionPlanner(instance, preference=(0.0, 1.0)).insert(
        routes, 4, (1.5, 0.0)
    )
    assert efficiency_decision.vehicle == 0
    assert fairness_decision.vehicle == 1
    assert efficiency_decision.assignment_scores[0] > efficiency_decision.assignment_scores[1]
    assert fairness_decision.assignment_scores[1] > fairness_decision.assignment_scores[0]


def test_negative_distance_delta_is_kept_and_replanned_route_is_used() -> None:
    instance = MTSPInstance(
        "negative-delta", [[0, 0], [100, 0]], [[10, 0], [1, 0], [101, 0]],
    )
    updated, decision = DynamicInsertionPlanner(instance, preference=(1.0, 0.0)).insert(
        [[1, 2], [3]], 4, (5, 0)
    )
    assert decision.vehicle == 0
    assert decision.distance_delta < 0
    assert updated[0] == [2, 4, 1]


def test_vehicle_replan_matches_brute_force_open_tsp_optimum() -> None:
    instance = MTSPInstance(
        "exact-tsp", [[0, 0]], [[7, 2], [1, 8], [5, 5], [9, 9]],
    )
    route = [1, 2, 3]
    updated, _ = DynamicInsertionPlanner(instance, preference=(1.0, 0.0)).insert(
        [route], 5, (9, 9)
    )
    coords = {int(task): point for task, point in zip(instance.task_ids, instance.task_coordinates)}
    coords[5] = np.asarray((9, 9), dtype=float)

    def open_length(order):
        points = [np.asarray((0, 0), dtype=float), *(coords[task] for task in order)]
        return sum(float(np.linalg.norm(right - left)) for left, right in zip(points, points[1:]))

    actual = open_length(updated[0])
    optimum = min(open_length(order) for order in permutations((1, 2, 3, 5)))
    assert np.isclose(actual, optimum)


def test_branch_and_bound_matches_brute_force() -> None:
    instance = MTSPInstance(
        "branch-bound", [[0, 0]], [[7, 2], [1, 8], [5, 5], [9, 9], [4, 1]],
    )
    planner = DynamicInsertionPlanner(instance)
    tasks = [1, 2, 3, 4, 5]
    incumbent, _ = planner._heuristic(0, tasks, tasks[:-1])
    order, length, _, exact = planner._branch_and_bound(0, tasks, incumbent)

    def open_length(route):
        return planner._route_length(0, route)

    optimum = min(open_length(route) for route in permutations(tasks))
    assert exact
    assert np.isclose(length, optimum)
    assert np.isclose(open_length(order), optimum)


def test_solver_tier_is_selected_for_all_vehicles_from_event_maximum() -> None:
    coords = [[float(index), 0.0] for index in range(1, 20)] + [[100.0, 0.0]]
    instance = MTSPInstance("branch-tier", [[0, 0], [100, 0]], coords)
    routes = [list(range(1, 19)), [19, 20]]
    _, decision = DynamicInsertionPlanner(instance, preference=(1.0, 0.0)).insert(
        routes, 21, (18.5, 0.0)
    )
    assert decision.candidate_task_counts.tolist() == [19, 3]
    assert decision.solver_method == "branch_and_bound"
    assert decision.fallback_reason is None
    assert len(decision.branch_bound_nodes) == 2


def test_large_event_uses_heuristic_for_all_vehicle_candidates() -> None:
    coords = [[float(index), 0.0] for index in range(1, 32)]
    instance = MTSPInstance("heuristic-tier", [[0, 0], [100, 0]], coords)
    routes = [list(range(1, 31)), [31]]
    _, decision = DynamicInsertionPlanner(instance, preference=(1.0, 0.0)).insert(
        routes, 32, (15.5, 0.0)
    )
    assert decision.candidate_task_counts.tolist() == [31, 2]
    assert decision.solver_method == "heuristic"
    assert decision.fallback_reason is None


def test_greedy_score_path_runs_tsp_only_for_selected_vehicle() -> None:
    class CountingPlanner(GreedyScoreDynamicInsertionPlanner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.replanned_vehicles = []

        def _replan_vehicle(self, vehicle, route, new_task_id, solver):
            self.replanned_vehicles.append(vehicle)
            return super()._replan_vehicle(vehicle, route, new_task_id, solver)

    instance = MTSPInstance(
        "greedy-score", [[0, 0], [100, 0]], [[1, 0], [2, 0], [101, 0]],
    )
    planner = CountingPlanner(instance, preference=(1.0, 0.0))
    updated, decision = planner.insert([[1, 2], [3]], 4, (1.5, 0.0))

    assert planner.replanned_vehicles == [0]
    assert decision.strategy == "greedy_score_then_tsp"
    assert decision.vehicle == 0
    assert decision.scoring_distance_deltas[0] < decision.scoring_distance_deltas[1]
    assert decision.greedy_insertion_positions.shape == (2,)
    assert updated[1] == [3]
    assert sum(route.count(4) for route in updated) == 1
    assert decision.solver_method == "held_karp"


def test_execution_selects_alternate_dynamic_strategy_without_changing_default() -> None:
    instance = MTSPInstance(
        "strategy-log", [[0, 0], [100, 0]], [[1, 0], [2, 0], [101, 0]],
        events=[DynamicEvent(task_id=4, coordinate=(1.5, 0), arrival_completed_tasks=1)],
    )
    execution = execute_routes(
        instance,
        [[1, 2], [3]],
        preference=(1.0, 0.0),
        dynamic_strategy="greedy_score_then_tsp",
    )
    event = execution.event_log[0]
    assert event["strategy"] == "greedy_score_then_tsp"
    assert len(event["scoring_distance_deltas"]) == 2
    assert event["selected_tsp_seconds"] >= 0


def test_execution_injects_events_without_running_nsga3_again() -> None:
    instance = MTSPInstance(
        "event", [[0, 0], [10, 0]], [[1, 0], [8, 0]],
        events=[DynamicEvent(arrival_completed_tasks=1, task_id=3, coordinate=(4, 0))],
    )
    execution = execute_routes(instance, [[1], [2]])
    assert execution.instance.task_count == 3
    assert len(execution.event_log) == 1
    assert sum(len(route) for route in execution.routes) == 3
    assert execution.event_log[0]["candidate_planning_seconds"] >= 0
    assert len(execution.event_log[0]["vehicle_planning_seconds"]) == 2


def test_execution_uses_explicit_preference_for_dynamic_insertion() -> None:
    instance = MTSPInstance(
        "event-preference", [[0, 0], [100, 0]], [[1, 0], [2, 0], [101, 0]],
        events=[DynamicEvent(task_id=4, coordinate=(1.5, 0), arrival_completed_tasks=1)],
    )
    execution = execute_routes(instance, [[1, 2], [3]], preference=(1.0, 0.0))
    assert execution.event_log[0]["vehicle"] == 0


def test_execution_uses_parallel_equal_speed_completion_order() -> None:
    """The global counter follows projected arrival times, not vehicle polling."""
    instance = MTSPInstance(
        "parallel", [[0, 0], [100, 0]], [[10, 0], [90, 0], [20, 0], [80, 0]],
        events=[DynamicEvent(task_id=5, coordinate=(50, 0), arrival_completed_tasks=2)],
    )
    execution = execute_routes(instance, [[1, 3], [2, 4]])
    assert sum(route.count(5) for route in execution.routes) == 1
    assert execution.event_log[0]["completed_tasks"] == 2
    assert execution.event_log[0]["simulated_time_seconds"] == 10.0
    assert execution.simulated_time_seconds >= 20.0
