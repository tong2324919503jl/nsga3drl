from nsga3drl.data.schema import DynamicEvent, MTSPInstance
from nsga3drl.dynamic.execution import execute_routes
from nsga3drl.dynamic.insertion import DynamicInsertionPlanner


def test_new_task_is_inserted_once() -> None:
    instance = MTSPInstance("dynamic", [[0, 0], [10, 0]], [[1, 0], [8, 0]])
    routes, decision = DynamicInsertionPlanner(instance, k_neighbors=2).insert([[1], [2]], 3, (1.5, 0.0))
    assert sum(route.count(3) for route in routes) == 1
    assert decision.vehicle == 0


def test_execution_injects_events_without_running_nsga3_again() -> None:
    instance = MTSPInstance(
        "event", [[0, 0], [10, 0]], [[1, 0], [8, 0]],
        events=[DynamicEvent(arrival_completed_tasks=1, task_id=3, coordinate=(4, 0))],
    )
    execution = execute_routes(instance, [[1], [2]])
    assert execution.instance.task_count == 3
    assert len(execution.event_log) == 1
    assert sum(len(route) for route in execution.routes) == 3


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
