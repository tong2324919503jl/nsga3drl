from nsga3drl.algorithms.nsga3 import NSGA3
from nsga3drl.data.schema import DynamicEvent, MTSPInstance
from nsga3drl.dynamic.insertion import DynamicInsertionPlanner
from nsga3drl.problem.encoding import validate
from nsga3drl.problem.open_mtsp import OpenMTSPProblem


def test_new_task_is_inserted_once() -> None:
    instance = MTSPInstance("dynamic", [[0, 0], [10, 0]], [[1, 0], [8, 0]])
    routes, decision = DynamicInsertionPlanner(instance, k_neighbors=2).insert([[1], [2]], 3, (1.5, 0.0))
    assert sum(route.count(3) for route in routes) == 1
    assert decision.vehicle == 0


def test_nsga3_injects_event_and_expands_every_genome() -> None:
    instance = MTSPInstance(
        "event", [[0, 0], [10, 0]], [[1, 0], [8, 0]],
        events=[DynamicEvent(arrival_generation=1, task_id=3, coordinate=(4, 0))],
    )
    algorithm = NSGA3(OpenMTSPProblem(instance), population_size=8, seed=3)
    result = algorithm.run(1)
    assert algorithm.problem.task_count == 3
    assert len(algorithm.event_log) == 1
    assert all(validate(genome, 3, 2) for genome in result.population)
