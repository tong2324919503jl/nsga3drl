import numpy as np

from nsga3drl.algorithms.route_construction import improve_routes, initial_population
from nsga3drl.data.generator import generate_instance
from nsga3drl.problem.encoding import decode, validate


def test_constructed_population_assigns_every_vehicle() -> None:
    instance = generate_instance(30, 10, seed=11, distribution="clustered")
    population = initial_population(instance, population_size=12, rng=np.random.default_rng(11))
    assert all(validate(genome, 30, 10) for genome in population)
    assert all(all(route for route in decode(genome, 10)) for genome in population)


def test_local_search_preserves_tasks_and_nonempty_routes() -> None:
    instance = generate_instance(20, 4, seed=12, distribution="clustered")
    improved = improve_routes([[1, 2, 3, 4, 5], [], [6, 7, 8, 9, 10], list(range(11, 21))], instance)
    assert all(improved)
    assert sorted(task for route in improved for task in route) == list(range(1, 21))
