import numpy as np

from nsga3drl.data.schema import MTSPInstance
from nsga3drl.problem.encoding import decode, random_genome, validate
from nsga3drl.problem.open_mtsp import OpenMTSPProblem


def test_genome_is_legal_and_round_trips() -> None:
    genome = random_genome(6, 3, np.random.default_rng(7))
    assert validate(genome, 6, 3)
    assert sum(len(route) for route in decode(genome, 3)) == 6


def test_open_route_objectives_have_no_return_leg() -> None:
    instance = MTSPInstance("tiny", [[0, 0], [10, 0]], [[3, 0], [8, 0]])
    problem = OpenMTSPProblem(instance)
    objectives = problem.evaluate(np.asarray([1, 0, 2]))
    assert np.allclose(objectives, [5.0, 0.25])
