from nsga3drl.algorithms.nsga3 import NSGA3
from nsga3drl.data.generator import generate_instance
from nsga3drl.problem.encoding import decode, validate
from nsga3drl.problem.open_mtsp import OpenMTSPProblem


def test_nsga3_preserves_legal_population() -> None:
    instance = generate_instance(12, 3, seed=5)
    result = NSGA3(OpenMTSPProblem(instance), population_size=12, seed=5).run(3)
    assert all(validate(genome, 12, 3) for genome in result.population)
    assert all(all(route for route in decode(genome, 3)) for genome in result.population)
    assert result.history[-1]["hypervolume"] >= 0
    assert result.preferred_index((0.5, 0.5)) in result.pareto_indices
