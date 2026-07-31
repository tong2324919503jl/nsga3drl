from __future__ import annotations

import numpy as np

from ..algorithms.nsga3 import NSGA3, non_dominated_sort
from ..data.schema import MTSPInstance
from ..problem.open_mtsp import OpenMTSPProblem


def population_features(algorithm: NSGA3, max_generations: int) -> np.ndarray:
    assert algorithm.objectives is not None
    objectives = algorithm.objectives
    current_hv = algorithm.current_hypervolume()
    reference = np.maximum(algorithm.hv_reference, 1e-9)
    # Use the fixed HV reference for this dynamic segment. Dividing each mean
    # by itself would make both mean features always equal to one.
    front_size = len(non_dominated_sort(objectives)[0])
    return np.asarray([
        algorithm.generation / max(max_generations, 1),
        current_hv / max(float(algorithm.hv_reference.prod()), 1e-9),
        algorithm.last_hv_delta / max(abs(current_hv), 1e-9),
        objectives[:, 0].mean() / reference[0],
        objectives[:, 0].std() / reference[0],
        objectives[:, 0].min() / reference[0],
        objectives[:, 1].mean() / reference[1],
        objectives[:, 1].std() / reference[1],
        objectives[:, 1].min() / reference[1],
        front_size / len(objectives),
        len(np.unique(np.argmin(np.abs(objectives[:, 0, None] - np.linspace(objectives[:, 0].min(), objectives[:, 0].max(), 8)), axis=1))) / 8,
        algorithm.problem.instance.preference[0],
        algorithm.problem.instance.preference[1],
    ], dtype=np.float32)


class OperatorSelectionEnv:
    """One RL step equals one NSGA-III generation; action selects a mutation operator."""

    action_count = len(NSGA3.ACTIONS)
    observation_size = 13

    def __init__(self, instance: MTSPInstance, population_size: int = 48, generations: int = 40, seed: int = 0):
        self.instance = instance
        self.population_size = population_size
        self.generations = generations
        self.seed = seed
        self.algorithm: NSGA3 | None = None

    def reset(self, seed: int | None = None) -> np.ndarray:
        actual_seed = self.seed if seed is None else seed
        self.algorithm = NSGA3(OpenMTSPProblem(self.instance), self.population_size, seed=actual_seed)
        self.algorithm.initialize()
        return population_features(self.algorithm, self.generations)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.algorithm is None:
            raise RuntimeError("call reset before step")
        if not 0 <= action < self.action_count:
            raise ValueError("invalid action")
        injected = self.algorithm.process_events(self.algorithm.generation + 1)
        before = self.algorithm.current_hypervolume()
        record = self.algorithm.step(action, process_dynamic_events=False)
        after = self.algorithm.current_hypervolume()
        reward = float(np.clip((after - before) / max(abs(before), 1e-9), -1.0, 1.0))
        done = self.algorithm.generation >= self.generations
        info = {
            "hypervolume": after,
            "generation": self.algorithm.generation,
            "operator": NSGA3.ACTIONS[action],
            "events_injected": injected,
        }
        return population_features(self.algorithm, self.generations), reward, done, info
