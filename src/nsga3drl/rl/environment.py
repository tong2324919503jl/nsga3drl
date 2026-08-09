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
    front_indices = non_dominated_sort(objectives)[0]
    front_size = len(front_indices)
    front = objectives[front_indices]
    front_range = _safe_range(front, reference)
    efficiency_extreme = front[np.argmin(front[:, 0])]
    fairness_extreme = front[np.argmin(front[:, 1])]
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
        front_range[0],
        front_range[1],
        efficiency_extreme[1] / reference[1],
        fairness_extreme[0] / reference[0],
        *algorithm.last_operator_credits.tolist(),
        *algorithm.last_operator_allocation.tolist(),
        algorithm.stagnation_generations / max(max_generations, 1),
    ], dtype=np.float32)


def _safe_range(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(2, dtype=float)
    return (values.max(axis=0) - values.min(axis=0)) / reference


class OperatorSelectionEnv:
    """One RL step equals one NSGA-III generation; action selects a mutation operator."""

    action_count = len(NSGA3.ACTIONS)
    observation_size = 26

    def __init__(
        self,
        instance: MTSPInstance,
        population_size: int = 48,
        generations: int = 40,
        seed: int = 0,
        preference: tuple[float, float] | None = None,
    ):
        self.instance = instance
        self.population_size = population_size
        self.generations = generations
        self.seed = seed
        self.preference = preference if preference is not None else instance.preference
        if self.preference is None:
            raise ValueError("a preference must be present in the dataset or supplied to the environment")
        self.algorithm: NSGA3 | None = None

    def reset(self, seed: int | None = None) -> np.ndarray:
        actual_seed = self.seed if seed is None else seed
        # Dynamic events are deliberately excluded from the learning episode.
        # They are applied later by the execution-only insertion simulator.
        static_instance = self.instance.without_events().with_preference(self.preference)
        self.algorithm = NSGA3(OpenMTSPProblem(static_instance), self.population_size, seed=actual_seed)
        self.algorithm.initialize()
        return population_features(self.algorithm, self.generations)

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        if self.algorithm is None:
            raise RuntimeError("call reset before step")
        actions = np.asarray(action, dtype=int)
        if actions.ndim == 0:
            if not 0 <= int(actions) < self.action_count:
                raise ValueError("invalid action")
            action_counts = np.bincount([int(actions)], minlength=self.action_count)
        else:
            if actions.shape != (self.population_size,) or np.any((actions < 0) | (actions >= self.action_count)):
                raise ValueError("action vector must contain one valid action per offspring")
            action_counts = np.bincount(actions, minlength=self.action_count)
        before = self.algorithm.current_hypervolume()
        record = self.algorithm.step(action, process_dynamic_events=False)
        after = self.algorithm.current_hypervolume()
        reward = float(np.clip((after - before) / max(abs(before), 1e-9), -1.0, 1.0))
        done = self.algorithm.generation >= self.generations
        info = {
            "hypervolume": after,
            "generation": self.algorithm.generation,
            "operator": NSGA3.ACTIONS[int(actions)] if actions.ndim == 0 else "mixed",
            "operator_counts": {name: int(action_counts[index]) for index, name in enumerate(NSGA3.ACTIONS)},
            "operator_credits": dict(zip(NSGA3.ACTIONS, self.algorithm.last_operator_credits.tolist())),
            "events_injected": 0,
        }
        return population_features(self.algorithm, self.generations), reward, done, info
