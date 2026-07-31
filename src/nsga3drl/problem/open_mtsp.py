from __future__ import annotations

import numpy as np

from ..data.schema import MTSPInstance
from .encoding import decode, validate


class OpenMTSPProblem:
    def __init__(self, instance: MTSPInstance):
        self.instance = instance
        self.task_count = instance.task_count
        self.vehicle_count = instance.vehicle_count
        # The separator encoding reserves 0 and uses 1..n as task genes.
        if sorted(instance.task_ids.tolist()) != list(range(1, self.task_count + 1)):
            raise ValueError("this MVP requires initial task_ids to be consecutive integers 1..n")
        self._id_to_index = {int(task_id): index for index, task_id in enumerate(instance.task_ids)}

    def route_lengths(self, genome: np.ndarray | list[int]) -> np.ndarray:
        if not validate(genome, self.task_count, self.vehicle_count):
            raise ValueError("illegal genome")
        lengths = np.zeros(self.vehicle_count, dtype=float)
        for vehicle, route in enumerate(decode(genome, self.vehicle_count)):
            current = self.instance.vehicle_starts[vehicle]
            for task_id in route:
                target = self.instance.task_coordinates[self._id_to_index[task_id]]
                lengths[vehicle] += float(np.linalg.norm(target - current))
                current = target
        return lengths

    def evaluate(self, genome: np.ndarray | list[int]) -> np.ndarray:
        lengths = self.route_lengths(genome)
        return np.asarray([lengths.sum(), lengths.var()], dtype=float)

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        return np.vstack([self.evaluate(individual) for individual in population])
