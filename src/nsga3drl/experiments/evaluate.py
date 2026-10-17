from __future__ import annotations

import time
from collections import Counter

import numpy as np

from ..algorithms.nsga3 import NSGA3, NSGA3Result
from ..data.schema import MTSPInstance
from ..dynamic.execution import execute_routes
from ..problem.encoding import decode, encode
from ..problem.open_mtsp import OpenMTSPProblem


_PPO_TRAINERS: dict[str, object] = {}


def _load_trainer(checkpoint: str):
    trainer = _PPO_TRAINERS.get(checkpoint)
    if trainer is None:
        from ..rl.ppo import PPOTrainer

        import torch

        torch.set_num_threads(1)
        trainer = PPOTrainer.load(checkpoint, device="cpu")
        _PPO_TRAINERS[checkpoint] = trainer
    return trainer


def evaluate_one(task: dict) -> dict:
    """Evaluate one method/instance/preference/strategy combination in a worker."""
    instance = MTSPInstance.load(task["instance_path"])
    preference = task["preference"]
    if preference is None:
        preference = instance.preference
    if preference is None:
        raise ValueError(f"{instance.instance_id} has no preference; pass --preference")
    preference = tuple(float(value) for value in preference)
    started = time.perf_counter()

    if task["method"] == "baseline":
        instance = instance.with_preference(preference)
        algorithm = NSGA3(
            OpenMTSPProblem(instance),
            population_size=task["population"],
            seed=task["seed"],
        )
        result = algorithm.run(task["generations"])
        operator_counts = {}
        mean_probabilities = {}
        reward_sum = 0.0
        final_hv = float(result.history[-1]["hypervolume"])
    else:
        from ..rl.environment import OperatorSelectionEnv

        trainer = _load_trainer(task["checkpoint"])
        environment = OperatorSelectionEnv(
            instance,
            population_size=task["population"],
            generations=task["generations"],
            seed=task["seed"],
            preference=preference,
        )
        observation = environment.reset()
        reward_sum = 0.0
        operator_counts: Counter[str] = Counter()
        probability_sum = np.zeros(OperatorSelectionEnv.action_count, dtype=float)
        done = False
        while not done:
            probability_sum += trainer.operator_probabilities(observation)
            actions = trainer.greedy_actions(observation, environment.population_size)
            observation, reward, done, info = environment.step(actions)
            reward_sum += reward
            operator_counts.update(info["operator_counts"])
        algorithm = environment.algorithm
        result = NSGA3Result(algorithm.population, algorithm.objectives, algorithm.history)
        final_hv = float(info["hypervolume"])
        operator_counts = {
            name: int(operator_counts[name]) for name in algorithm.ACTIONS
        }
        mean_probabilities = {
            name: float(probability_sum[index] / task["generations"])
            for index, name in enumerate(algorithm.ACTIONS)
        }

    preferred_index = result.preferred_index(preference)
    preferred_genome = result.population[preferred_index]
    initial_objective = result.objectives[preferred_index]
    initial_problem = OpenMTSPProblem(instance.without_events())
    initial_lengths = initial_problem.route_lengths(preferred_genome).tolist()
    execution = execute_routes(
        instance,
        decode(preferred_genome, instance.vehicle_count),
        preference,
        task["dynamic_strategy"],
    )
    executed_problem = OpenMTSPProblem(execution.instance)
    executed_genome = encode(execution.routes, instance.vehicle_count)
    final_objective = executed_problem.evaluate(executed_genome)
    total_elapsed = time.perf_counter() - started

    row = {
        "instance_id": instance.instance_id,
        "final_hv": final_hv,
        "reward_sum": float(reward_sum),
        "operator_counts": operator_counts,
        "mean_operator_probabilities": mean_probabilities,
        "events": len(execution.event_log),
        "active_tasks": executed_problem.task_count,
        "dynamic_time_seconds": execution.elapsed_seconds,
        "dynamic_strategy": task["dynamic_strategy"],
        "dynamic_tsp_planning_seconds": sum(
            event["candidate_planning_seconds"] for event in execution.event_log
        ),
        "dynamic_event_log": execution.event_log,
        "elapsed_seconds": total_elapsed,
        "preferred_solution": {
            "preference": list(preference),
            "initial_total_distance": float(initial_objective[0]),
            "initial_variance": float(initial_objective[1]),
            "initial_vehicle_route_lengths": initial_lengths,
            "total_distance": float(final_objective[0]),
            "variance": float(final_objective[1]),
            "vehicle_route_lengths": executed_problem.route_lengths(executed_genome).tolist(),
        },
    }
    return row


def preference_label(preference: tuple[float, float] | list[float] | None) -> str:
    if preference is None:
        return "dataset_preference"
    efficiency, fairness = preference
    return f"e{round(efficiency * 100):03d}_f{round(fairness * 100):03d}"


def summarize_case(
    rows: list[dict],
    method: str,
    model_version: str | None,
    strategy: str,
    preference: list[float] | None,
    dataset: str,
    generations: int,
    population: int,
    checkpoint: str | None,
) -> dict:
    """Build a result document compatible with existing evaluation JSON files."""
    operator_counts: Counter[str] = Counter()
    probability_sums: Counter[str] = Counter()
    for row in rows:
        operator_counts.update(row["operator_counts"])
        probability_sums.update(row["mean_operator_probabilities"])
    mean_probabilities = {
        name: probability_sums[name] / max(len(rows), 1)
        for name in probability_sums
    }
    report = {
        "mode": "evaluation" if method == "ppo" else "baseline",
        "method": method,
        "model_version": model_version,
        "dataset": dataset,
        "instances": len(rows),
        "generations": generations,
        "population": population,
        "dynamic_strategy": strategy,
        "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
        "task_elapsed_seconds_sum": sum(row["elapsed_seconds"] for row in rows),
        "mean_instance_seconds": (
            sum(row["elapsed_seconds"] for row in rows) / len(rows) if rows else 0.0
        ),
        "operator_counts": dict(operator_counts),
        "mean_operator_probabilities": mean_probabilities,
        "results": rows,
    }
    if checkpoint is not None:
        report["checkpoint"] = checkpoint
    if preference is not None:
        report["preference_override"] = preference
    return report
