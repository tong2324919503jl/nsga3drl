import numpy as np

from nsga3drl.data.generator import generate_instance
from nsga3drl.rl.environment import OperatorSelectionEnv
from nsga3drl.rl.ppo import PPOTrainer


def test_rl_environment_keeps_dynamic_events_out_of_training() -> None:
    instance = generate_instance(6, 2, seed=2, dynamic_rate=0.2)
    instance.events[0] = instance.events[0].__class__(
        task_id=instance.events[0].task_id,
        coordinate=instance.events[0].coordinate,
        arrival_completed_tasks=1,
    )
    environment = OperatorSelectionEnv(instance, population_size=8, generations=2, seed=1)
    observation = environment.reset()
    observation, reward, done, info = environment.step(0)
    assert observation.shape == (OperatorSelectionEnv.observation_size,)
    assert np.isfinite(observation).all() and np.isfinite(reward)
    assert info["events_injected"] == 0
    assert environment.algorithm.problem.task_count == 6


def test_ppo_update_and_checkpoint(tmp_path) -> None:
    environment = OperatorSelectionEnv(generate_instance(6, 2, seed=4), population_size=8, generations=2, seed=4)
    trainer = PPOTrainer(OperatorSelectionEnv.observation_size, OperatorSelectionEnv.action_count, seed=4)
    metrics = trainer.update(trainer.collect([environment]))
    assert all(np.isfinite(value) for value in metrics.values())
    checkpoint = tmp_path / "ppo.pt"
    trainer.save(checkpoint)
    assert PPOTrainer.load(checkpoint).greedy_action(environment.reset()) in range(OperatorSelectionEnv.action_count)


def test_ppo_assigns_multiple_operators_to_one_generation() -> None:
    environment = OperatorSelectionEnv(generate_instance(6, 2, seed=5), population_size=8, generations=2, seed=5)
    trainer = PPOTrainer(OperatorSelectionEnv.observation_size, OperatorSelectionEnv.action_count, seed=5)
    actions = trainer.greedy_actions(environment.reset(), environment.population_size)
    assert actions.shape == (environment.population_size,)
    assert set(actions.tolist()) == set(range(OperatorSelectionEnv.action_count))
    _, _, _, info = environment.step(actions)
    assert sum(info["operator_counts"].values()) == environment.population_size
