import pytest

from nsga3drl.experiments.evaluate import preference_label
from nsga3drl.experiments.evaluate import evaluate_one
from nsga3drl.experiments.batch import parse_checkpoints, parse_preferences
from nsga3drl.experiments.resume_store import ResumeStore, task_key
from nsga3drl.data.generator import generate_instance


def test_resume_store_keeps_completed_results_and_accepts_same_config(tmp_path) -> None:
    path = tmp_path / "resume.sqlite3"
    config = {"dataset": "test", "seed": 42}
    task = {"instance_id": "case-1", "preference": [1.0, 0.0]}
    result = {"preferred_solution": {"total_distance": 123.0}}

    store = ResumeStore(path, config)
    store.save(task, result)
    store.close()

    resumed = ResumeStore(path, config)
    assert task_key(task) in resumed.completed_keys()
    assert resumed.records() == [(task, result)]
    resumed.close()


def test_resume_store_rejects_different_run_config(tmp_path) -> None:
    path = tmp_path / "resume.sqlite3"
    ResumeStore(path, {"seed": 42}).close()

    with pytest.raises(ValueError, match="different run configuration"):
        ResumeStore(path, {"seed": 43})


def test_preference_labels_are_stable_for_report_names() -> None:
    assert preference_label([1.0, 0.0]) == "e100_f000"
    assert preference_label([0.2, 0.8]) == "e020_f080"
    assert preference_label(None) == "dataset_preference"


def test_preference_grid_is_validated() -> None:
    assert parse_preferences([[1.0, 0.0], [0.2, 0.8]]) == [[1.0, 0.0], [0.2, 0.8]]
    with pytest.raises(ValueError, match="sum to 1"):
        parse_preferences([[2.0, 8.0]])


def test_multiple_checkpoint_versions_are_supported(tmp_path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.touch()
    second.touch()

    assert parse_checkpoints([f"short={first}", str(second)]) == [
        ("short", str(first.resolve())),
        ("second", str(second.resolve())),
    ]


def test_parallel_baseline_evaluation_applies_preference_override(tmp_path) -> None:
    instance = generate_instance(
        task_count=10,
        vehicle_count=2,
        seed=11,
        dynamic_rate=0.1,
        preference=(1.0, 0.0),
    )
    path = tmp_path / "instance.json"
    instance.save(path)

    result = evaluate_one({
        "instance_path": str(path),
        "instance_id": instance.instance_id,
        "method": "baseline",
        "model_version": None,
        "dynamic_strategy": "greedy_score_then_tsp",
        "preference": [0.0, 1.0],
        "checkpoint": None,
        "generations": 1,
        "population": 4,
        "seed": 7,
    })

    assert result["preferred_solution"]["preference"] == [0.0, 1.0]
    assert result["events"] == 1
    assert result["preferred_solution"]["total_distance"] >= 0
