from nsga3drl.data.generator import generate_dataset, generate_instance
from nsga3drl.data.io import load_dataset
from nsga3drl.data.schema import DynamicEvent, MTSPInstance
from nsga3drl.data.validation import validate_dataset, validate_instance


def test_generated_instance_is_valid() -> None:
    report = validate_instance(generate_instance(10, 3, 4, dynamic_rate=0.2))
    assert report.valid
    assert report.event_count == 2


def test_dataset_manifest_matches_files(tmp_path) -> None:
    generate_dataset(tmp_path, 3, 10, 2, seed=3)
    assert all(report.valid for report in validate_dataset(tmp_path))
    assert len(load_dataset(tmp_path)) == 3


def test_overlapping_task_coordinates_fail_validation() -> None:
    instance = MTSPInstance("overlap", [[0, 0]], [[1, 1], [1, 1]])
    report = validate_instance(instance)
    assert not report.valid
    assert "coordinates overlap" in report.errors[0]


def test_vehicle_starting_at_task_is_a_warning() -> None:
    instance = MTSPInstance("start", [[1, 1]], [[1, 1], [2, 2]])
    report = validate_instance(instance)
    assert report.valid
    assert len(report.warnings) == 1
