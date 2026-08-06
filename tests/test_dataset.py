from nsga3drl.data.generator import BALANCED_PREFERENCES, generate_dataset, generate_instance
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


def test_balanced_preferences_are_written_to_instances(tmp_path) -> None:
    generate_dataset(tmp_path, 5, 10, 2, seed=3, preference_choices=BALANCED_PREFERENCES)
    # Dataset loading sorts files by instance ID, not generation order.
    assert set(instance.preference for instance in load_dataset(tmp_path)) == set(BALANCED_PREFERENCES)


def test_paired_preferences_share_the_same_spatial_instance(tmp_path) -> None:
    generate_dataset(
        tmp_path, 1, 10, 2, seed=3, preference_choices=BALANCED_PREFERENCES,
        distribution_choices=("clustered",), paired_preferences=True,
    )
    instances = load_dataset(tmp_path)
    assert len(instances) == len(BALANCED_PREFERENCES)
    assert all((instance.task_coordinates == instances[0].task_coordinates).all() for instance in instances[1:])
    assert all((instance.vehicle_starts == instances[0].vehicle_starts).all() for instance in instances[1:])


def test_preference_can_be_omitted_from_json(tmp_path) -> None:
    generate_dataset(tmp_path, 1, 10, 2, seed=3, include_preference=False)
    assert load_dataset(tmp_path)[0].preference is None


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
