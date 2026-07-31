from __future__ import annotations

from pathlib import Path

from .schema import MTSPInstance


def load_dataset(directory: str | Path) -> list[MTSPInstance]:
    """Load all instance JSON files in a generated dataset directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {directory}")
    paths = sorted(path for path in directory.glob("*.json") if path.name != "manifest.json")
    if not paths:
        raise ValueError(f"no instance JSON files found in: {directory}")
    return [MTSPInstance.load(path) for path in paths]
