from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

from ..data.schema import MTSPInstance
from .evaluate import evaluate_one, preference_label, summarize_case
from .resume_store import ResumeStore, task_key


def parse_preferences(values: list[list[float]] | None) -> list[list[float] | None]:
    if not values:
        return [None]
    preferences = []
    seen = set()
    for pair in values:
        if min(pair) < 0 or abs(sum(pair) - 1.0) > 1e-8:
            raise ValueError(f"preference weights must be non-negative and sum to 1: {pair}")
        preference = (float(pair[0]), float(pair[1]))
        if preference in seen:
            raise ValueError(f"duplicate preference requested: {pair}")
        seen.add(preference)
        preferences.append(list(preference))
    return preferences


def parse_checkpoints(values: list[str] | None) -> list[tuple[str, str]]:
    """Accept checkpoint paths or repeatable LABEL=PATH values."""
    if not values:
        return []
    parsed = []
    seen_labels = set()
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
        else:
            path = value
            label = Path(path).stem
        if not label or not path or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise ValueError(f"invalid checkpoint specification: {value!r}; use LABEL=PATH")
        if label in seen_labels:
            raise ValueError(f"duplicate checkpoint label: {label}")
        checkpoint = Path(path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"PPO checkpoint does not exist: {checkpoint}")
        seen_labels.add(label)
        parsed.append((label, str(checkpoint)))
    return parsed


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _run_config(args, dataset_files: list[Path], preferences, checkpoints) -> dict:
    files = [
        {"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in dataset_files
    ]
    checkpoint_versions = []
    for label, path in checkpoints:
        stat = Path(path).stat()
        checkpoint_versions.append({
            "label": label,
            "path": path,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return {
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_files": files,
        "methods": sorted(set(args.method)),
        "strategies": sorted(set(args.dynamic_strategy)),
        "preferences": preferences,
        "checkpoints": checkpoint_versions,
        "generations": args.generations,
        "population": args.population,
        "seed": args.seed,
    }


def _build_tasks(args, dataset_files: list[Path], instances: list[MTSPInstance], preferences, checkpoints) -> list[dict]:
    tasks = []
    for method in sorted(set(args.method)):
        versions = checkpoints if method == "ppo" else [(None, None)]
        for model_version, checkpoint in versions:
            for strategy in sorted(set(args.dynamic_strategy)):
                for preference in preferences:
                    for index, (path, instance) in enumerate(zip(dataset_files, instances)):
                        effective_preference = preference
                        if effective_preference is None:
                            effective_preference = (
                                list(instance.preference) if instance.preference is not None else None
                            )
                        if effective_preference is None:
                            raise ValueError(
                                f"{instance.instance_id} has no stored preference; provide --preference"
                            )
                        tasks.append({
                            "instance_path": str(path.resolve()),
                            "instance_id": instance.instance_id,
                            "method": method,
                            "model_version": model_version,
                            "dynamic_strategy": strategy,
                            "preference": effective_preference,
                            "checkpoint": checkpoint,
                            "generations": args.generations,
                            "population": args.population,
                            "seed": args.seed + index,
                        })
    return tasks


def _atomic_write_json(path: Path, value: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def _write_reports(store: ResumeStore, args, started: float) -> int:
    records = store.records()
    groups: dict[tuple, list[tuple[dict, dict]]] = {}
    for task, row in records:
        key = (
            task["method"], task["model_version"], task["dynamic_strategy"],
            tuple(task["preference"]) if task["preference"] is not None else None,
        )
        groups.setdefault(key, []).append((task, row))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for (method, model_version, strategy, preference), items in groups.items():
        items.sort(key=lambda item: item[0]["instance_id"])
        rows = [row for _, row in items]
        first_task = items[0][0]
        tag = preference_label(preference)
        report = summarize_case(
            rows, method, model_version, strategy,
            list(preference) if preference else None,
            args.dataset, args.generations, args.population,
            first_task["checkpoint"],
        )
        wall_time = time.perf_counter() - started
        report["completed_instances"] = len(rows)
        report["expected_instances"] = args.expected_instances
        report["parallel_wall_clock_seconds_this_run"] = wall_time
        report["task_elapsed_seconds_sum"] = sum(row["elapsed_seconds"] for row in rows)
        version_tag = f"_{model_version}" if model_version else ""
        name = f"{method}{version_tag}_{strategy}_{tag}.eval.json"
        _atomic_write_json(output_dir / name, report)

    complete = len(records)
    state = {
        "completed_tasks": complete,
        "total_tasks": args.total_tasks,
        "complete": complete == args.total_tasks,
        "wall_clock_seconds_this_run": time.perf_counter() - started,
        "state_database": str(Path(args.state_db).resolve()),
        "reports_directory": str(output_dir.resolve()),
    }
    _atomic_write_json(output_dir / "batch_status.json", state)
    return complete


def run_batch(args, dataset_files: list[Path], instances: list[MTSPInstance], preferences, checkpoints) -> None:
    tasks = _build_tasks(args, dataset_files, instances, preferences, checkpoints)
    args.total_tasks = len(tasks)
    args.state_db = args.state_db or str(Path(args.output_dir) / "resume.sqlite3")
    store = ResumeStore(args.state_db, _run_config(args, dataset_files, preferences, checkpoints))
    completed = store.completed_keys()
    pending = [task for task in tasks if task_key(task) not in completed]
    started = time.perf_counter()
    print(
        f"jobs={len(tasks)} completed={len(completed)} pending={len(pending)} "
        f"workers={args.workers} state_db={args.state_db}",
        flush=True,
    )

    errors = 0
    interrupted = False
    previous_handlers = {}
    futures = {}

    def interrupt_handler(_signum, _frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, interrupt_handler)

    def record_completed(future, task) -> None:
        nonlocal errors
        try:
            row = future.result()
            store.save(task, row)
            completed.add(task_key(task))
            print(
                f"done={len(completed)}/{len(tasks)} method={task['method']} "
                f"version={task['model_version']} strategy={task['dynamic_strategy']} "
                f"preference={task['preference']} instance={task['instance_id']} "
                f"distance={row['preferred_solution']['total_distance']:.3f}",
                flush=True,
            )
        except Exception as error:
            errors += 1
            print(
                f"ERROR method={task['method']} version={task['model_version']} "
                f"strategy={task['dynamic_strategy']} preference={task['preference']} "
                f"instance={task['instance_id']}: {error}",
                file=sys.stderr,
                flush=True,
            )

    try:
        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init)
        try:
            remaining = iter(pending)
            for _ in range(min(args.workers, len(pending))):
                task = next(remaining)
                futures[pool.submit(evaluate_one, task)] = task
            while futures:
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in finished:
                    task = futures.pop(future)
                    record_completed(future, task)
                    try:
                        next_task = next(remaining)
                    except StopIteration:
                        continue
                    futures[pool.submit(evaluate_one, next_task)] = next_task
        except KeyboardInterrupt:
            interrupted = True
            print("Stop requested: waiting for active evaluations and saving them before exit.", flush=True)
            for future in futures:
                future.cancel()
            while futures:
                finished, _ = wait(futures)
                for future in finished:
                    task = futures.pop(future)
                    if not future.cancelled():
                        record_completed(future, task)
        finally:
            pool.shutdown(wait=True, cancel_futures=interrupted)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    complete = _write_reports(store, args, started)
    store.close()
    if interrupted:
        print(
            f"Interrupted cleanly: saved {complete}/{len(tasks)} jobs. "
            f"Rerun the same command to resume; pending jobs will continue.",
            flush=True,
        )
        return

    print(
        f"finished={complete}/{len(tasks)} errors={errors} reports={args.output_dir} "
        f"state_db={args.state_db}",
        flush=True,
    )
    if errors or complete < len(tasks):
        raise SystemExit(1)
