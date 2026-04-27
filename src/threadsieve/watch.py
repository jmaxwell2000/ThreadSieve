from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .pipeline import SOURCE_SUFFIXES, extract_sources


@dataclass(frozen=True)
class FileSnapshot:
    size: int
    mtime: float
    observed_at: float


def discover_watch_files(source: Path) -> list[Path]:
    if source.is_file():
        raise RuntimeError("Watch mode requires a folder source. Use `threadsieve extract --source FILE --out DIR` for one file.")
    if not source.exists():
        raise RuntimeError(f"Source folder does not exist: {source}")
    if not source.is_dir():
        raise RuntimeError(f"Source is not a folder: {source}")
    return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES)


def current_snapshot(path: Path, now: float) -> FileSnapshot:
    stat = path.stat()
    return FileSnapshot(size=stat.st_size, mtime=stat.st_mtime, observed_at=now)


def stable_ready_files(
    source: Path,
    snapshots: dict[Path, FileSnapshot],
    settle_seconds: float,
    now: float,
) -> list[Path]:
    ready: list[Path] = []
    for path in discover_watch_files(source):
        try:
            current = current_snapshot(path, now)
        except FileNotFoundError:
            snapshots.pop(path, None)
            continue
        previous = snapshots.get(path)
        if previous is None or previous.size != current.size or previous.mtime != current.mtime:
            snapshots[path] = current
            continue
        if now - previous.observed_at >= settle_seconds:
            ready.append(path)
    return ready


def build_watch_model_config(config: Any, provider: str | None = None, model: str | None = None) -> dict[str, Any]:
    model_config = dict(config.extract_model)
    provider_config = config.raw.get("provider")
    if isinstance(provider_config, dict):
        model_config.update(provider_config)
        if "name" in provider_config and "provider" not in provider_config:
            model_config["provider"] = provider_config["name"]
    if provider:
        model_config["provider"] = provider
        if provider == "openrouter" and "model" not in model_config:
            model_config["model"] = "openai/gpt-4.1-mini"
    if model:
        model_config["model"] = model
    return model_config


def run_watch(
    *,
    source: Path,
    output_root: Path,
    model_config: dict[str, Any],
    threshold: float,
    force: bool,
    system_prompt: str,
    write_index: bool,
    overwrite_existing: bool,
    semantic_logs: bool,
    semantic_prompt: str,
    interval_seconds: float = 5,
    settle_seconds: float = 2,
    once: bool = False,
    summary_printer: Callable[[dict[str, object]], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    if interval_seconds <= 0:
        raise RuntimeError("--interval must be greater than 0.")
    if settle_seconds < 0:
        raise RuntimeError("--settle-seconds must be 0 or greater.")

    source = source.resolve()
    discover_watch_files(source)
    output_root.mkdir(parents=True, exist_ok=True)
    snapshots: dict[Path, FileSnapshot] = {}
    processed_snapshots: dict[Path, tuple[int, float]] = {}
    summaries: list[dict[str, Any]] = []

    while True:
        now = clock()
        ready = stable_ready_files(source, snapshots, settle_seconds, now)
        for path in ready:
            snapshot = snapshots.get(path)
            if not snapshot:
                continue
            key = (snapshot.size, snapshot.mtime)
            if processed_snapshots.get(path) == key:
                continue
            summary = extract_sources(
                source=path,
                output_root=output_root,
                model_config=model_config,
                threshold=threshold,
                force=force,
                dry_run=False,
                system_prompt=system_prompt,
                write_index=write_index,
                overwrite_existing=overwrite_existing,
                semantic_logs=semantic_logs,
                semantic_prompt=semantic_prompt,
            )
            summaries.append(summary)
            processed_snapshots[path] = key
            if summary_printer:
                summary_printer(summary)
        if once:
            if ready or not discover_watch_files(source):
                return summaries
        sleeper(interval_seconds if not once else min(interval_seconds, max(0.05, settle_seconds or 0.05)))
