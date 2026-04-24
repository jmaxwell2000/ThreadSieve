from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "workspace": "./ThreadSieve",
    "confidence_threshold": 0.55,
    "models": {
        "extract": {
            "provider": "offline",
            "base_url": "http://localhost:11434/v1",
            "model": "qwen2.5:14b",
            "api_key_env": "THREADSIEVE_API_KEY",
        }
    },
    "redaction": {
        "enabled": False,
        "patterns": [],
    },
}


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]

    @property
    def workspace(self) -> Path:
        return expand_path(str(self.raw.get("workspace", DEFAULT_CONFIG["workspace"]))).resolve()

    @property
    def confidence_threshold(self) -> float:
        return float(self.raw.get("confidence_threshold", DEFAULT_CONFIG["confidence_threshold"]))

    @property
    def extract_model(self) -> dict[str, Any]:
        models = self.raw.get("models") or {}
        return dict(models.get("extract") or DEFAULT_CONFIG["models"]["extract"])


def default_config_path() -> Path:
    return Path.home() / ".threadsieve" / "config.json"


def expand_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value)))


def load_config(config_path: str | None = None) -> Config:
    path = expand_path(config_path) if config_path else default_config_path()
    if not path.exists():
        return Config(path=path, raw=json.loads(json.dumps(DEFAULT_CONFIG)))
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    merged = merge_dicts(DEFAULT_CONFIG, loaded)
    return Config(path=path, raw=merged)


def write_default_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8") as handle:
            json.dump(DEFAULT_CONFIG, handle, indent=2)
            handle.write("\n")


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
