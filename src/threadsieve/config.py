from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "workspace": "./ThreadSieve",
    "sources": [],
    "output": "./knowledge",
    "confidence_threshold": 0.55,
    "models": {
        "extract": {
            "provider": "offline",
        }
    },
    "prompts": {
        "extract": "~/.threadsieve/prompts/extract.md",
        "semantic": "~/.threadsieve/prompts/semantic.md",
    },
    "redaction": {
        "enabled": False,
        "patterns": [],
    },
    "extractors": ["idea"],
    "behavior": {
        "skip_processed": True,
        "needs_review_confidence_threshold": 0.75,
        "overwrite_existing": False,
        "write_index": True,
        "dry_run": False,
        "semantic_logs": True,
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
    path = expand_path(config_path) if config_path else discover_config_path()
    if not path.exists():
        return Config(path=path, raw=json.loads(json.dumps(DEFAULT_CONFIG)))
    with path.open("r", encoding="utf-8") as handle:
        loaded = load_config_text(handle.read(), path)
    merged = merge_dicts(DEFAULT_CONFIG, loaded)
    return Config(path=path, raw=merged)


def discover_config_path() -> Path:
    local_yaml = Path.cwd() / "threadsieve.yaml"
    local_yml = Path.cwd() / "threadsieve.yml"
    if local_yaml.exists():
        return local_yaml
    if local_yml.exists():
        return local_yml
    return default_config_path()


def load_config_text(text: str, path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return parse_simple_yaml(text)
    return json.loads(text)


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


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small threadsieve.yaml shape without adding a runtime dependency."""
    root: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0:
            current_section = None
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                root[key] = parse_yaml_scalar(value)
            else:
                root[key] = {}
                current_section = key
            continue

        if current_section is None:
            continue
        if stripped.startswith("- "):
            if not isinstance(root.get(current_section), list):
                root[current_section] = []
            root[current_section].append(parse_yaml_scalar(stripped[2:].strip()))
            continue
        if ":" in stripped:
            if not isinstance(root.get(current_section), dict):
                root[current_section] = {}
            key, value = stripped.split(":", 1)
            root[current_section][key.strip()] = parse_yaml_scalar(value.strip()) if value.strip() else {}
    return root


def parse_yaml_scalar(value: str) -> Any:
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_yaml_scalar(part.strip()) for part in inner.split(",")]
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
