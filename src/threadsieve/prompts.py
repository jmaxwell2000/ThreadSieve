from __future__ import annotations

from pathlib import Path

from .config import Config, DEFAULT_CONFIG, default_config_path, expand_path


DEFAULT_EXTRACT_PROMPT = """You are ThreadSieve, a precise conversation-to-knowledge extractor.
Return JSON only. Extract fewer, higher-value knowledge objects.
Every item must be grounded in source_refs using message IDs from the input.
When possible, include exact_text on each source_ref so spans can be repaired if character offsets are wrong.
Do not invent facts that are not supported by cited messages.
Supported item types: idea, decision, open_loop, question, task, draft, product_concept, technical_pattern, research_lead, project_note, framework.
"""


def default_extract_prompt_path(config_path: Path) -> Path:
    return config_path.parent / "prompts" / "extract.md"


def configured_extract_prompt_path(config: Config) -> Path:
    prompts = config.raw.get("prompts") or {}
    configured = prompts.get("extract")
    if configured:
        if str(configured) == DEFAULT_CONFIG["prompts"]["extract"] and config.path != default_config_path():
            return default_extract_prompt_path(config.path)
        return expand_path(str(configured))
    return default_extract_prompt_path(config.path)


def ensure_default_prompt(config: Config) -> Path:
    path = configured_extract_prompt_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(DEFAULT_EXTRACT_PROMPT, encoding="utf-8")
    return path


def load_extract_prompt(config: Config) -> str:
    path = configured_extract_prompt_path(config)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return DEFAULT_EXTRACT_PROMPT
