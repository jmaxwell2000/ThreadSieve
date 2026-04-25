from __future__ import annotations

from pathlib import Path

from .config import Config, DEFAULT_CONFIG, default_config_path, expand_path


DEFAULT_EXTRACT_PROMPT = """You are ThreadSieve, a precise conversation-to-knowledge extractor.
Return JSON only. Extract fewer, higher-value knowledge objects.
Every item must be grounded in source_refs using message IDs from the input.
When possible, include exact_text on each source_ref so spans can be repaired if character offsets are wrong.
Do not invent facts that are not supported by cited messages.
Supported item types: idea, task, decision, question, feature, insight, requirement, risk, open_loop, draft, product_concept, technical_pattern, research_lead, project_note, framework.
Include origin as one of: user, assistant, mixed, unclear.
Include evidence as short source excerpts when useful, not full transcripts.
Prefer the user's evolving thought process over assistant suggestions.
Only save assistant-introduced ideas when the user clearly adopts, modifies, questions, or builds on them.
"""

DEFAULT_SEMANTIC_PROMPT = """You are processing a transcript of a conversation to prepare it for long-term semantic memory extraction.
Your goal is to preserve the user's exact words while compressing the AI's responses into dense, high-signal metadata.

Instructions:
1. Preserve every user message verbatim, labeled as USER_STATEMENT:.
2. Replace the AI's messages with a structured block labeled AI_CONTEXT:.
3. Do not write narrative summaries for AI_CONTEXT.
4. Use brief, comma-separated keywords and active verbs.
5. Read the next user message before writing BRIDGE so the user's logical next move is easy to follow.

Format each AI_CONTEXT block strictly as:
- ACTION: 1-3 words describing the rhetorical move the AI made.
- CONCEPTS_INTRODUCED: Keywords only. Specific terminology, facts, or ideas the AI brought into the space.
- BRIDGE: One sentence fragment stating exactly what the next user message is reacting to.

Return only the semantic log. Keep message IDs in headings exactly as provided.
"""

DEFAULT_PROMPTS = {
    "extract": DEFAULT_EXTRACT_PROMPT,
    "semantic": DEFAULT_SEMANTIC_PROMPT,
}


def default_prompt_path(config_path: Path, kind: str) -> Path:
    return config_path.parent / "prompts" / f"{kind}.md"


def configured_prompt_path(config: Config, kind: str = "extract") -> Path:
    validate_prompt_kind(kind)
    prompts = config.raw.get("prompts") or {}
    configured = prompts.get(kind)
    if configured:
        if str(configured) == DEFAULT_CONFIG["prompts"].get(kind) and config.path != default_config_path():
            return default_prompt_path(config.path, kind)
        return expand_path(str(configured))
    return default_prompt_path(config.path, kind)


def ensure_default_prompt(config: Config, kind: str = "extract", force: bool = False) -> Path:
    validate_prompt_kind(kind)
    path = configured_prompt_path(config, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    if force or not path.exists():
        path.write_text(DEFAULT_PROMPTS[kind], encoding="utf-8")
    return path


def load_extract_prompt(config: Config) -> str:
    return load_prompt(config, "extract")


def load_semantic_prompt(config: Config) -> str:
    return load_prompt(config, "semantic")


def load_prompt(config: Config, kind: str) -> str:
    validate_prompt_kind(kind)
    path = configured_prompt_path(config, kind)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return DEFAULT_PROMPTS[kind]


def validate_prompt_kind(kind: str) -> None:
    if kind not in DEFAULT_PROMPTS:
        choices = ", ".join(sorted(DEFAULT_PROMPTS))
        raise RuntimeError(f"Unknown prompt kind {kind!r}. Choose one of: {choices}.")
