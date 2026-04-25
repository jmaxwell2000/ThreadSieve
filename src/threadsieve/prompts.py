from __future__ import annotations

from pathlib import Path

from .config import Config, DEFAULT_CONFIG, default_config_path, expand_path


DEFAULT_EXTRACT_PROMPT = """You are ThreadSieve, a precise conversation-to-knowledge extractor.
Return JSON only. Extract fewer, higher-value knowledge objects.
Every item must be grounded in source_refs using message IDs from the input.
When possible, include exact_text on each source_ref so spans can be repaired if character offsets are wrong.
Do not invent facts that are not supported by cited messages.
Supported item types: idea, task, decision, question, feature, insight, requirement, risk, open_loop, draft, product_concept, technical_pattern, research_lead, project_note, framework.
Supported object_role values: durable_note, artifact_spec, revision, decision, raw_capture.
Include origin as one of: user, assistant, mixed, unclear.
Include evidence as short source excerpts when useful, not full transcripts.
If the user provides a named protocol, framework, prompt, spec, mode, rubric, or set of directives, extract it as a complete structured artifact. Prefer type framework and object_role artifact_spec unless a more specific supported type clearly fits.
For user-authored protocols/specs/directives, preserve every essential directive in summary, body, canonical_statement, evidence, and source_refs. Do not cite only the title or first lines.
For protocol/framework artifacts, summary must name the protocol's purpose and its major constraints. body must enumerate the essential directives in plain language. canonical_statement must be a compact full specification, not merely a title or generic description.
Prefer the user's evolving thought process over assistant suggestions.
Only save assistant-introduced ideas when the user clearly adopts, modifies, questions, or builds on them.
When consecutive user messages refine the same artifact, preference, requirement, or conceptual object, merge them into one higher-value item with multiple source_refs. Emit separate items only when a later message introduces an independent object, a decision, a task, or a contradiction.
If the user edits or constrains an assistant-generated artifact, extract the user's adopted or modified specification, not merely the individual edit instruction. Include assistant source_refs only when necessary to identify the artifact under revision. Set origin to mixed when the durable object depends on assistant-generated wording but user-directed modifications define its meaning.
Include canonical_statement for durable propositions or specifications.
Include extraction_rationale explaining why the object exists, especially when it merges multiple turns.
"""

DEFAULT_SEMANTIC_PROMPT = """You are converting a conversation into a semantic log for later memory extraction.

Your job:
- Preserve the user's actual words.
- Compress ordinary assistant replies.
- Preserve assistant-generated artifacts when the user is revising or building on them.

Use three block types:

1. USER_STATEMENT
Use this for every user message. Preserve the user's message verbatim.

2. AI_CONTEXT
Use this for ordinary assistant replies that only provide context, explanation, suggestions, examples, or framing.

3. AI_ARTIFACT
Some assistant messages are artifacts the user is working on. Use AI_ARTIFACT when the assistant produced something like:
- a prompt
- a plan
- a schema
- code
- a checklist
- a product spec
- a requirements list
- a draft document

If the next user message edits, critiques, accepts, rejects, expands, or refers back to the assistant's artifact, use AI_ARTIFACT instead of AI_CONTEXT.
Do not use AI_ARTIFACT merely because the assistant provided examples or a list. If the next user only says "continue", "yes, continue", "more examples", "give me more examples", or "what else", keep the assistant message as AI_CONTEXT because the user is requesting continuation, not revising an artifact.

AI_CONTEXT format:
- ACTION: 1-3 words describing the assistant's conversational move.
- CONCEPTS_INTRODUCED: Keywords only. Specific terminology, facts, or ideas the assistant brought into the conversation.
- NEXT_USER_REF: message_id of the next user message, or none.
- NEXT_USER_REACTION: One sentence fragment explaining what part of the assistant message the user reacts to.

AI_ARTIFACT format:
- ARTIFACT_TYPE: prompt | schema | draft | code | plan | list | product_spec | requirements | other
- ACTION: 1-3 words describing what the assistant produced.
- CONTENT_EXCERPT: Smallest sufficient excerpt needed to understand the user's next move.
- CONTENT_HASH: Stable short hash of the full assistant artifact if available.
- NEXT_USER_REF: message_id of the next user message, or none.
- NEXT_USER_REACTION: Exact explanation of how the next user statement reacts to this artifact.

Return only the semantic log. Keep message IDs in headings exactly as provided.
Do not wrap blocks in Markdown code fences.
Every block must start with a heading in this exact form:
## <message_id> USER_STATEMENT
## <message_id> AI_CONTEXT
## <message_id> AI_ARTIFACT
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
