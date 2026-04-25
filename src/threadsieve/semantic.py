from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ids import short_hash, slugify
from .models import Message, Thread, utc_now_iso
from .prompts import DEFAULT_SEMANTIC_PROMPT
from .providers import build_provider, fetch_json, provider_request, response_message_content


EXTRACTION_FROM_SEMANTIC_LOG_PROMPT = """ThreadSieve extraction priority:
You are reading a semantic log, not a raw transcript.
Treat USER_STATEMENT content as primary evidence and the user's evolving thought process as the main object of extraction.
Treat AI_CONTEXT content as scaffolding only: it explains what the user was reacting to.
Treat AI_ARTIFACT content as a revisable artifact only when the user accepts, rejects, edits, extends, or refers to it.
Prefer user-originated ideas, questions, requirements, decisions, and tasks.
Do not extract an assistant-originated idea unless the user explicitly adopts, modifies, questions, or builds on it.
When citing source_refs, prefer user message IDs. Use assistant message IDs only when the assistant context itself is essential evidence.
Set origin to user when the durable object comes from the user's words, mixed when the user develops an assistant-introduced concept, assistant only when the saved object is intentionally about an assistant contribution.
When consecutive user turns refine the same artifact, preference, requirement, or conceptual object, merge them into one higher-value item with multiple source_refs.
If a user edits an AI_ARTIFACT, extract the resulting user-directed specification rather than the isolated edit instruction.
"""


@dataclass(frozen=True)
class SemanticLog:
    text: str
    extraction_thread: Thread


def build_semantic_log(thread: Thread, model_config: dict[str, Any], semantic_prompt: str | None = None) -> SemanticLog:
    provider = build_provider(model_config)
    if provider.kind == "offline" or provider.name in {"offline", "none", "heuristic"}:
        return offline_semantic_log(thread)
    text = model_semantic_log(thread, model_config, semantic_prompt or DEFAULT_SEMANTIC_PROMPT)
    return SemanticLog(text=text, extraction_thread=thread_from_semantic_text(thread, text))


def offline_semantic_log(thread: Thread) -> SemanticLog:
    lines = semantic_header(thread)
    transformed_messages: list[Message] = []
    for index, message in enumerate(thread.messages):
        if message.role == "user":
            lines.extend([f"## {message.id} USER_STATEMENT", "", f"USER_STATEMENT: {message.content.strip()}", ""])
            transformed_messages.append(message)
            continue

        next_user = next_user_message(thread.messages, index)
        next_ref = next_user.id if next_user else "none"
        reaction = infer_reaction(next_user)
        if looks_like_artifact(message.content, next_user):
            context = (
                f"- ARTIFACT_TYPE: {infer_artifact_type(message.content)}\n"
                f"- ACTION: {infer_artifact_action(message.content)}\n"
                f"- CONTENT_EXCERPT: {artifact_excerpt(message.content)}\n"
                f"- CONTENT_HASH: {short_hash(message.content, 12)}\n"
                f"- NEXT_USER_REF: {next_ref}\n"
                f"- NEXT_USER_REACTION: {reaction}"
            )
            label = "AI_ARTIFACT"
        else:
            action = infer_action(message.content)
            concepts = ", ".join(infer_concepts(message.content))
            context = (
                f"- ACTION: {action}\n"
                f"- CONCEPTS_INTRODUCED: {concepts}\n"
                f"- NEXT_USER_REF: {next_ref}\n"
                f"- NEXT_USER_REACTION: {reaction}"
            )
            label = "AI_CONTEXT"
        lines.extend([f"## {message.id} {label}", "", f"{label}:", context, ""])
        transformed_messages.append(
            Message(
                id=message.id,
                thread_id=message.thread_id,
                role=message.role,
                index=message.index,
                content=context,
                timestamp=message.timestamp,
                attachments=message.attachments,
                metadata={**message.metadata, "semantic_context": True, "semantic_artifact": label == "AI_ARTIFACT"},
            )
        )
    return SemanticLog(text="\n".join(lines).rstrip() + "\n", extraction_thread=replace_thread_messages(thread, transformed_messages))


def model_semantic_log(thread: Thread, model_config: dict[str, Any], semantic_prompt: str = DEFAULT_SEMANTIC_PROMPT) -> str:
    provider = build_provider(model_config)
    payload = {
        "thread": {
            "id": thread.id,
            "title": thread.title,
            "messages": [
                {"id": message.id, "role": message.role, "timestamp": message.timestamp, "content": message.content}
                for message in thread.messages
            ],
        }
    }
    request = provider_request(
        provider,
        messages=[
            {"role": "system", "content": semantic_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    response_data = fetch_json(request, timeout=provider.timeout_seconds)
    content = response_message_content(response_data, "semantic log request")
    return normalize_semantic_log_text(thread, content)


def normalize_semantic_log_text(thread: Thread, text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("# "):
        return "\n".join(semantic_header(thread) + ["", stripped]).rstrip() + "\n"
    return stripped + "\n"


def thread_from_semantic_text(original: Thread, semantic_text: str) -> Thread:
    transformed = []
    original_by_id = {message.id: message for message in original.messages}
    blocks = re.split(r"^##\s+(\S+)\s+(USER_STATEMENT|AI_CONTEXT|AI_ARTIFACT)\s*$", semantic_text, flags=re.MULTILINE)
    if len(blocks) < 4:
        return offline_semantic_log(original).extraction_thread
    for index in range(1, len(blocks), 3):
        message_id = blocks[index]
        label = blocks[index + 1]
        body = blocks[index + 2].strip()
        original_message = original_by_id.get(message_id)
        if not original_message:
            continue
        if label == "USER_STATEMENT":
            content = original_message.content
        else:
            content = body.replace(f"{label}:", "", 1).strip() or original_message.content
        transformed.append(
            Message(
                id=original_message.id,
                thread_id=original_message.thread_id,
                role=original_message.role,
                index=original_message.index,
                content=content,
                timestamp=original_message.timestamp,
                attachments=original_message.attachments,
                metadata={
                    **original_message.metadata,
                    "semantic_context": label in {"AI_CONTEXT", "AI_ARTIFACT"},
                    "semantic_artifact": label == "AI_ARTIFACT",
                },
            )
        )
    return replace_thread_messages(original, transformed or original.messages)


def replace_thread_messages(thread: Thread, messages: list[Message]) -> Thread:
    return Thread(
        id=thread.id,
        source_app=thread.source_app,
        title=thread.title,
        messages=messages,
        source_uri=thread.source_uri,
        participants=thread.participants,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
        metadata={**thread.metadata, "semantic_log_generated_at": utc_now_iso()},
    )


def semantic_header(thread: Thread) -> list[str]:
    lines = [
        f"# Semantic Log: {thread.title}",
        "",
        f"- Thread ID: {thread.id}",
        f"- Source app: {thread.source_app}",
    ]
    if thread.source_uri:
        lines.append(f"- Source URI: {thread.source_uri}")
    lines.append("")
    return lines


def write_semantic_log(directory: Path, thread: Thread, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{thread.id}-{slugify(thread.title)}.md"
    path.write_text(text, encoding="utf-8")
    return path


def next_user_message(messages: list[Message], index: int) -> Message | None:
    for message in messages[index + 1 :]:
        if message.role == "user":
            return message
    return None


def infer_action(content: str) -> str:
    lower = content.lower()
    if "?" in content:
        return "Asked question"
    if any(word in lower for word in ["however", "but", "instead", "risk", "problem"]):
        return "Challenged premise"
    if any(word in lower for word in ["example", "for instance", "such as"]):
        return "Provided examples"
    if any(word in lower for word in ["plan", "step", "should", "recommend"]):
        return "Proposed plan"
    return "Responded"


def infer_concepts(content: str, limit: int = 10) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", content)
    stop = {
        "that",
        "this",
        "with",
        "from",
        "have",
        "would",
        "could",
        "should",
        "there",
        "their",
        "about",
        "your",
        "into",
        "when",
        "then",
        "than",
    }
    concepts: list[str] = []
    for word in words:
        normalized = word.strip().lower()
        if normalized in stop or normalized in concepts:
            continue
        concepts.append(normalized)
        if len(concepts) >= limit:
            break
    return concepts or ["context"]


def infer_reaction(next_user: Message | None) -> str:
    if not next_user:
        return "No following user statement."
    compact = " ".join(next_user.content.split())
    if len(compact) > 140:
        compact = compact[:137].rstrip() + "..."
    return f"Next user responds: {compact}"


def looks_like_artifact(content: str, next_user: Message | None) -> bool:
    lower = content.lower()
    next_lower = (next_user.content.lower() if next_user else "")
    artifact_markers = [
        "```",
        "prompt",
        "schema",
        "draft",
        "requirements",
        "specification",
        "plan:",
        "roadmap",
        "code",
        "yaml",
        "json",
    ]
    revision_markers = [
        "remove",
        "change",
        "revise",
        "edit",
        "add",
        "include",
        "section",
        "make it",
        "instead",
        "version",
        "prompt",
    ]
    has_artifact_shape = any(marker in lower for marker in artifact_markers) or len(content) > 700
    next_revises = any(marker in next_lower for marker in revision_markers)
    return has_artifact_shape and next_revises


def infer_artifact_type(content: str) -> str:
    lower = content.lower()
    if "```" in content or "def " in content or "function " in lower:
        return "code"
    if "schema" in lower or "json" in lower or "yaml" in lower:
        return "schema"
    if "prompt" in lower:
        return "prompt"
    if "plan" in lower or "roadmap" in lower:
        return "plan"
    if re.search(r"^\s*[-*]\s+", content, re.MULTILINE):
        return "list"
    if "draft" in lower:
        return "draft"
    return "other"


def infer_artifact_action(content: str) -> str:
    artifact_type = infer_artifact_type(content)
    return {
        "code": "Produced code",
        "schema": "Produced schema",
        "prompt": "Drafted prompt",
        "plan": "Proposed plan",
        "list": "Listed requirements",
        "draft": "Drafted artifact",
    }.get(artifact_type, "Produced artifact")


def artifact_excerpt(content: str, max_chars: int = 900) -> str:
    compact = " ".join(content.split())
    if len(compact) <= max_chars:
        return compact
    cutoff = compact.rfind(" ", 0, max_chars)
    return compact[: cutoff if cutoff > 240 else max_chars].rstrip() + "..."
