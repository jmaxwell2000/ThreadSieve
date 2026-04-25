from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ids import slugify
from .models import Message, Thread, utc_now_iso
from .prompts import DEFAULT_SEMANTIC_PROMPT
from .providers import build_provider, fetch_json, provider_request


EXTRACTION_FROM_SEMANTIC_LOG_PROMPT = """ThreadSieve extraction priority:
You are reading a semantic log, not a raw transcript.
Treat USER_STATEMENT content as primary evidence and the user's evolving thought process as the main object of extraction.
Treat AI_CONTEXT content as scaffolding only: it explains what the user was reacting to.
Prefer user-originated ideas, questions, requirements, decisions, and tasks.
Do not extract an assistant-originated idea unless the user explicitly adopts, modifies, questions, or builds on it.
When citing source_refs, prefer user message IDs. Use assistant message IDs only when the assistant context itself is essential evidence.
Set origin to user when the durable object comes from the user's words, mixed when the user develops an assistant-introduced concept, assistant only when the saved object is intentionally about an assistant contribution.
"""


@dataclass(frozen=True)
class SemanticLog:
    text: str
    extraction_thread: Thread


def build_semantic_log(thread: Thread, model_config: dict[str, Any], semantic_prompt: str | None = None) -> SemanticLog:
    provider = build_provider(model_config)
    if provider.kind == "offline" or provider.name in {"offline", "none", "heuristic"}:
        return offline_semantic_log(thread)
    try:
        text = model_semantic_log(thread, model_config, semantic_prompt or DEFAULT_SEMANTIC_PROMPT)
    except Exception:
        return offline_semantic_log(thread)
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
        action = infer_action(message.content)
        concepts = ", ".join(infer_concepts(message.content))
        bridge = infer_bridge(next_user)
        context = f"- ACTION: {action}\n- CONCEPTS_INTRODUCED: {concepts}\n- BRIDGE: {bridge}"
        lines.extend([f"## {message.id} AI_CONTEXT", "", "AI_CONTEXT:", context, ""])
        transformed_messages.append(
            Message(
                id=message.id,
                thread_id=message.thread_id,
                role=message.role,
                index=message.index,
                content=context,
                timestamp=message.timestamp,
                attachments=message.attachments,
                metadata={**message.metadata, "semantic_context": True},
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
    content = response_data["choices"][0]["message"]["content"]
    return normalize_semantic_log_text(thread, content)


def normalize_semantic_log_text(thread: Thread, text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("# "):
        return "\n".join(semantic_header(thread) + ["", stripped]).rstrip() + "\n"
    return stripped + "\n"


def thread_from_semantic_text(original: Thread, semantic_text: str) -> Thread:
    transformed = []
    original_by_id = {message.id: message for message in original.messages}
    blocks = re.split(r"^##\s+(\S+)\s+(USER_STATEMENT|AI_CONTEXT)\s*$", semantic_text, flags=re.MULTILINE)
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
            content = body.replace("AI_CONTEXT:", "", 1).strip() or original_message.content
        transformed.append(
            Message(
                id=original_message.id,
                thread_id=original_message.thread_id,
                role=original_message.role,
                index=original_message.index,
                content=content,
                timestamp=original_message.timestamp,
                attachments=original_message.attachments,
                metadata={**original_message.metadata, "semantic_context": label == "AI_CONTEXT"},
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


def infer_bridge(next_user: Message | None) -> str:
    if not next_user:
        return "No following user statement."
    compact = " ".join(next_user.content.split())
    if len(compact) > 140:
        compact = compact[:137].rstrip() + "..."
    return f"Next user responds: {compact}"
