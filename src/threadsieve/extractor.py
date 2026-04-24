from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Any

from .ids import stable_item_id
from .models import KnowledgeItem, SourceRef, Thread


EXTRACTION_SYSTEM_PROMPT = """You are ThreadSieve, a precise conversation-to-knowledge extractor.
Return JSON only. Extract fewer, higher-value knowledge objects.
Every item must be grounded in source_refs using message IDs from the input.
Do not invent facts that are not supported by cited messages.
Supported item types: idea, decision, open_loop, question, task, draft, product_concept, technical_pattern, research_lead, project_note, framework.
"""


def extract_items(thread: Thread, model_config: dict[str, Any], threshold: float) -> list[KnowledgeItem]:
    provider = str(model_config.get("provider") or "offline").lower()
    if provider in {"offline", "none", "heuristic"}:
        raw_items = offline_extract(thread)
    else:
        raw_items = openai_compatible_extract(thread, model_config)
    return validate_items(thread, raw_items, threshold)


def offline_extract(thread: Thread) -> list[dict[str, Any]]:
    """Deterministic local fallback so the CLI works before a model is configured."""
    text = "\n\n".join(f"{message.role}: {message.content}" for message in thread.messages)
    tags = infer_tags(text)
    items: list[dict[str, Any]] = []
    if thread.messages:
        first_ref = {
            "message_id": thread.messages[0].id,
            "start_char": 0,
            "end_char": min(len(thread.messages[0].content), 1200),
        }
        items.append(
            {
                "type": "project_note",
                "title": thread.title,
                "summary": summarize_text(text),
                "body": "Offline extraction captured this as a source-linked project note. Configure an OpenAI-compatible provider for richer typed extraction.",
                "tags": tags,
                "source_refs": [first_ref],
                "confidence": 0.62,
            }
        )
    questions = find_question_items(thread, tags)
    items.extend(questions[:5])
    return items


def openai_compatible_extract(thread: Thread, model_config: dict[str, Any]) -> list[dict[str, Any]]:
    base_url = str(model_config.get("base_url") or "").rstrip("/")
    model = str(model_config.get("model") or "")
    api_key_env = str(model_config.get("api_key_env") or "THREADSIEVE_API_KEY")
    api_key = os.environ.get(api_key_env) or str(model_config.get("api_key") or "")
    if not base_url or not model:
        raise RuntimeError("Model config must include base_url and model.")

    payload = {
        "model": model,
        "temperature": float(model_config.get("temperature", 0.1)),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(extraction_payload(thread), ensure_ascii=False)},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "Bearer no-key",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(model_config.get("timeout_seconds", 120))) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Model request failed: {exc}") from exc

    content = response_data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if isinstance(parsed, dict):
        return list(parsed.get("items") or [])
    raise RuntimeError("Model returned JSON, but not the expected object shape.")


def extraction_payload(thread: Thread) -> dict[str, Any]:
    return {
        "schema": {
            "items": [
                {
                    "type": "idea | decision | open_loop | question | task | draft | product_concept | technical_pattern | research_lead | project_note | framework",
                    "title": "short durable title",
                    "summary": "grounded summary",
                    "body": "optional useful detail",
                    "tags": ["stable", "reusable", "tags"],
                    "source_refs": [{"message_id": "msg_id", "start_char": 0, "end_char": 120}],
                    "confidence": 0.0,
                }
            ]
        },
        "thread": {
            "id": thread.id,
            "source_app": thread.source_app,
            "title": thread.title,
            "messages": [
                {"id": message.id, "role": message.role, "index": message.index, "content": message.content}
                for message in thread.messages
            ],
        },
    }


def validate_items(thread: Thread, raw_items: list[dict[str, Any]], threshold: float) -> list[KnowledgeItem]:
    message_by_id = {message.id: message for message in thread.messages}
    items: list[KnowledgeItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        refs = []
        for ref in raw.get("source_refs") or []:
            if not isinstance(ref, dict):
                continue
            source_ref = SourceRef.from_dict(ref)
            message = message_by_id.get(source_ref.message_id)
            if not message:
                continue
            start = max(0, min(source_ref.start_char, len(message.content)))
            end = max(start, min(source_ref.end_char, len(message.content)))
            refs.append({"message_id": source_ref.message_id, "start_char": start, "end_char": end})
        if not refs:
            continue
        raw = dict(raw)
        raw["source_refs"] = refs
        source_key = json.dumps(refs, sort_keys=True)
        item_type = str(raw.get("type", "idea"))
        title = str(raw.get("title", "Untitled"))
        item = KnowledgeItem.from_dict(raw, stable_item_id(item_type, title, source_key))
        if item.confidence >= threshold:
            items.append(item)
    return dedupe_items(items)


def dedupe_items(items: list[KnowledgeItem]) -> list[KnowledgeItem]:
    seen: set[tuple[str, str]] = set()
    unique: list[KnowledgeItem] = []
    for item in items:
        key = (item.type, item.title.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def summarize_text(text: str, max_chars: int = 420) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    cutoff = compact.rfind(" ", 0, max_chars)
    return compact[: cutoff if cutoff > 120 else max_chars].rstrip() + "..."


def infer_tags(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{3,}", text.lower())
    stop = {
        "that",
        "this",
        "with",
        "from",
        "have",
        "should",
        "would",
        "could",
        "there",
        "their",
        "about",
        "thread",
        "conversation",
        "messages",
        "user",
        "assistant",
    }
    counts = Counter(word for word in words if word not in stop)
    return [word for word, _ in counts.most_common(8)] or ["thread"]


def find_question_items(thread: Thread, fallback_tags: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in thread.messages:
        for match in re.finditer(r"([^?\n]{12,}\?)", message.content):
            question = " ".join(match.group(1).split())
            items.append(
                {
                    "type": "question",
                    "title": question[:90],
                    "summary": question,
                    "tags": fallback_tags[:5],
                    "source_refs": [
                        {
                            "message_id": message.id,
                            "start_char": match.start(1),
                            "end_char": match.end(1),
                        }
                    ],
                    "confidence": 0.68,
                }
            )
    return items
