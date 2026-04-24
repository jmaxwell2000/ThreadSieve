from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from .ids import stable_item_id
from .models import KnowledgeItem, SourceRef, Thread
from .prompts import DEFAULT_EXTRACT_PROMPT
from .providers import build_provider, fetch_json, provider_request


def extract_items(thread: Thread, model_config: dict[str, Any], threshold: float, system_prompt: str | None = None) -> list[KnowledgeItem]:
    provider = build_provider(model_config)
    if provider.kind == "offline" or provider.name in {"offline", "none", "heuristic"}:
        raw_items = offline_extract(thread)
    else:
        raw_items = openai_compatible_extract(thread, model_config, system_prompt or DEFAULT_EXTRACT_PROMPT)
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


def openai_compatible_extract(thread: Thread, model_config: dict[str, Any], system_prompt: str = DEFAULT_EXTRACT_PROMPT) -> list[dict[str, Any]]:
    provider = build_provider(model_config)
    request = provider_request(
        provider,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(extraction_payload(thread), ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )
    response_data = fetch_json(request, timeout=provider.timeout_seconds)

    content = response_data["choices"][0]["message"]["content"]
    parsed = parse_model_json(content)
    if isinstance(parsed, dict):
        return list(parsed.get("items") or [])
    raise RuntimeError("Model returned JSON, but not the expected object shape.")

def parse_model_json(content: str) -> Any:
    """Parse JSON from common chat-model wrappers without accepting arbitrary prose as data."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.IGNORECASE | re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", content):
        try:
            parsed, _ = decoder.raw_decode(content[match.start() :])
        except json.JSONDecodeError:
            continue
        return parsed
    preview = " ".join(content.split())[:160]
    raise RuntimeError(f"Model did not return parseable JSON. Response began: {preview}")


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
                    "source_refs": [{"message_id": "msg_id", "start_char": 0, "end_char": 120, "exact_text": "cited text"}],
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
            start, end = repair_span(message.content, ref, source_ref.start_char, source_ref.end_char)
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


def repair_span(content: str, raw_ref: dict[str, Any], start_char: int, end_char: int) -> tuple[int, int]:
    quote = first_text(raw_ref, "exact_text", "text", "quote", "excerpt")
    if quote:
        exact = content.find(quote)
        if exact >= 0:
            return exact, exact + len(quote)
        fuzzy = fuzzy_span(content, quote)
        if fuzzy:
            return fuzzy

    start = max(0, min(start_char, len(content)))
    end = max(start, min(end_char, len(content)))
    return start, end


def first_text(raw: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def fuzzy_span(content: str, quote: str) -> tuple[int, int] | None:
    normalized_content, position_map = normalize_with_positions(content)
    normalized_quote, _ = normalize_with_positions(quote)
    if not normalized_content or not normalized_quote:
        return None

    exact = normalized_content.find(normalized_quote)
    if exact >= 0:
        return position_map[exact], position_map[exact + len(normalized_quote) - 1] + 1

    quote_len = len(normalized_quote)
    best: tuple[float, int, int] | None = None
    step = max(1, quote_len // 8)
    min_len = max(8, int(quote_len * 0.65))
    max_len = max(min_len, int(quote_len * 1.35))
    for window_len in range(min_len, max_len + 1, step):
        for start in range(0, max(1, len(normalized_content) - window_len + 1), step):
            window = normalized_content[start : start + window_len]
            ratio = SequenceMatcher(None, normalized_quote, window).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, start, start + window_len)

    if best and best[0] >= 0.72:
        _, start, end = best
        return position_map[start], position_map[end - 1] + 1
    return None


def normalize_with_positions(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    previous_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if chars and not previous_space:
                chars.append(" ")
                positions.append(index)
            previous_space = True
            continue
        chars.append(char.lower())
        positions.append(index)
        previous_space = False
    if chars and chars[-1] == " ":
        chars.pop()
        positions.pop()
    return "".join(chars), positions


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
