from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from .ids import stable_item_id
from .models import KnowledgeItem, SourceRef, Thread, normalize_type
from .prompts import DEFAULT_EXTRACT_PROMPT
from .providers import build_provider, fetch_json, provider_request, response_message_content


RESPONSE_FORMAT_PROTOCOL_PROMPT = """ThreadSieve protocol requirement:
Return a single valid JSON object with an "items" array. Do not wrap the JSON in Markdown.
"""


def extract_items(
    thread: Thread,
    model_config: dict[str, Any],
    threshold: float,
    system_prompt: str | None = None,
    dropped: Counter[str] | None = None,
) -> list[KnowledgeItem]:
    provider = build_provider(model_config)
    if provider.kind == "offline" or provider.name in {"offline", "none", "heuristic"}:
        raw_items = offline_extract(thread)
    else:
        raw_items = openai_compatible_extract(thread, model_config, system_prompt or DEFAULT_EXTRACT_PROMPT)
    return validate_items(thread, raw_items, threshold, dropped=dropped)


def offline_extract(thread: Thread) -> list[dict[str, Any]]:
    """Deterministic smoke-test fallback.

    Offline mode is intentionally conservative. It should prove that parsing,
    provenance, and file writing work, but it should not pretend to perform
    semantic extraction.
    """
    text = "\n\n".join(f"{message.role}: {message.content}" for message in thread.messages)
    tags = infer_tags(text)
    items: list[dict[str, Any]] = []
    questions = find_question_items(thread, tags)
    items.extend(questions[:5])
    return items


def openai_compatible_extract(thread: Thread, model_config: dict[str, Any], system_prompt: str = DEFAULT_EXTRACT_PROMPT) -> list[dict[str, Any]]:
    provider = build_provider(model_config)
    request = provider_request(
        provider,
        messages=build_extraction_messages(thread, system_prompt),
        response_format={"type": "json_object"},
    )
    response_data = fetch_json(request, timeout=provider.timeout_seconds)

    content = response_message_content(response_data, "extraction request")
    parsed = parse_model_json(content)
    if isinstance(parsed, dict):
        return list(parsed.get("items") or [])
    raise RuntimeError("Model returned JSON, but not the expected object shape.")


def build_extraction_messages(thread: Thread, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": RESPONSE_FORMAT_PROTOCOL_PROMPT},
        {"role": "user", "content": json.dumps(extraction_payload(thread), ensure_ascii=False)},
    ]


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
                    "type": "idea | task | decision | question | feature | insight | requirement | risk | open_loop | draft | product_concept | technical_pattern | research_lead | project_note | framework",
                    "title": "short durable title",
                    "summary": "grounded summary",
                    "body": "optional useful detail",
                    "object_role": "durable_note | artifact_spec | revision | decision | raw_capture",
                    "canonical_statement": "stable proposition or durable specification",
                    "parent_object_id": "optional parent object id or null",
                    "supersedes": ["optional prior item ids"],
                    "extraction_rationale": "why this object exists, especially if merged from multiple turns",
                    "thread_position": {"first_message_index": 0, "last_message_index": 2},
                    "tags": ["stable", "reusable", "tags"],
                    "origin": "user | assistant | mixed | unclear",
                    "evidence": ["short excerpt from the source"],
                    "source_refs": [
                        {
                            "message_id": "msg_id",
                            "start_char": 0,
                            "end_char": 120,
                            "exact_text": "cited text",
                            "ref_type": "evidence | initial_request | revision_instruction | assistant_artifact | adoption | rejection | expansion_requirement",
                        }
                    ],
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


def validate_items(thread: Thread, raw_items: list[dict[str, Any]], threshold: float, dropped: Counter[str] | None = None) -> list[KnowledgeItem]:
    message_by_id = {message.id: message for message in thread.messages}
    items: list[KnowledgeItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            count_drop(dropped, "invalid_candidate")
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
            repaired_ref = {"message_id": source_ref.message_id, "start_char": start, "end_char": end}
            if source_ref.ref_type:
                repaired_ref["ref_type"] = source_ref.ref_type
            refs.append(repaired_ref)
        if not refs:
            refs = repair_missing_source_refs(thread, raw)
        if not refs:
            count_drop(dropped, "missing_source_refs")
            continue
        if assistant_context_only(thread, refs):
            count_drop(dropped, "assistant_context_only")
            continue
        if assistant_context_with_only_example_requests(thread, refs):
            count_drop(dropped, "assistant_example_context")
            continue
        raw = dict(raw)
        raw["source_refs"] = refs
        raw = normalize_artifact_role(thread, raw, refs)
        if dict(raw.get("metadata") or {}).get("artifact_downgraded"):
            count_drop(dropped, "unsupported_artifact_role")
        raw = strengthen_framework_artifact(thread, raw, refs)
        source_key = json.dumps(refs, sort_keys=True)
        item_type = normalize_type(str(raw.get("type", "idea")))
        title = str(raw.get("title", "Untitled"))
        item = KnowledgeItem.from_dict(raw, stable_item_id(item_type, title, source_key))
        if item.confidence >= threshold:
            items.append(item)
        else:
            count_drop(dropped, "below_threshold")
    return dedupe_items(items)


def count_drop(dropped: Counter[str] | None, reason: str) -> None:
    if dropped is not None:
        dropped[reason] += 1


def repair_missing_source_refs(thread: Thread, raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Recover provenance when a model cites evidence but omits source_refs.

    The prompt requires source_refs, but smaller or stricter JSON-mode models may
    return only message IDs or evidence snippets. Accepting those without repair
    would break ThreadSieve's provenance invariant, so we convert only verifiable
    references into normal source_refs.
    """
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()

    for message_id in candidate_message_ids(raw):
        message = next((message for message in thread.messages if message.id == message_id), None)
        if not message:
            continue
        key = (message.id, 0, len(message.content))
        if key in seen:
            continue
        seen.add(key)
        refs.append(
            {
                "message_id": message.id,
                "start_char": 0,
                "end_char": len(message.content),
                "ref_type": "evidence",
            }
        )

    for quote in candidate_evidence_quotes(raw):
        repaired = ref_from_evidence_quote(thread, quote)
        if not repaired:
            continue
        key = (str(repaired["message_id"]), int(repaired["start_char"]), int(repaired["end_char"]))
        if key in seen:
            continue
        seen.add(key)
        refs.append(repaired)

    return refs


def candidate_message_ids(raw: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("source_message_ids", "message_ids", "source_messages", "messages"):
        value = raw.get(key)
        if isinstance(value, str):
            ids.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    ids.append(item)
                elif isinstance(item, dict):
                    text = first_text(item, "message_id", "id")
                    if text:
                        ids.append(text)
    return ids


def candidate_evidence_quotes(raw: dict[str, Any]) -> list[str]:
    quotes: list[str] = []
    for key in ("evidence", "exact_text", "quote", "excerpt"):
        value = raw.get(key)
        if isinstance(value, str):
            quotes.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    quotes.append(item)
                elif isinstance(item, dict):
                    text = first_text(item, "exact_text", "text", "quote", "excerpt")
                    if text:
                        quotes.append(text)
    return [quote for quote in quotes if usable_evidence_quote(quote)]


def usable_evidence_quote(quote: str) -> bool:
    compact = " ".join(quote.split())
    if len(compact) < 12:
        return False
    if re.fullmatch(r"msg_[A-Za-z0-9_]+", compact):
        return False
    return True


def ref_from_evidence_quote(thread: Thread, quote: str) -> dict[str, Any] | None:
    user_messages = [message for message in thread.messages if message.role == "user"]
    other_messages = [message for message in thread.messages if message.role != "user"]
    for message in user_messages + other_messages:
        span = repair_span(message.content, {"exact_text": quote}, 0, 0)
        if span == (0, 0):
            continue
        start, end = span
        return {
            "message_id": message.id,
            "start_char": start,
            "end_char": end,
            "ref_type": "evidence",
        }
    return None


def strengthen_framework_artifact(thread: Thread, raw: dict[str, Any], refs: list[dict[str, Any]]) -> dict[str, Any]:
    item_type = str(raw.get("type", "")).strip().lower()
    object_role = str(raw.get("object_role", "")).strip().lower()
    if item_type != "framework" and object_role != "artifact_spec":
        return raw

    source_text = referenced_user_text(thread, refs)
    directives = extract_directives(source_text)
    if not directives:
        return raw

    title = str(raw.get("title") or first_nonempty_line(source_text) or "Framework").strip()
    directive_names = [name for name, _ in directives]
    body = "\n".join(f"{index}. {name}: {description}" for index, (name, description) in enumerate(directives, start=1))
    canonical = f"{title} requires " + "; ".join(
        f"{name} ({sentence_fragment(description)})" for name, description in directives
    ) + "."
    summary = (
        f"{title} is a user-authored framework with constraints covering "
        f"{', '.join(directive_names[:-1])}, and {directive_names[-1]}."
        if len(directive_names) > 1
        else f"{title} is a user-authored framework defining {directive_names[0]}."
    )

    strengthened = dict(raw)
    if generic_or_short(strengthened.get("summary"), directive_names, min_chars=120):
        strengthened["summary"] = summary
    if generic_or_short(strengthened.get("body"), directive_names, min_chars=180):
        strengthened["body"] = body
    if generic_or_short(strengthened.get("canonical_statement"), directive_names, min_chars=180):
        strengthened["canonical_statement"] = canonical
    strengthened["object_role"] = "artifact_spec"
    return strengthened


def normalize_artifact_role(thread: Thread, raw: dict[str, Any], refs: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep artifact_spec/framework reserved for actual artifacts.

    Models sometimes over-upgrade ordinary ideas or assistant example lists into
    frameworks. That makes the output look authoritative when it is only context,
    so the validator downgrades unsupported artifact labels before writing.
    """
    item_type = str(raw.get("type", "")).strip().lower()
    object_role = str(raw.get("object_role", "")).strip().lower()
    if item_type != "framework" and object_role != "artifact_spec":
        return raw
    if artifact_spec_supported(thread, refs):
        return raw

    normalized = dict(raw)
    if item_type == "framework":
        normalized["type"] = "idea"
    if object_role == "artifact_spec":
        normalized["object_role"] = "durable_note"
    metadata = dict(normalized.get("metadata") or {})
    metadata["artifact_downgraded"] = True
    metadata["artifact_downgrade_reason"] = "No named/directive artifact or assistant artifact under user revision was cited."
    normalized["metadata"] = metadata
    return normalized


def assistant_context_only(thread: Thread, refs: list[dict[str, Any]]) -> bool:
    """Drop objects grounded only in compressed ordinary assistant context.

    Assistant context explains what the user was reacting to. It should not
    become a saved object unless a user message or AI_ARTIFACT is also cited.
    """
    message_by_id = {message.id: message for message in thread.messages}
    cited_messages = [message_by_id.get(str(ref.get("message_id", ""))) for ref in refs]
    cited_messages = [message for message in cited_messages if message is not None]
    if not cited_messages:
        return False
    return all(
        message.role == "assistant"
        and message.metadata.get("semantic_context")
        and not message.metadata.get("semantic_artifact")
        for message in cited_messages
    )


def assistant_context_with_only_example_requests(thread: Thread, refs: list[dict[str, Any]]) -> bool:
    message_by_id = {message.id: message for message in thread.messages}
    cited_messages = [message_by_id.get(str(ref.get("message_id", ""))) for ref in refs]
    cited_messages = [message for message in cited_messages if message is not None]
    if not cited_messages:
        return False
    assistant_contexts = [
        message
        for message in cited_messages
        if message.role == "assistant" and message.metadata.get("semantic_context") and not message.metadata.get("semantic_artifact")
    ]
    user_messages = [message for message in cited_messages if message.role == "user"]
    if not assistant_contexts or not user_messages:
        return False
    return all(is_example_request_or_continuation(message.content) for message in user_messages)


def is_example_request_or_continuation(text: str) -> bool:
    compact = " ".join(text.lower().strip().strip(".!?").split())
    if compact in {"continue", "yes continue", "yes, continue", "more examples", "give me more examples", "what else"}:
        return True
    if re.fullmatch(r"(yes[, ]+)?(?:continue|more|give me more examples)(?: please)?", compact):
        return True
    example_request = "example" in compact or "examples" in compact
    request_shape = any(phrase in compact for phrase in ["give ", "give me", "show me", "provide", "list", "what else"])
    return example_request and request_shape


def artifact_spec_supported(thread: Thread, refs: list[dict[str, Any]]) -> bool:
    message_by_id = {message.id: message for message in thread.messages}
    for ref in refs:
        message = message_by_id.get(str(ref.get("message_id", "")))
        if not message:
            continue
        if message.metadata.get("semantic_artifact"):
            return True
        if message.role == "user" and looks_like_user_artifact_text(message.content):
            return True
    return False


def looks_like_user_artifact_text(text: str) -> bool:
    lower = text.lower()
    if has_directives_section(text):
        return True
    artifact_terms = [
        "protocol",
        "framework",
        "prompt",
        "specification",
        " spec",
        "mode",
        "rubric",
        "checklist",
        "requirements",
    ]
    if not any(term in lower for term in artifact_terms):
        return False
    first_lines = "\n".join(text.splitlines()[:8])
    has_name_shape = ":" in first_lines or re.search(r"^\s*(?:[-*]|\d+\.)\s+", text, re.MULTILINE)
    return bool(has_name_shape)


def referenced_user_text(thread: Thread, refs: list[dict[str, Any]]) -> str:
    message_by_id = {message.id: message for message in thread.messages}
    chunks = []
    for ref in refs:
        message = message_by_id.get(str(ref.get("message_id", "")))
        if not message or message.role != "user":
            continue
        chunks.append(message.content)
    return "\n\n".join(chunks)


def extract_directives(text: str) -> list[tuple[str, str]]:
    if not text:
        return []
    marker = re.search(r"\bDirectives\s*:\s*", text, flags=re.IGNORECASE)
    if not marker:
        return []
    directive_text = text[marker.end() :]
    pairs: list[tuple[str, str]] = []
    for paragraph in re.split(r"\n\s*\n", directive_text.strip()):
        compact = " ".join(paragraph.split())
        if ":" not in compact:
            continue
        name, description = compact.split(":", 1)
        name = name.strip()
        description = description.strip()
        if not valid_directive_name(name) or not description:
            continue
        pairs.append((name, description))
    return pairs[:12]


def valid_directive_name(name: str) -> bool:
    lowered = name.lower()
    if lowered in {"user", "assistant", "ai", "example", "examples", "directives", "decision"}:
        return False
    if lowered.startswith(("yes.", "no.", "okay.", "ok.", "decision.")):
        return False
    if re.search(r"[.!?]", name):
        return False
    if len(name) < 3 or len(name) > 80:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _/\-()]*", name))


def has_directives_section(text: str) -> bool:
    return bool(re.search(r"\bDirectives\s*:\s*", text, flags=re.IGNORECASE))


def generic_or_short(value: Any, directive_names: list[str], min_chars: int) -> bool:
    if isinstance(value, (list, tuple)):
        return True
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        return True
    if len(text) < min_chars:
        return True
    lowered = text.lower()
    matched = sum(1 for name in directive_names if name.lower() in lowered)
    return matched < min(3, len(directive_names))


def sentence_fragment(text: str) -> str:
    return text.strip().rstrip(".")


def first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        compact = line.strip()
        if compact:
            return compact
    return None


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
        end_index = min(exact + len(normalized_quote) - 1, len(position_map) - 1)
        return position_map[exact], position_map[end_index] + 1

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
        start_index = min(start, len(position_map) - 1)
        end_index = min(max(start_index, end - 1), len(position_map) - 1)
        return position_map[start_index], position_map[end_index] + 1
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
        if message.role != "user":
            continue
        for match in re.finditer(r"([^?\n]{12,}\?)", message.content):
            question = " ".join(match.group(1).split())
            items.append(
                {
                    "type": "question",
                    "title": question[:90],
                    "summary": question,
                    "tags": fallback_tags[:5],
                    "origin": message.role if message.role in {"user", "assistant"} else "unclear",
                    "evidence": [question],
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
