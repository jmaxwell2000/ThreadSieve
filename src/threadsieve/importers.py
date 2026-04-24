from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .ids import stable_message_id, stable_thread_id
from .models import Message, Thread, utc_now_iso


ROLE_PATTERN = re.compile(r"^(user|assistant|system|human|ai|you|me|chatgpt|claude)\s*:\s*(.*)$", re.I)


def import_file(path: Path, source_app: str | None = None) -> Thread:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
        return import_json(raw, source_app=source_app or infer_source_app(path, raw), source_uri=str(path))
    return import_text(text, title=path.stem, source_app=source_app or "file", source_uri=str(path))


def import_text(text: str, title: str = "Pasted thread", source_app: str = "text", source_uri: str | None = None) -> Thread:
    parsed = parse_role_blocks(text)
    if not parsed:
        parsed = [{"role": "user", "content": text.strip()}]
    thread_id = stable_thread_id(source_app, title, text)
    messages = [
        Message(
            id=stable_message_id(thread_id, index, item["role"], item["content"]),
            thread_id=thread_id,
            role=item["role"],
            index=index,
            content=item["content"],
        )
        for index, item in enumerate(parsed)
        if item["content"].strip()
    ]
    return Thread(
        id=thread_id,
        source_app=source_app,
        source_uri=source_uri,
        title=title,
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        messages=messages,
    )


def import_json(raw: Any, source_app: str = "json", source_uri: str | None = None) -> Thread:
    if isinstance(raw, list):
        if raw and all(isinstance(item, dict) and "mapping" in item for item in raw):
            raw = raw[0]
        elif raw and all(isinstance(item, dict) and ("role" in item or "author" in item) for item in raw):
            return import_message_list(raw, source_app=source_app, source_uri=source_uri)

    if isinstance(raw, dict) and "mapping" in raw:
        return import_chatgpt_conversation(raw, source_uri=source_uri)

    if isinstance(raw, dict) and isinstance(raw.get("messages"), list):
        title = str(raw.get("title") or "JSON thread")
        thread_id = str(raw.get("id") or stable_thread_id(source_app, title, json.dumps(raw, sort_keys=True)))
        messages = build_messages(thread_id, raw["messages"])
        return Thread(
            id=thread_id,
            source_app=str(raw.get("source_app") or source_app),
            source_uri=source_uri or raw.get("source_uri"),
            title=title,
            participants=list(raw.get("participants") or []),
            created_at=string_or_none(raw.get("created_at")),
            updated_at=string_or_none(raw.get("updated_at")),
            messages=messages,
            metadata={key: value for key, value in raw.items() if key not in {"messages", "title"}},
        )

    return import_text(json.dumps(raw, indent=2), title="JSON thread", source_app=source_app, source_uri=source_uri)


def import_message_list(raw_messages: list[dict[str, Any]], source_app: str, source_uri: str | None) -> Thread:
    title = "Message list"
    content = json.dumps(raw_messages, sort_keys=True)
    thread_id = stable_thread_id(source_app, title, content)
    return Thread(
        id=thread_id,
        source_app=source_app,
        source_uri=source_uri,
        title=title,
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        messages=build_messages(thread_id, raw_messages),
    )


def import_chatgpt_conversation(raw: dict[str, Any], source_uri: str | None = None) -> Thread:
    title = str(raw.get("title") or "ChatGPT conversation")
    thread_id = f"chatgpt_{raw.get('id') or stable_thread_id('chatgpt', title, json.dumps(raw, sort_keys=True))}"
    mapping = raw.get("mapping") or {}
    nodes = []
    for node in mapping.values():
        message = node.get("message") if isinstance(node, dict) else None
        if not message:
            continue
        content = extract_chatgpt_content(message.get("content") or {})
        if not content.strip():
            continue
        author = message.get("author") or {}
        role = normalize_role(str(author.get("role") or "unknown"))
        create_time = message.get("create_time")
        nodes.append(
            {
                "role": role,
                "content": content,
                "timestamp": str(create_time) if create_time is not None else None,
                "metadata": {"source_message_id": message.get("id")},
            }
        )
    messages = [
        Message(
            id=stable_message_id(thread_id, index, node["role"], node["content"]),
            thread_id=thread_id,
            role=node["role"],
            index=index,
            content=node["content"],
            timestamp=node["timestamp"],
            metadata=node["metadata"],
        )
        for index, node in enumerate(nodes)
    ]
    return Thread(
        id=thread_id,
        source_app="chatgpt",
        source_uri=source_uri,
        title=title,
        created_at=string_or_none(raw.get("create_time")),
        updated_at=string_or_none(raw.get("update_time")),
        messages=messages,
        metadata={"conversation_id": raw.get("id")},
    )


def build_messages(thread_id: str, raw_messages: list[dict[str, Any]]) -> list[Message]:
    messages: list[Message] = []
    for index, raw in enumerate(raw_messages):
        role = normalize_role(str(raw.get("role") or raw.get("author") or raw.get("role_or_author") or "unknown"))
        content = str(raw.get("content") or raw.get("text") or raw.get("body") or "")
        if not content.strip():
            continue
        messages.append(
            Message(
                id=str(raw.get("id") or stable_message_id(thread_id, index, role, content)),
                thread_id=thread_id,
                role=role,
                index=index,
                content=content,
                timestamp=string_or_none(raw.get("timestamp")),
                attachments=list(raw.get("attachments") or []),
                metadata=dict(raw.get("metadata") or {}),
            )
        )
    return messages


def parse_role_blocks(text: str) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    current_role: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        match = ROLE_PATTERN.match(line)
        if match:
            if current_role and current_lines:
                blocks.append({"role": normalize_role(current_role), "content": "\n".join(current_lines).strip()})
            current_role = match.group(1)
            current_lines = [match.group(2)] if match.group(2) else []
        elif current_role:
            current_lines.append(line)
    if current_role and current_lines:
        blocks.append({"role": normalize_role(current_role), "content": "\n".join(current_lines).strip()})
    return [block for block in blocks if block["content"]]


def extract_chatgpt_content(content: dict[str, Any]) -> str:
    parts = content.get("parts")
    if isinstance(parts, list):
        clean_parts = []
        for part in parts:
            if isinstance(part, str):
                clean_parts.append(part)
            elif isinstance(part, dict):
                clean_parts.append(json.dumps(part, sort_keys=True))
        return "\n\n".join(clean_parts)
    text = content.get("text")
    return str(text or "")


def infer_source_app(path: Path, raw: Any) -> str:
    if isinstance(raw, dict) and "mapping" in raw:
        return "chatgpt"
    lower = path.name.lower()
    if "chatgpt" in lower:
        return "chatgpt"
    if "claude" in lower:
        return "claude"
    return "json"


def normalize_role(role: str) -> str:
    role = role.strip().lower()
    return {"human": "user", "me": "user", "ai": "assistant", "chatgpt": "assistant", "claude": "assistant"}.get(role, role)


def string_or_none(value: Any) -> str | None:
    return None if value is None else str(value)
