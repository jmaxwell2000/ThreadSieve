from __future__ import annotations

import json
from pathlib import Path

from .ids import slugify
from .models import KnowledgeItem, TYPE_DIRS, Thread


def write_item(workspace: Path, item: KnowledgeItem, thread: Thread, source_dir: Path) -> Path:
    directory = workspace / "Knowledge" / TYPE_DIRS.get(item.type, "Ideas")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{item.id}-{slugify(item.title)}.md"
    path.write_text(render_item_markdown(item, thread, source_dir), encoding="utf-8")
    return path


def append_jsonl(workspace: Path, item: KnowledgeItem, thread: Thread, path: Path) -> None:
    archive_path = workspace / "objects.jsonl"
    record = item.to_dict()
    record["source_thread_id"] = thread.id
    record["local_path"] = str(path)
    with archive_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def render_item_markdown(item: KnowledgeItem, thread: Thread, source_dir: Path) -> str:
    frontmatter = {
        "id": item.id,
        "type": item.type,
        "title": item.title,
        "created_at": item.created_at,
        "status": item.status,
        "confidence": item.confidence,
        "tags": item.tags,
        "source": {
            "app": thread.source_app,
            "thread_id": thread.id,
            "thread_title": thread.title,
            "local_thread_path": str(source_dir / "thread.md"),
            "source_url": thread.source_uri,
            "message_refs": [ref.to_dict() for ref in item.source_refs],
        },
    }
    lines = ["---", to_yaml_like(frontmatter).rstrip(), "---", "", f"# {item.title}", ""]
    lines.extend(["## Summary", "", item.summary.strip() or "No summary provided.", ""])
    if item.body:
        lines.extend(["## Details", "", item.body.strip(), ""])
    lines.extend(["## Source", "", f"- Thread: [{thread.title}]({source_dir / 'thread.md'})"])
    for ref in item.source_refs:
        message = next((message for message in thread.messages if message.id == ref.message_id), None)
        label = f"{ref.message_id}:{ref.start_char}-{ref.end_char}"
        if message:
            excerpt = message.content[ref.start_char : ref.end_char].strip()
            lines.extend(["", f"### `{label}`", "", blockquote(excerpt[:1200] or message.content[:1200])])
        else:
            lines.append(f"- `{label}`")
    return "\n".join(lines).rstrip() + "\n"


def to_yaml_like(value: object, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(to_yaml_like(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {format_scalar(item)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        if not value:
            return f"{pad}[]\n"
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(to_yaml_like(item, indent + 2))
            else:
                lines.append(f"{pad}- {format_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{pad}{format_scalar(value)}\n"


def format_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace('"', '\\"')
    if not text or any(ch in text for ch in [":", "#", "[", "]", "{", "}", "\n"]):
        return f'"{text}"'
    return text


def blockquote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())
