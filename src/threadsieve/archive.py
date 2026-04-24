from __future__ import annotations

import json
from pathlib import Path

from .ids import slugify
from .models import Thread, utc_now_iso


def archive_thread(workspace: Path, thread: Thread) -> Path:
    source_dir = workspace / "Sources" / slugify(thread.source_app) / f"{thread.id}-{slugify(thread.title)}"
    source_dir.mkdir(parents=True, exist_ok=True)
    thread_json = source_dir / "thread.json"
    thread_md = source_dir / "thread.md"
    manifest = source_dir / "manifest.json"

    with thread_json.open("w", encoding="utf-8") as handle:
        json.dump(thread.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    thread_md.write_text(render_thread_markdown(thread), encoding="utf-8")
    with manifest.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "thread_id": thread.id,
                "source_app": thread.source_app,
                "title": thread.title,
                "archived_at": utc_now_iso(),
                "thread_json": "thread.json",
                "thread_markdown": "thread.md",
                "source_uri": thread.source_uri,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return source_dir


def load_thread_from_archive(source_dir: Path) -> Thread:
    from .importers import import_json

    raw = json.loads((source_dir / "thread.json").read_text(encoding="utf-8"))
    return import_json(raw, source_app=raw.get("source_app") or "archive", source_uri=str(source_dir / "thread.json"))


def render_thread_markdown(thread: Thread) -> str:
    lines = [
        f"# {thread.title}",
        "",
        f"- Source app: {thread.source_app}",
        f"- Thread ID: {thread.id}",
    ]
    if thread.source_uri:
        lines.append(f"- Source URI: {thread.source_uri}")
    lines.append("")
    for message in thread.messages:
        lines.extend(
            [
                f"## {message.index + 1}. {message.role} `{message.id}`",
                "",
                message.content.strip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
