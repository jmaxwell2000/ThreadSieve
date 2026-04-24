from __future__ import annotations

import hashlib
import re


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def short_hash(value: str, length: int = 12) -> str:
    return sha256_text(value)[:length]


def slugify(value: str, fallback: str = "untitled", max_length: int = 64) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return (slug or fallback)[:max_length].strip("-") or fallback


def stable_thread_id(source_app: str, title: str, content: str) -> str:
    return f"thread_{short_hash(source_app + title + content, 16)}"


def stable_message_id(thread_id: str, index: int, role: str, content: str) -> str:
    return f"msg_{short_hash(thread_id + str(index) + role + content, 16)}"


def stable_item_id(item_type: str, title: str, source_refs: str) -> str:
    return f"{item_type}_{short_hash(item_type + title + source_refs, 12)}"
