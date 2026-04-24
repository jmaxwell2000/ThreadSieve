from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class Message:
    id: str
    thread_id: str
    role: str
    index: int
    content: str
    timestamp: str | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "role": self.role,
            "index": self.index,
            "content": self.content,
            "timestamp": self.timestamp,
            "attachments": self.attachments,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Thread:
    id: str
    source_app: str
    title: str
    messages: list[Message]
    source_uri: str | None = None
    participants: list[str] = field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_app": self.source_app,
            "source_uri": self.source_uri,
            "title": self.title,
            "participants": self.participants,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [message.to_dict() for message in self.messages],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SourceRef:
    message_id: str
    start_char: int
    end_char: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SourceRef":
        return cls(
            message_id=str(raw.get("message_id", "")),
            start_char=int(raw.get("start_char", 0)),
            end_char=int(raw.get("end_char", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass(frozen=True)
class KnowledgeItem:
    id: str
    type: str
    title: str
    summary: str
    tags: list[str]
    source_refs: list[SourceRef]
    confidence: float
    body: str | None = None
    status: str = "raw"
    created_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], item_id: str) -> "KnowledgeItem":
        refs = raw.get("source_refs") or []
        body = raw.get("body") or raw.get("details") or raw.get("content")
        return cls(
            id=item_id,
            type=normalize_type(str(raw.get("type", "idea"))),
            title=clean_title(str(raw.get("title", "Untitled"))),
            summary=str(raw.get("summary", "")).strip(),
            tags=normalize_tags(raw.get("tags") or []),
            source_refs=[SourceRef.from_dict(ref) for ref in refs if isinstance(ref, dict)],
            confidence=float(raw.get("confidence", 0.0)),
            body=str(body).strip() if body else None,
            status=str(raw.get("status", "raw")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "summary": self.summary,
            "tags": self.tags,
            "source_refs": [ref.to_dict() for ref in self.source_refs],
            "confidence": self.confidence,
            "body": self.body,
            "status": self.status,
            "created_at": self.created_at,
        }


TYPE_DIRS = {
    "idea": "Ideas",
    "decision": "Decisions",
    "open_loop": "Open Loops",
    "question": "Questions",
    "task": "Tasks",
    "draft": "Drafts",
    "product_concept": "Product Concepts",
    "technical_pattern": "Technical Patterns",
    "research_lead": "Research Leads",
    "project_note": "Project Notes",
    "framework": "Frameworks",
}


def normalize_type(value: str) -> str:
    value = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "openloop": "open_loop",
        "todo": "task",
        "pattern": "technical_pattern",
        "research": "research_lead",
        "note": "project_note",
    }
    return aliases.get(value, value if value in TYPE_DIRS else "idea")


def normalize_tags(raw_tags: list[Any]) -> list[str]:
    tags: list[str] = []
    for tag in raw_tags:
        normalized = str(tag).strip().lower().replace(" ", "-").replace("_", "-")
        normalized = "".join(ch for ch in normalized if ch.isalnum() or ch == "-").strip("-")
        if normalized and normalized not in tags:
            tags.append(normalized)
    return tags[:12]


def clean_title(title: str) -> str:
    title = " ".join(title.split())
    return title[:120] if title else "Untitled"
