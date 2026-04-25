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
    source_id: str | None = None
    source_path: str | None = None
    chat_id: str | None = None
    role: str | None = None
    timestamp: str | None = None
    granularity: str = "message"
    ref_type: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SourceRef":
        return cls(
            message_id=str(raw.get("message_id", "")),
            start_char=int(raw.get("start_char", 0)),
            end_char=int(raw.get("end_char", 0)),
            source_id=string_or_none(raw.get("source_id")),
            source_path=string_or_none(raw.get("source_path")),
            chat_id=string_or_none(raw.get("chat_id")),
            role=string_or_none(raw.get("role")),
            timestamp=string_or_none(raw.get("timestamp")),
            granularity=str(raw.get("granularity") or "message"),
            ref_type=string_or_none(raw.get("ref_type")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "message_id": self.message_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }
        optional = {
            "source_id": self.source_id,
            "source_path": self.source_path,
            "chat_id": self.chat_id,
            "role": self.role,
            "timestamp": self.timestamp,
            "granularity": self.granularity,
            "ref_type": self.ref_type,
        }
        for key, value in optional.items():
            if value is not None:
                data[key] = value
        return data


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
    updated_at: str | None = None
    origin: str = "unclear"
    evidence: list[str] = field(default_factory=list)
    generated_by: dict[str, Any] = field(default_factory=dict)
    object_role: str = "durable_note"
    canonical_statement: str | None = None
    parent_object_id: str | None = None
    supersedes: list[str] = field(default_factory=list)
    extraction_rationale: str | None = None
    thread_position: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

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
            updated_at=string_or_none(raw.get("updated_at")),
            origin=normalize_origin(str(raw.get("origin", "unclear"))),
            evidence=normalize_evidence(raw.get("evidence") or []),
            generated_by=dict(raw.get("generated_by") or {}),
            object_role=normalize_object_role(str(raw.get("object_role", "durable_note"))),
            canonical_statement=string_or_none(raw.get("canonical_statement")),
            parent_object_id=string_or_none(raw.get("parent_object_id")),
            supersedes=normalize_string_list(raw.get("supersedes") or []),
            extraction_rationale=string_or_none(raw.get("extraction_rationale")),
            thread_position=dict(raw.get("thread_position") or {}),
            metadata=dict(raw.get("metadata") or {}),
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
            "updated_at": self.updated_at or self.created_at,
            "origin": self.origin,
            "evidence": self.evidence,
            "generated_by": self.generated_by,
            "object_role": self.object_role,
            "canonical_statement": self.canonical_statement,
            "parent_object_id": self.parent_object_id,
            "supersedes": self.supersedes,
            "extraction_rationale": self.extraction_rationale,
            "thread_position": self.thread_position,
            "metadata": self.metadata,
        }


TYPE_DIRS = {
    "idea": "Ideas",
    "decision": "Decisions",
    "open_loop": "Open Loops",
    "question": "Questions",
    "task": "Tasks",
    "feature": "Features",
    "insight": "Insights",
    "requirement": "Requirements",
    "risk": "Risks",
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


def normalize_origin(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in {"user", "assistant", "mixed", "unclear"} else "unclear"


def normalize_evidence(raw_evidence: list[Any]) -> list[str]:
    evidence: list[str] = []
    for item in raw_evidence:
        text = str(item).strip()
        if is_bare_message_id(text):
            continue
        if text and text not in evidence:
            evidence.append(text[:1200])
    return evidence[:8]


def is_bare_message_id(text: str) -> bool:
    return text.startswith("msg_") and all(ch.isalnum() or ch == "_" for ch in text) and len(text) <= 80


def normalize_object_role(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    allowed = {"durable_note", "artifact_spec", "revision", "decision", "raw_capture"}
    return normalized if normalized in allowed else "durable_note"


def normalize_string_list(raw_items: list[Any]) -> list[str]:
    items: list[str] = []
    for item in raw_items:
        text = str(item).strip()
        if text and text not in items:
            items.append(text)
    return items


def clean_title(title: str) -> str:
    title = " ".join(title.split())
    return title[:120] if title else "Untitled"


def string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
