from __future__ import annotations

from pathlib import Path
from typing import Any

from .pipeline import find_object_record, parse_frontmatter, rebuild_index, trace_object


REVIEW_STATUSES = {"raw", "reviewed", "accepted", "rejected", "superseded"}


def list_review_objects(knowledge: Path, limit: int = 50, item_type: str | None = None) -> list[dict[str, Any]]:
    records = []
    for path in sorted(knowledge.rglob("*.md")):
        if ".threadsieve" in path.parts or "_runs" in path.parts or "semantic_logs" in path.parts:
            continue
        data = parse_frontmatter(path)
        if not data.get("id"):
            continue
        data["path"] = str(path)
        data["needs_review"] = "_needs_review" in path.parts
        status = str(data.get("status") or "raw")
        if status != "raw" and not data["needs_review"]:
            continue
        if item_type and data.get("type") != item_type:
            continue
        records.append(data)
    records.sort(key=review_sort_key)
    return records[: max(0, limit)]


def review_sort_key(record: dict[str, Any]) -> tuple[int, float, str]:
    needs_review_rank = 0 if record.get("needs_review") else 1
    try:
        confidence = float(record.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0
    return needs_review_rank, confidence, str(record.get("title") or record.get("id") or "")


def review_object_record(knowledge: Path, object_id_or_path: str) -> dict[str, Any]:
    record = find_object_record(knowledge, object_id_or_path)
    if not record:
        raise RuntimeError(f"No object found for {object_id_or_path}")
    return record


def update_review_status(knowledge: Path, object_id_or_path: str, status: str) -> dict[str, Any]:
    if status not in REVIEW_STATUSES:
        choices = ", ".join(sorted(REVIEW_STATUSES))
        raise RuntimeError(f"Invalid status {status!r}. Choose one of: {choices}.")
    record = review_object_record(knowledge, object_id_or_path)
    path = Path(str(record["path"]))
    replace_frontmatter_status(path, status)
    rebuild_index(knowledge)
    updated = parse_frontmatter(path)
    updated["path"] = str(path)
    return updated


def replace_frontmatter_status(path: Path, status: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise RuntimeError(f"Object file has no front matter: {path}")
    try:
        end = lines[1:].index("---") + 1
    except ValueError as exc:
        raise RuntimeError(f"Object file has unterminated front matter: {path}") from exc

    replacement = f"status: {status}"
    for index in range(1, end):
        line = lines[index]
        if line.startswith("status:"):
            lines[index] = replacement
            path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
            return

    lines.insert(end, replacement)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def format_review_list(records: list[dict[str, Any]]) -> str:
    if not records:
        return "No raw or needs-review objects found.\n"
    lines = ["ThreadSieve review", ""]
    for record in records:
        confidence = record.get("confidence")
        confidence_text = f"{float(confidence):.2f}" if isinstance(confidence, (int, float)) else str(confidence or "n/a")
        location = "needs_review" if record.get("needs_review") else "main"
        lines.append(
            f"{record.get('id')}\t{record.get('type', 'unknown')}\t{record.get('status', 'raw')}\t{confidence_text}\t{location}\t{record.get('title', '')}"
        )
        lines.append(f"  {record.get('path')}")
    return "\n".join(lines).rstrip() + "\n"


def format_review_detail(knowledge: Path, record: dict[str, Any]) -> str:
    lines = [
        f"{record.get('title') or record.get('id')}",
        f"ID: {record.get('id')}",
        f"Type: {record.get('type', 'unknown')}",
        f"Status: {record.get('status', 'raw')}",
        f"Confidence: {record.get('confidence', 'unknown')}",
        f"Path: {record.get('path')}",
        "",
        "Summary:",
        str(record.get("summary") or "").strip() or "No summary.",
    ]
    canonical = str(record.get("canonical_statement") or "").strip()
    if canonical:
        lines.extend(["", "Canonical statement:", canonical])
    evidence = record.get("evidence")
    if evidence:
        lines.extend(["", "Evidence:"])
        if isinstance(evidence, list):
            lines.extend(f"- {item}" for item in evidence)
        else:
            lines.append(str(evidence))
    lines.extend(["", "Trace:", trace_object(knowledge, str(record.get("id"))).rstrip(), ""])
    return "\n".join(lines).rstrip() + "\n"
