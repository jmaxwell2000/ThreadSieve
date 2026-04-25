from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from .extractor import extract_items
from .ids import sha256_text, short_hash
from .importers import import_file
from .models import KnowledgeItem, SourceRef, Thread, utc_now_iso
from .prompts import DEFAULT_EXTRACT_PROMPT
from .semantic import EXTRACTION_FROM_SEMANTIC_LOG_PROMPT, build_semantic_log, write_semantic_log
from .writer import write_pipeline_item


SOURCE_SUFFIXES = {".md", ".markdown", ".txt", ".json"}


def run_id() -> str:
    return "run_" + utc_now_iso().replace("+00:00", "Z").replace(":", "").replace("-", "")


def source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.exists():
        raise RuntimeError(f"Source path does not exist: {source}")
    files = [path for path in sorted(source.rglob("*")) if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES]
    if not files:
        raise RuntimeError(f"No supported source files found in {source}")
    return files


def source_hash(path: Path) -> str:
    return sha256_text(path.read_text(encoding="utf-8"))


def load_state(output_root: Path) -> dict[str, Any]:
    path = state_path(output_root)
    if not path.exists():
        return {"processed": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(output_root: Path, state: dict[str, Any]) -> None:
    path = state_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def state_path(output_root: Path) -> Path:
    return output_root / ".threadsieve" / "state.json"


def extract_sources(
    source: Path,
    output_root: Path,
    model_config: dict[str, Any],
    threshold: float,
    force: bool = False,
    dry_run: bool = False,
    system_prompt: str | None = None,
    write_index: bool = True,
    overwrite_existing: bool = False,
    semantic_logs: bool = True,
    semantic_prompt: str | None = None,
) -> dict[str, Any]:
    started_at = utc_now_iso()
    current_run_id = run_id()
    state = load_state(output_root)
    processed = state.setdefault("processed", {})
    summary: dict[str, Any] = {
        "run_id": current_run_id,
        "started_at": started_at,
        "finished_at": None,
        "source": str(source),
        "output": str(output_root),
        "source_files_seen": 0,
        "source_files_processed": 0,
        "source_files_skipped": 0,
        "messages_parsed": 0,
        "objects_created": 0,
        "objects_by_type": {},
        "needs_review_count": 0,
        "warnings": [],
        "errors": [],
        "created_files": [],
        "semantic_logs_created": [],
        "provider": model_config.get("provider", "offline"),
        "model": model_config.get("model"),
    }
    type_counts: Counter[str] = Counter()
    if str(model_config.get("provider") or "offline").lower() in {"offline", "none", "heuristic"}:
        summary["warnings"].append(
            "Provider is offline; this is only a smoke test. Configure OpenRouter or another model for real semantic extraction."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    for path in source_files(source):
        summary["source_files_seen"] += 1
        resolved = path.resolve()
        digest = source_hash(resolved)
        state_key = str(resolved)
        previous = processed.get(state_key)
        if previous and previous.get("source_hash") == digest and not force:
            summary["source_files_skipped"] += 1
            continue
        if previous and previous.get("source_hash") != digest and not force:
            summary["source_files_skipped"] += 1
            summary["warnings"].append(f"{resolved} changed after a previous run; use --force to reprocess it.")
            continue

        thread = import_file(resolved)
        summary["source_files_processed"] += 1
        summary["messages_parsed"] += len(thread.messages)
        extraction_thread = thread
        extraction_prompt = system_prompt or DEFAULT_EXTRACT_PROMPT
        semantic_path: Path | None = None
        if semantic_logs:
            semantic_log = build_semantic_log(thread, model_config, semantic_prompt=semantic_prompt)
            extraction_thread = semantic_log.extraction_thread
            extraction_prompt = f"{extraction_prompt.rstrip()}\n\n{EXTRACTION_FROM_SEMANTIC_LOG_PROMPT}"
            semantic_path = output_root / "semantic_logs" / f"{thread.id}.md"
            if not dry_run:
                semantic_path = write_semantic_log(output_root / "semantic_logs", thread, semantic_log.text)
            summary["semantic_logs_created"].append(str(semantic_path))
        raw_items = extract_items(extraction_thread, model_config, threshold=0.0, system_prompt=extraction_prompt)
        written_for_source: list[str] = []
        for item in raw_items:
            enriched = enrich_item(item, thread, current_run_id, resolved, threshold, semantic_path)
            needs_review = enriched.confidence < threshold or weak_provenance(enriched)
            try:
                if dry_run:
                    created_path = output_root / ("_needs_review" if needs_review else f"{enriched.type}s") / f"{enriched.id}.md"
                else:
                    created_path = write_pipeline_item(output_root, enriched, thread, needs_review, overwrite=overwrite_existing)
            except FileExistsError as exc:
                summary["warnings"].append(str(exc))
                continue
            summary["objects_created"] += 1
            type_counts[enriched.type] += 1
            if needs_review:
                summary["needs_review_count"] += 1
            summary["created_files"].append(str(created_path))
            written_for_source.append(str(created_path))

        if not dry_run:
            processed[state_key] = {
                "source_path": state_key,
                "source_hash": digest,
                "last_processed_at": utc_now_iso(),
                "run_id": current_run_id,
                "objects_created": written_for_source,
            }

    summary["objects_by_type"] = dict(sorted(type_counts.items()))
    summary["finished_at"] = utc_now_iso()
    if not dry_run:
        save_state(output_root, state)
        write_run_record(output_root, summary)
        if write_index:
            rebuild_index(output_root)
    return summary


def enrich_item(
    item: KnowledgeItem,
    thread: Thread,
    current_run_id: str,
    source_path: Path,
    threshold: float,
    semantic_log_path: Path | None = None,
) -> KnowledgeItem:
    refs = [enrich_ref(ref, thread, source_path) for ref in item.source_refs]
    evidence = item.evidence or excerpts_for_refs(thread, refs)
    generated_by = {"app": "threadsieve", "extractor": item.type, "run_id": current_run_id}
    if semantic_log_path:
        generated_by["semantic_log"] = str(semantic_log_path)
    return replace(
        item,
        status=item.status or "raw",
        origin=item.origin or "unclear",
        source_refs=refs,
        evidence=evidence,
        generated_by=generated_by,
    )


def enrich_ref(ref: SourceRef, thread: Thread, source_path: Path) -> SourceRef:
    message = next((candidate for candidate in thread.messages if candidate.id == ref.message_id), None)
    source_id = thread.id
    chat_id = thread.metadata.get("chat_id") if isinstance(thread.metadata, dict) else None
    if not message:
        return replace(
            ref,
            source_id=source_id,
            source_path=str(source_path),
            chat_id=chat_id,
            granularity=ref.granularity or "message",
        )
    start = max(0, min(ref.start_char, len(message.content)))
    end = max(start, min(ref.end_char, len(message.content)))
    return SourceRef(
        source_id=source_id,
        source_path=str(source_path),
        chat_id=chat_id,
        message_id=message.id,
        role=message.role,
        timestamp=message.timestamp,
        granularity="message" if start == 0 and end >= len(message.content) else "span",
        start_char=start,
        end_char=end,
    )


def excerpts_for_refs(thread: Thread, refs: list[SourceRef]) -> list[str]:
    excerpts: list[str] = []
    messages = {message.id: message for message in thread.messages}
    for ref in refs:
        message = messages.get(ref.message_id)
        if not message:
            continue
        excerpt = message.content[ref.start_char : ref.end_char].strip() or message.content[:1200].strip()
        if excerpt and excerpt not in excerpts:
            excerpts.append(excerpt[:1200])
    return excerpts[:8]


def weak_provenance(item: KnowledgeItem) -> bool:
    return not item.source_refs or any(ref.granularity == "file" or not ref.message_id for ref in item.source_refs)


def write_run_record(output_root: Path, summary: dict[str, Any]) -> Path:
    runs = output_root / "_runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{summary['run_id']}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def rebuild_index(output_root: Path) -> Path:
    records = []
    for path in sorted(output_root.rglob("*.md")):
        if ".threadsieve" in path.parts or "_runs" in path.parts:
            continue
        data = parse_frontmatter(path)
        if not data.get("id"):
            continue
        data["path"] = str(path)
        records.append(data)
    index_path = output_root / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return index_path


def parse_frontmatter(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    try:
        end = lines[1:].index("---") + 1
    except ValueError:
        return {}
    data: dict[str, Any] = {}
    source_refs: list[dict[str, Any]] = []
    in_refs = False
    current_ref: dict[str, Any] | None = None
    for line in lines[1:end]:
        if line.startswith("source_refs:"):
            in_refs = True
            continue
        if in_refs and line and not line.startswith(" "):
            in_refs = False
            if current_ref:
                source_refs.append(current_ref)
                current_ref = None
        if in_refs:
            stripped = line.strip()
            if stripped == "-":
                if current_ref:
                    source_refs.append(current_ref)
                current_ref = {}
                continue
            if current_ref is not None and ":" in stripped:
                key, value = stripped.split(":", 1)
                current_ref[key.strip()] = parse_scalar(value.strip())
            continue
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            value = value.strip()
            if value and value != "[]":
                data[key.strip()] = parse_scalar(value)
    if current_ref:
        source_refs.append(current_ref)
    if source_refs:
        data["source_refs"] = source_refs
    return data


def parse_scalar(value: str) -> Any:
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('\\"', '"')
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def find_object_record(output_root: Path, object_id_or_path: str) -> dict[str, Any] | None:
    candidate = Path(object_id_or_path)
    if candidate.exists():
        record = parse_frontmatter(candidate)
        record["path"] = str(candidate)
        return record
    index_path = output_root / "index.jsonl"
    if not index_path.exists():
        rebuild_index(output_root)
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("id") == object_id_or_path:
                return record
    return None


def trace_object(output_root: Path, object_id_or_path: str) -> str:
    record = find_object_record(output_root, object_id_or_path)
    if not record:
        raise RuntimeError(f"No object found for {object_id_or_path}")
    lines = [
        f"{record.get('title') or record.get('id')}",
        f"Type: {record.get('type', 'unknown')}",
        f"Object: {record.get('id')}",
        "",
    ]
    refs = record.get("source_refs") or []
    if not refs:
        lines.append("No source_refs found.")
        return "\n".join(lines).rstrip() + "\n"

    for ref in refs:
        source_path = ref.get("source_path")
        lines.append(f"Source: {source_path or 'unknown'}")
        lines.append(f"Message: {ref.get('message_id')} ({ref.get('role') or 'unknown'} {ref.get('timestamp') or ''})".rstrip())
        if source_path and Path(str(source_path)).exists():
            try:
                thread = import_file(Path(str(source_path)))
                message = next((candidate for candidate in thread.messages if candidate.id == ref.get("message_id")), None)
                if message:
                    start = int(ref.get("start_char") or 0)
                    end = int(ref.get("end_char") or len(message.content))
                    excerpt = message.content[start:end].strip() or message.content.strip()
                    lines.extend(["", excerpt[:2400], ""])
                else:
                    lines.append("Warning: message ID was not found in the current source file.")
            except Exception as exc:
                lines.append(f"Warning: could not parse source file: {exc}")
        else:
            lines.append("Warning: source file is missing or was moved.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def short_source_id(path: Path) -> str:
    return f"source_{short_hash(str(path.resolve()), 12)}"
