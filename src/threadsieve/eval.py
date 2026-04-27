from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .extractor import looks_like_user_artifact_text
from .ids import slugify
from .importers import import_file
from .pipeline import extract_sources, parse_frontmatter
from .prompts import DEFAULT_EXTRACT_PROMPT, DEFAULT_SEMANTIC_PROMPT


DEFAULT_EVAL_MODELS = [
    "openai/gpt-5-mini",
    "google/gemini-3.1-flash-lite-preview",
    "qwen/qwen3-30b-a3b",
]


def default_fixture_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "regression"


def eval_fixture_files(fixtures: Path) -> list[Path]:
    if fixtures.is_file():
        return [fixtures]
    if not fixtures.exists():
        raise RuntimeError(f"Fixture path does not exist: {fixtures}")
    files = sorted(path for path in fixtures.glob("sample-*.md") if path.is_file())
    if not files:
        raise RuntimeError(f"No sample-*.md fixtures found in {fixtures}")
    return files


def estimate_model_calls(fixture_count: int, model_count: int, semantic_logs: bool = True) -> int:
    return fixture_count * model_count * (2 if semantic_logs else 1)


def run_live_eval(
    *,
    fixtures: Path,
    output_root: Path,
    provider: str,
    models: list[str],
    model_config_base: dict[str, Any],
    threshold: float,
    max_calls: int,
) -> dict[str, Any]:
    fixture_paths = eval_fixture_files(fixtures)
    estimated_calls = estimate_model_calls(len(fixture_paths), len(models), semantic_logs=True)
    if estimated_calls > max_calls:
        raise RuntimeError(f"Eval would make about {estimated_calls} model calls, above --max-calls {max_calls}.")

    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "fixtures": [str(path) for path in fixture_paths],
        "models": models,
        "provider": provider,
        "output": str(output_root),
        "estimated_model_calls": estimated_calls,
        "max_calls": max_calls,
        "results": [],
    }

    for model in models:
        model_output = output_root / slugify(model)
        model_config = dict(model_config_base)
        model_config["provider"] = provider
        model_config["model"] = model
        model_result: dict[str, Any] = {
            "model": model,
            "output": str(model_output),
            "fixtures": [],
            "passed": True,
            "errors": [],
        }
        for fixture_path in fixture_paths:
            fixture_result = run_fixture_eval(fixture_path, model_output, model_config, threshold)
            model_result["fixtures"].append(fixture_result)
            if not fixture_result["passed"]:
                model_result["passed"] = False
        report["results"].append(model_result)

    report["passed"] = all(result["passed"] for result in report["results"])
    (output_root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_fixture_eval(fixture_path: Path, output_root: Path, model_config: dict[str, Any], threshold: float) -> dict[str, Any]:
    thread = import_file(fixture_path)
    result: dict[str, Any] = {
        "fixture": str(fixture_path),
        "thread_id": thread.id,
        "passed": True,
        "checks": [],
        "summary": {},
    }
    try:
        summary = extract_sources(
            source=fixture_path,
            output_root=output_root,
            model_config=model_config,
            threshold=threshold,
            force=True,
            dry_run=False,
            system_prompt=DEFAULT_EXTRACT_PROMPT,
            write_index=True,
            overwrite_existing=True,
            semantic_logs=True,
            semantic_prompt=DEFAULT_SEMANTIC_PROMPT,
        )
        result["summary"] = summary
    except Exception as exc:
        add_check(result, "pipeline completed", False, str(exc))
        return result

    run_quality_checks(result, thread, output_root, fixture_path)
    return result


def run_quality_checks(result: dict[str, Any], thread: Any, output_root: Path, fixture_path: Path) -> None:
    semantic_text = semantic_log_text(output_root, thread.id)
    add_check(result, "semantic log exists", bool(semantic_text), f"Missing semantic log for {thread.id}")
    if semantic_text:
        for message in thread.messages:
            add_check(
                result,
                f"semantic log contains {message.id}",
                f"## {message.id} " in semantic_text,
                f"Missing semantic block for {message.id}",
            )
        for message in thread.messages:
            if message.role == "user":
                add_check(
                    result,
                    f"user message preserved {message.id}",
                    message.content in semantic_text,
                    f"User message changed or missing: {message.id}",
                )
        if "example-continuation" in fixture_path.name or "long-mixed" in fixture_path.name:
            add_check(
                result,
                "continuation examples stay context",
                "ARTIFACT_TYPE: examples" not in semantic_text,
                "Example continuation was labeled as AI_ARTIFACT.",
            )

    created_files = [Path(path) for path in (result.get("summary") or {}).get("created_files") or []]
    records = object_records(output_root, created_files)
    created = int(dict(result.get("summary") or {}).get("objects_created") or 0)
    if "example-continuation" not in fixture_path.name:
        add_check(result, "objects created", created > 0, "No objects were created.")
    for record in records:
        add_check(result, f"{record.get('id')} has source refs", bool(record.get("source_refs")), "Object missing source_refs.")
        if record.get("type") == "framework" or record.get("object_role") == "artifact_spec":
            add_check(
                result,
                f"{record.get('id')} artifact role is supported",
                record_has_artifact_support(record, thread, semantic_text),
                "framework/artifact_spec object was not grounded in a named user artifact or AI_ARTIFACT under revision.",
            )
    if "example-continuation" in fixture_path.name:
        for record in records:
            add_check(
                result,
                f"{record.get('id')} does not promote examples to artifact",
                record.get("type") != "framework" and record.get("object_role") != "artifact_spec",
                "Example-continuation output should not become a framework or artifact_spec.",
            )
            add_check(
                result,
                f"{record.get('id')} does not save assistant examples",
                not record_has_assistant_refs(record),
                "Example-continuation output was grounded in assistant examples.",
            )
    add_check(result, "no bare message-id evidence", not has_bare_message_id_evidence(output_root), "Bare msg_ evidence found.")
    add_check(result, "no rendered list body", not has_rendered_list_body(output_root), "Python/JSON-style list body found.")

    if "directive-framework" in fixture_path.name or "long-mixed" in fixture_path.name:
        combined = "\n".join(path.read_text(encoding="utf-8") for path in created_files if path.exists())
        expected = (
            ["Zero Preamble", "Brevity Limit", "Signal Words", "Failure Mode"]
            if "directive-framework" in fixture_path.name
            else ["Source First", "Small Objects", "Review Flag", "No Private Data"]
        )
        for directive in expected:
            add_check(result, f"framework preserves {directive}", directive in combined, f"Missing directive: {directive}")


def record_has_artifact_support(record: dict[str, Any], thread: Any, semantic_text: str) -> bool:
    messages = {message.id: message for message in thread.messages}
    for ref in record.get("source_refs") or []:
        message_id = str(ref.get("message_id") or "")
        if f"## {message_id} AI_ARTIFACT" in semantic_text:
            return True
        message = messages.get(message_id)
        if message and message.role == "user" and looks_like_user_artifact_text(message.content):
            return True
    return False


def record_has_assistant_refs(record: dict[str, Any]) -> bool:
    refs = record.get("source_refs") or []
    if not refs:
        return False
    return any(str(ref.get("role") or "").lower() == "assistant" for ref in refs)


def add_check(result: dict[str, Any], name: str, passed: bool, message: str) -> None:
    result["checks"].append({"name": name, "passed": passed, "message": "" if passed else message})
    if not passed:
        result["passed"] = False


def semantic_log_text(output_root: Path, thread_id: str) -> str:
    semantic_dir = output_root / "semantic_logs"
    if not semantic_dir.exists():
        return ""
    matches = sorted(semantic_dir.glob(f"{thread_id}*.md"))
    return matches[0].read_text(encoding="utf-8") if matches else ""


def object_records(output_root: Path, paths: list[Path] | None = None) -> list[dict[str, Any]]:
    records = []
    candidates = paths if paths is not None else sorted(output_root.rglob("*.md"))
    for path in candidates:
        if "semantic_logs" in path.parts or "_runs" in path.parts:
            continue
        data = parse_frontmatter(path)
        if data.get("id"):
            records.append(data)
    return records


def has_bare_message_id_evidence(output_root: Path) -> bool:
    for path in output_root.rglob("*.md"):
        if path.name.startswith("run_") or "semantic_logs" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip().strip('"')
            if stripped.startswith("- msg_") or stripped.startswith("> msg_"):
                return True
    return False


def has_rendered_list_body(output_root: Path) -> bool:
    for path in output_root.rglob("*.md"):
        if "semantic_logs" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "\n## Details\n\n[" in text:
            return True
    return False


def print_eval_report(report: dict[str, Any]) -> None:
    print("ThreadSieve live eval")
    print(f"Provider: {report['provider']}")
    print(f"Models: {', '.join(report['models'])}")
    print(f"Fixtures: {len(report['fixtures'])}")
    print(f"Estimated model calls: {report['estimated_model_calls']} / {report['max_calls']}")
    print(f"Output: {report['output']}")
    print("")
    for model_result in report["results"]:
        status = "PASS" if model_result["passed"] else "FAIL"
        print(f"{status} {model_result['model']}")
        for fixture_result in model_result["fixtures"]:
            fixture_status = "PASS" if fixture_result["passed"] else "FAIL"
            print(f"  {fixture_status} {Path(fixture_result['fixture']).name}")
            for check in fixture_result["checks"]:
                if not check["passed"]:
                    print(f"    - {check['name']}: {check['message']}")
        print("")
