from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .archive import archive_thread, load_thread_from_archive
from .config import Config, default_config_path, expand_path, load_config, write_default_config
from .eval import DEFAULT_EVAL_MODELS, default_fixture_dir, print_eval_report, run_live_eval
from .extractor import extract_items
from .importers import import_file, import_text
from .index import get_object, index_object, index_thread, latest_thread_path, search
from .prompts import DEFAULT_PROMPTS, ensure_default_prompt, load_extract_prompt, load_semantic_prompt
from .providers import PROVIDER_PRESETS, build_provider, fetch_json, provider_request, provider_status
from .semantic import EXTRACTION_FROM_SEMANTIC_LOG_PROMPT, build_semantic_log, write_semantic_log
from .writer import append_jsonl, write_item
from .pipeline import extract_sources, rebuild_index, trace_object


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        if os.environ.get("THREADSIEVE_DEBUG"):
            traceback.print_exc()
        print(f"threadsieve: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="threadsieve", description="Extract durable, source-linked notes from conversation threads.")
    parser.add_argument("--config", help="Path to config JSON. Defaults to ~/.threadsieve/config.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create config and workspace folders.")
    init.add_argument("--workspace", help="Workspace path. Defaults to ./ThreadSieve in the config.")
    init.set_defaults(func=cmd_init)

    providers = subparsers.add_parser("providers", help="List supported LLM provider presets.")
    providers.set_defaults(func=cmd_providers)

    config_cmd = subparsers.add_parser("configure-provider", help="Write an extraction provider preset into the config.")
    config_cmd.add_argument("provider", choices=sorted(PROVIDER_PRESETS.keys()))
    config_cmd.add_argument("--model", help="Override the preset model.")
    config_cmd.add_argument("--base-url", help="Override the preset base URL.")
    config_cmd.add_argument("--api-key-env", help="Override the preset API key environment variable.")
    config_cmd.set_defaults(func=cmd_configure_provider)

    doctor = subparsers.add_parser("doctor", help="Inspect local config and provider setup without sending thread contents.")
    doctor.set_defaults(func=cmd_doctor)

    test_provider = subparsers.add_parser("test-provider", help="Send a tiny test prompt to the configured provider.")
    test_provider.add_argument("--prompt", default="Reply with JSON: {\"ok\": true}", help="Tiny prompt to send to the configured provider.")
    test_provider.set_defaults(func=cmd_test_provider)

    regression = subparsers.add_parser("regression", help="Run privacy-safe regression fixture tests.")
    regression.set_defaults(func=cmd_regression)

    eval_cmd = subparsers.add_parser("eval", help="Run live-model evals on privacy-safe fixtures.")
    eval_cmd.add_argument("--provider", default="openrouter", help="Provider to use. Defaults to openrouter.")
    eval_cmd.add_argument("--model", action="append", default=[], help="Model to test. May be repeated.")
    eval_cmd.add_argument("--fixtures", help="Fixture file/folder. Defaults to tests/fixtures/regression.")
    eval_cmd.add_argument("--out", help="Output folder. Defaults to ./eval-runs/<timestamp>.")
    eval_cmd.add_argument("--max-calls", type=int, default=75, help="Hard cap on estimated model calls. Defaults to 75.")
    eval_cmd.add_argument("--json", action="store_true", help="Print machine-readable report JSON.")
    eval_cmd.set_defaults(func=cmd_eval)

    show_prompt = subparsers.add_parser("show-prompt", help="Print the active prompt path and contents.")
    show_prompt.add_argument("--kind", choices=sorted(DEFAULT_PROMPTS), default="extract", help="Prompt to show. Defaults to extract.")
    show_prompt.add_argument("--path-only", action="store_true", help="Only print the prompt file path.")
    show_prompt.set_defaults(func=cmd_show_prompt)

    reset_prompt = subparsers.add_parser("reset-prompt", help="Create or restore editable prompts.")
    reset_prompt.add_argument("--kind", choices=sorted([*DEFAULT_PROMPTS, "all"]), default="extract", help="Prompt to reset. Defaults to extract.")
    reset_prompt.add_argument("--force", action="store_true", help="Overwrite the existing prompt with the default prompt.")
    reset_prompt.set_defaults(func=cmd_reset_prompt)

    ingest = subparsers.add_parser("ingest", help="Ingest and archive a thread file.")
    ingest.add_argument("path", help="Path to ChatGPT JSON, normalized JSON, Markdown, or text.")
    ingest.add_argument("--source-app", help="Override source app name.")
    ingest.set_defaults(func=cmd_ingest)

    extract = subparsers.add_parser("extract", help="Extract knowledge objects from a thread.")
    extract.add_argument("path", nargs="?", help="Optional file path to ingest and extract.")
    extract.add_argument("--config", dest="command_config", help="Path to config JSON for this command.")
    extract.add_argument("--source", help="File or folder to process into --out.")
    extract.add_argument("--out", help="Output folder for handoff-style extraction.")
    extract.add_argument("--thread", choices=["latest"], help="Extract from the latest ingested thread.")
    extract.add_argument("--file", dest="file_path", help="Explicit file path to extract.")
    extract.add_argument("--clipboard", action="store_true", help="Extract from system clipboard text.")
    extract.add_argument("--source-app", help="Override source app name for file/clipboard input.")
    extract.add_argument("--extractor", action="append", default=[], help="Extractor name to request. May be repeated.")
    extract.add_argument("--provider", help="Provider override. Example: openrouter.")
    extract.add_argument("--model", help="Model override.")
    extract.add_argument("--force", action="store_true", help="Reprocess source files even if state says they were already processed.")
    extract.add_argument("--dry-run", action="store_true", help="Run parsing/extraction without writing files, state, or index.")
    extract.add_argument("--json", action="store_true", help="Print a machine-readable JSON summary.")
    extract.add_argument("--no-semantic-log", action="store_true", help="Skip semantic-log generation and extract from the raw transcript.")
    extract.set_defaults(func=cmd_extract)

    index_cmd = subparsers.add_parser("index", help="Build or rebuild index.jsonl for a knowledge output folder.")
    index_cmd.add_argument("knowledge", nargs="?", help="Knowledge output folder.")
    index_cmd.add_argument("--knowledge", dest="knowledge_flag", help="Knowledge output folder.")
    index_cmd.set_defaults(func=cmd_index_jsonl)

    trace = subparsers.add_parser("trace", help="Print source context for a knowledge object.")
    trace.add_argument("object", help="Object ID or Markdown object file path.")
    trace.add_argument("--knowledge", default="./knowledge", help="Knowledge output folder containing index.jsonl.")
    trace.set_defaults(func=cmd_trace)

    watch = subparsers.add_parser("watch", help="Watch folders for extraction. Not implemented yet.")
    watch.set_defaults(func=cmd_watch)

    dedupe = subparsers.add_parser("dedupe", help="Detect semantic duplicates. Not implemented yet.")
    dedupe.set_defaults(func=cmd_dedupe)

    search_cmd = subparsers.add_parser("search", help="Search extracted knowledge objects.")
    search_cmd.add_argument("query", help="Search query.")
    search_cmd.add_argument("--limit", type=int, default=10)
    search_cmd.set_defaults(func=cmd_search)

    open_cmd = subparsers.add_parser("open", help="Print a knowledge object path, or open it/source with the OS file opener.")
    open_cmd.add_argument("object_id")
    open_cmd.add_argument("--source", action="store_true", help="Open the archived source thread instead of the note.")
    open_cmd.add_argument("--print", action="store_true", help="Print the path instead of opening it.")
    open_cmd.set_defaults(func=cmd_open)
    return parser


def cmd_init(args: argparse.Namespace) -> int:
    path = expand_path(args.config) if args.config else default_config_path()
    write_default_config(path)
    config = load_config(str(path))
    if args.workspace:
        config.raw["workspace"] = args.workspace
        path.write_text(json.dumps(config.raw, indent=2) + "\n", encoding="utf-8")
        config = load_config(str(path))
    prompt_path = ensure_default_prompt(config, "extract")
    semantic_prompt_path = ensure_default_prompt(config, "semantic")
    create_workspace(config.workspace)
    print(f"Config: {path}")
    print(f"Prompt: {prompt_path}")
    print(f"Semantic prompt: {semantic_prompt_path}")
    print(f"Workspace: {config.workspace}")
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    for name, preset in sorted(PROVIDER_PRESETS.items()):
        network = "network" if preset.get("network") else "local"
        print(f"{name}\t{network}\t{preset.get('description', '')}")
    return 0


def cmd_configure_provider(args: argparse.Namespace) -> int:
    path = expand_path(args.config) if args.config else default_config_path()
    write_default_config(path)
    config = load_config(str(path))
    preset = dict(PROVIDER_PRESETS[args.provider])
    preset["provider"] = args.provider
    preset.pop("description", None)
    preset.pop("network", None)
    if args.model:
        preset["model"] = args.model
    if args.base_url:
        preset["base_url"] = args.base_url
    if args.api_key_env:
        preset["api_key_env"] = args.api_key_env
    config.raw.setdefault("models", {})["extract"] = preset
    path.write_text(json.dumps(config.raw, indent=2) + "\n", encoding="utf-8")
    print(f"Configured extract provider: {args.provider}")
    print(f"Config: {path}")
    if preset.get("api_key_env"):
        print(f"Set your API key with: export {preset['api_key_env']}=\"...\"")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    provider = build_provider(config.extract_model)
    print(f"Config: {config.path}")
    print(f"Workspace: {config.workspace}")
    for key, value in provider_status(provider).items():
        print(f"{key}: {value}")
    if provider.network:
        print("network_notice: extraction will send thread text to this provider when you run `extract`.")
    return 0


def cmd_test_provider(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    provider = build_provider(config.extract_model)
    if provider.kind == "offline":
        print("offline provider is active; no network test is needed.")
        return 0
    print(f"Sending a tiny provider test to {provider.name} at {provider.base_url}.")
    request = provider_request(provider, messages=[{"role": "user", "content": args.prompt}], response_format={"type": "json_object"})
    response = fetch_json(request, timeout=provider.timeout_seconds)
    content = response.get("choices", [{}])[0].get("message", {}).get("content")
    print(content or json.dumps(response)[:1000])
    return 0


def cmd_regression(args: argparse.Namespace) -> int:
    tests_dir = find_tests_dir()
    command = [sys.executable, "-m", "unittest", "discover", "-s", str(tests_dir), "-p", "test_regression_fixtures.py"]
    print("Running privacy-safe regression fixtures...", flush=True)
    print(" ".join(command), flush=True)
    return subprocess.run(command).returncode


def cmd_eval(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    models = args.model or list(DEFAULT_EVAL_MODELS)
    fixtures = expand_path(args.fixtures) if args.fixtures else default_fixture_dir()
    output = expand_path(args.out) if args.out else Path.cwd() / "eval-runs" / utc_filename("eval")
    model_config = dict(config.extract_model)
    provider_config = config.raw.get("provider")
    if isinstance(provider_config, dict):
        model_config.update(provider_config)
        if "name" in provider_config and "provider" not in provider_config:
            model_config["provider"] = provider_config["name"]
    report = run_live_eval(
        fixtures=fixtures,
        output_root=output,
        provider=args.provider,
        models=models,
        model_config_base=model_config,
        threshold=float(config.raw.get("behavior", {}).get("needs_review_confidence_threshold", config.confidence_threshold)),
        max_calls=args.max_calls,
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print_eval_report(report)
    return 0 if report.get("passed") else 1


def utc_filename(prefix: str) -> str:
    return prefix + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def find_tests_dir() -> Path:
    candidates = [
        Path.cwd() / "tests",
        Path(__file__).resolve().parents[2] / "tests",
    ]
    for candidate in candidates:
        if (candidate / "test_regression_fixtures.py").exists():
            return candidate
    raise RuntimeError("Could not find tests/test_regression_fixtures.py. Run this from a ThreadSieve source checkout.")


def cmd_show_prompt(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    path = ensure_default_prompt(config, args.kind)
    print(path)
    if not args.path_only:
        print("")
        print(path.read_text(encoding="utf-8"))
    return 0


def cmd_reset_prompt(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    kinds = sorted(DEFAULT_PROMPTS) if args.kind == "all" else [args.kind]
    paths = [ensure_default_prompt(config, kind, force=args.force) for kind in kinds]
    if args.force:
        for path in paths:
            print(f"Reset prompt: {path}")
    else:
        for path in paths:
            print(f"Prompt exists: {path}")
        print("Use `threadsieve reset-prompt --force` to overwrite an existing prompt.")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    create_workspace(config.workspace)
    thread = import_file(expand_path(args.path), source_app=args.source_app)
    source_dir = archive_thread(config.workspace, thread)
    index_thread(config.workspace, thread, source_dir)
    print(f"Ingested: {thread.title}")
    print(f"Thread ID: {thread.id}")
    print(f"Archived: {source_dir}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    config = load_config(getattr(args, "command_config", None) or args.config)
    if args.source or args.out or config.raw.get("sources"):
        return cmd_extract_sources(args, config)
    create_workspace(config.workspace)
    thread, source_dir = resolve_thread_for_extract(args, config)
    if not source_dir.exists() or not (source_dir / "thread.json").exists():
        source_dir = archive_thread(config.workspace, thread)
        index_thread(config.workspace, thread, source_dir)

    extraction_thread = thread
    prompt = load_extract_prompt(config)
    behavior = dict(config.raw.get("behavior") or {})
    if str(config.extract_model.get("provider") or "offline").lower() in {"offline", "none", "heuristic"}:
        print("Warning: provider is offline; this is only a smoke test. Configure OpenRouter or another model for real semantic extraction.")
    if behavior.get("semantic_logs", True) and not args.no_semantic_log:
        semantic_log = build_semantic_log(thread, config.extract_model, semantic_prompt=load_semantic_prompt(config))
        semantic_path = write_semantic_log(config.workspace / "Semantic Logs", thread, semantic_log.text)
        extraction_thread = semantic_log.extraction_thread
        prompt = f"{prompt.rstrip()}\n\n{EXTRACTION_FROM_SEMANTIC_LOG_PROMPT}"
        print(f"Semantic log: {semantic_path}")

    items = extract_items(extraction_thread, config.extract_model, config.confidence_threshold, prompt)
    if not items:
        print("No source-grounded items met the confidence threshold.")
        return 0

    for item in items:
        path = write_item(config.workspace, item, thread, source_dir)
        append_jsonl(config.workspace, item, thread, path)
        index_object(config.workspace, item, thread, path)
        print(f"{item.id}\t{item.type}\t{path}")
    print(f"Extracted {len(items)} item(s).")
    return 0


def cmd_extract_sources(args: argparse.Namespace, config: Config) -> int:
    configured_sources = list(config.raw.get("sources") or [])
    source_values = [args.source] if args.source else configured_sources
    output_value = args.out or config.raw.get("output")
    if not source_values:
        raise RuntimeError("threadsieve extract requires --source or sources in config.")
    if not output_value:
        raise RuntimeError("threadsieve extract requires --out or output in config.")
    model_config = dict(config.extract_model)
    provider_config = config.raw.get("provider")
    if isinstance(provider_config, dict):
        model_config.update(provider_config)
        if "name" in provider_config and "provider" not in provider_config:
            model_config["provider"] = provider_config["name"]
    if args.provider:
        model_config["provider"] = args.provider
        if args.provider == "openrouter" and "model" not in model_config:
            model_config["model"] = "openai/gpt-4.1-mini"
    if args.model:
        model_config["model"] = args.model
    behavior = dict(config.raw.get("behavior") or {})
    threshold = float(behavior.get("needs_review_confidence_threshold", config.confidence_threshold))
    prompt = load_extract_prompt(config)
    extractors = args.extractor or list(config.raw.get("extractors") or [])
    if extractors:
        prompt = prompt.rstrip() + "\nRequested extractor types: " + ", ".join(extractors) + ".\n"
    summaries = [
        extract_sources(
            source=expand_path(source_value),
            output_root=expand_path(str(output_value)),
            model_config=model_config,
            threshold=threshold,
            force=args.force,
            dry_run=args.dry_run or bool(behavior.get("dry_run", False)),
            system_prompt=prompt,
            write_index=bool(behavior.get("write_index", True)),
            overwrite_existing=bool(behavior.get("overwrite_existing", False)),
            semantic_logs=bool(behavior.get("semantic_logs", True)) and not args.no_semantic_log,
            semantic_prompt=load_semantic_prompt(config),
        )
        for source_value in source_values
    ]
    summary = combine_summaries(summaries)
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print_extract_summary(summary)
    return 0


def combine_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if len(summaries) == 1:
        return summaries[0]
    combined = dict(summaries[-1])
    combined["source"] = ", ".join(str(summary["source"]) for summary in summaries)
    for key in [
        "source_files_seen",
        "source_files_processed",
        "source_files_skipped",
        "messages_parsed",
        "objects_created",
        "needs_review_count",
    ]:
        combined[key] = sum(int(summary.get(key, 0)) for summary in summaries)
    type_counts: dict[str, int] = {}
    for summary in summaries:
        for item_type, count in dict(summary.get("objects_by_type") or {}).items():
            type_counts[item_type] = type_counts.get(item_type, 0) + int(count)
    combined["objects_by_type"] = dict(sorted(type_counts.items()))
    combined["warnings"] = [warning for summary in summaries for warning in list(summary.get("warnings") or [])]
    combined["errors"] = [error for summary in summaries for error in list(summary.get("errors") or [])]
    combined["created_files"] = [path for summary in summaries for path in list(summary.get("created_files") or [])]
    return combined


def print_extract_summary(summary: dict[str, object]) -> None:
    print("ThreadSieve extract")
    print("")
    print(f"Source: {summary['source']}")
    print(f"Output: {summary['output']}")
    print(f"Provider: {summary['provider']}")
    if summary.get("model"):
        print(f"Model: {summary['model']}")
    print("")
    print(f"Processed: {summary['source_files_processed']} file(s)")
    print(f"Skipped: {summary['source_files_skipped']} file(s)")
    print(f"Messages parsed: {summary['messages_parsed']}")
    print("")
    print(f"Objects created: {summary['objects_created']}")
    for item_type, count in dict(summary.get("objects_by_type") or {}).items():
        print(f"  {item_type}s: {count}")
    print("")
    print(f"Needs review: {summary['needs_review_count']}")
    created_files = list(summary.get("created_files") or [])
    if created_files:
        print("")
        print("Created files:")
        for path in created_files:
            print(f"  {path}")
    warnings = list(summary.get("warnings") or [])
    if warnings:
        print("")
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")


def cmd_index_jsonl(args: argparse.Namespace) -> int:
    knowledge = args.knowledge_flag or args.knowledge
    if not knowledge:
        raise RuntimeError("Choose a knowledge folder, for example `threadsieve index ./knowledge`.")
    path = rebuild_index(expand_path(knowledge))
    print(path)
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    print(trace_object(expand_path(args.knowledge), args.object), end="")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    print("Watch mode is not implemented yet. Use `threadsieve extract --source ./incoming --out ./knowledge` for now.")
    return 0


def cmd_dedupe(args: argparse.Namespace) -> int:
    print("Semantic deduplication is not implemented yet. Exact source-file reprocessing protection is handled during extraction.")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    rows = search(config.workspace, args.query, args.limit)
    if not rows:
        print("No matches.")
        return 0
    for row in rows:
        print(f"{row['id']}\t{row['type']}\t{row['title']}")
        print(f"  {row['summary']}")
        print(f"  {row['local_path']}")
    return 0


def cmd_open(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    row = get_object(config.workspace, args.object_id)
    if not row:
        raise RuntimeError(f"No object found with id {args.object_id}")
    path = Path(str(row["local_path"]))
    if args.source:
        source_path = find_source_path(path)
        if source_path:
            path = source_path
    if args.print or os.environ.get("CI"):
        print(path)
        return 0
    open_path(path)
    return 0


def resolve_thread_for_extract(args: argparse.Namespace, config: Config):
    if args.clipboard:
        text = read_clipboard()
        thread = import_text(text, title="Clipboard thread", source_app=args.source_app or "clipboard")
        return thread, config.workspace / "Sources" / "clipboard" / thread.id
    path_arg = args.file_path or args.path
    if path_arg:
        thread = import_file(expand_path(path_arg), source_app=args.source_app)
        return thread, config.workspace / "Sources" / thread.source_app / thread.id
    if args.thread == "latest":
        source_dir = latest_thread_path(config.workspace)
        if not source_dir:
            raise RuntimeError("No ingested threads found. Run `threadsieve ingest PATH` first.")
        return load_thread_from_archive(source_dir), source_dir
    raise RuntimeError("Choose a file, --file, --clipboard, or --thread latest.")


def create_workspace(workspace: Path) -> None:
    for relative in [
        "Knowledge/Ideas",
        "Knowledge/Decisions",
        "Knowledge/Open Loops",
        "Knowledge/Tasks",
        "Knowledge/Drafts",
        "Semantic Logs",
        "Sources",
    ]:
        (workspace / relative).mkdir(parents=True, exist_ok=True)


def read_clipboard() -> str:
    commands = clipboard_commands()
    saw_clipboard_tool = False
    for command in commands:
        if not command_available(command[0]):
            continue
        saw_clipboard_tool = True
        try:
            result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        if result.stdout.strip():
            return result.stdout
    if saw_clipboard_tool:
        raise RuntimeError("Clipboard is empty.")
    names = ", ".join(command[0] for command in commands)
    raise RuntimeError(f"Could not read clipboard. Install or enable one of: {names}.")


def clipboard_commands() -> list[list[str]]:
    if sys.platform == "darwin":
        return [["pbpaste"]]
    if sys.platform.startswith("win"):
        return [["powershell.exe", "-NoProfile", "-Command", "Get-Clipboard"], ["powershell", "-NoProfile", "-Command", "Get-Clipboard"]]
    return [
        ["wl-paste", "--no-newline"],
        ["xclip", "-selection", "clipboard", "-o"],
        ["xsel", "--clipboard", "--output"],
    ]


def command_available(name: str) -> bool:
    if sys.platform.startswith("win") and name.lower().endswith(".exe"):
        return True
    return shutil.which(name) is not None


def open_path(path: Path) -> None:
    if sys.platform == "darwin":
        command = ["open", str(path)]
    elif sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
        return
    elif shutil.which("xdg-open"):
        command = ["xdg-open", str(path)]
    else:
        print(path)
        return
    try:
        subprocess.run(command, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not open {path}. Use `threadsieve open ID --print` to print the path.") from exc

def find_source_path(note_path: Path) -> Path | None:
    text = note_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("local_thread_path:"):
            raw = line.split(":", 1)[1].strip().strip('"')
            return Path(raw)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
