from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .archive import archive_thread, load_thread_from_archive
from .config import Config, default_config_path, expand_path, load_config, write_default_config
from .extractor import extract_items
from .importers import import_file, import_text
from .index import get_object, index_object, index_thread, latest_thread_path, search
from .prompts import ensure_default_prompt, load_extract_prompt
from .providers import PROVIDER_PRESETS, build_provider, fetch_json, provider_request, provider_status
from .writer import append_jsonl, write_item


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
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

    show_prompt = subparsers.add_parser("show-prompt", help="Print the active extraction prompt path and contents.")
    show_prompt.add_argument("--path-only", action="store_true", help="Only print the prompt file path.")
    show_prompt.set_defaults(func=cmd_show_prompt)

    reset_prompt = subparsers.add_parser("reset-prompt", help="Create the default editable extraction prompt if missing.")
    reset_prompt.add_argument("--force", action="store_true", help="Overwrite the existing prompt with the default prompt.")
    reset_prompt.set_defaults(func=cmd_reset_prompt)

    ingest = subparsers.add_parser("ingest", help="Ingest and archive a thread file.")
    ingest.add_argument("path", help="Path to ChatGPT JSON, normalized JSON, Markdown, or text.")
    ingest.add_argument("--source-app", help="Override source app name.")
    ingest.set_defaults(func=cmd_ingest)

    extract = subparsers.add_parser("extract", help="Extract knowledge objects from a thread.")
    extract.add_argument("path", nargs="?", help="Optional file path to ingest and extract.")
    extract.add_argument("--thread", choices=["latest"], help="Extract from the latest ingested thread.")
    extract.add_argument("--file", dest="file_path", help="Explicit file path to extract.")
    extract.add_argument("--clipboard", action="store_true", help="Extract from system clipboard text.")
    extract.add_argument("--source-app", help="Override source app name for file/clipboard input.")
    extract.set_defaults(func=cmd_extract)

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
    prompt_path = ensure_default_prompt(config)
    create_workspace(config.workspace)
    print(f"Config: {path}")
    print(f"Prompt: {prompt_path}")
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


def cmd_show_prompt(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    path = ensure_default_prompt(config)
    print(path)
    if not args.path_only:
        print("")
        print(path.read_text(encoding="utf-8"))
    return 0


def cmd_reset_prompt(args: argparse.Namespace) -> int:
    from .prompts import DEFAULT_EXTRACT_PROMPT

    config = load_config(args.config)
    path = ensure_default_prompt(config)
    if args.force:
        path.write_text(DEFAULT_EXTRACT_PROMPT, encoding="utf-8")
        print(f"Reset prompt: {path}")
    else:
        print(f"Prompt exists: {path}")
        print("Use `threadsieve reset-prompt --force` to overwrite it.")
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
    config = load_config(args.config)
    create_workspace(config.workspace)
    thread, source_dir = resolve_thread_for_extract(args, config)
    if not source_dir.exists() or not (source_dir / "thread.json").exists():
        source_dir = archive_thread(config.workspace, thread)
        index_thread(config.workspace, thread, source_dir)

    items = extract_items(thread, config.extract_model, config.confidence_threshold, load_extract_prompt(config))
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
