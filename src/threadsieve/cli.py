from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .archive import archive_thread, load_thread_from_archive
from .config import Config, default_config_path, expand_path, load_config, write_default_config
from .extractor import extract_items
from .importers import import_file, import_text
from .index import get_object, index_object, index_thread, latest_thread_path, search
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

    ingest = subparsers.add_parser("ingest", help="Ingest and archive a thread file.")
    ingest.add_argument("path", help="Path to ChatGPT JSON, normalized JSON, Markdown, or text.")
    ingest.add_argument("--source-app", help="Override source app name.")
    ingest.set_defaults(func=cmd_ingest)

    extract = subparsers.add_parser("extract", help="Extract knowledge objects from a thread.")
    extract.add_argument("path", nargs="?", help="Optional file path to ingest and extract.")
    extract.add_argument("--thread", choices=["latest"], help="Extract from the latest ingested thread.")
    extract.add_argument("--file", dest="file_path", help="Explicit file path to extract.")
    extract.add_argument("--clipboard", action="store_true", help="Extract from macOS clipboard text using pbpaste.")
    extract.add_argument("--source-app", help="Override source app name for file/clipboard input.")
    extract.set_defaults(func=cmd_extract)

    search_cmd = subparsers.add_parser("search", help="Search extracted knowledge objects.")
    search_cmd.add_argument("query", help="Search query.")
    search_cmd.add_argument("--limit", type=int, default=10)
    search_cmd.set_defaults(func=cmd_search)

    open_cmd = subparsers.add_parser("open", help="Print a knowledge object path, or open it/source on macOS.")
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
    create_workspace(config.workspace)
    print(f"Config: {path}")
    print(f"Workspace: {config.workspace}")
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

    items = extract_items(thread, config.extract_model, config.confidence_threshold)
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
    subprocess.run(["open", str(path)], check=True)
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
    try:
        result = subprocess.run(["pbpaste"], check=True, text=True, stdout=subprocess.PIPE)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not read clipboard. On macOS, make sure `pbpaste` is available.") from exc
    if not result.stdout.strip():
        raise RuntimeError("Clipboard is empty.")
    return result.stdout


def find_source_path(note_path: Path) -> Path | None:
    text = note_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("local_thread_path:"):
            raw = line.split(":", 1)[1].strip().strip('"')
            return Path(raw)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
