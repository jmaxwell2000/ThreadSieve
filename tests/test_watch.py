from contextlib import redirect_stdout
from pathlib import Path
import io
import tempfile
import unittest

import _bootstrap  # noqa: F401

from threadsieve.cli import build_parser, main
from threadsieve.watch import discover_watch_files, stable_ready_files


class WatchTests(unittest.TestCase):
    def test_watch_command_is_registered_with_expected_flags(self):
        args = build_parser().parse_args(
            [
                "watch",
                "--source",
                "./incoming",
                "--out",
                "./knowledge",
                "--provider",
                "offline",
                "--model",
                "test-model",
                "--extractor",
                "idea",
                "--force",
                "--no-semantic-log",
                "--interval",
                "1",
                "--settle-seconds",
                "0",
                "--once",
            ]
        )

        self.assertEqual(args.command, "watch")
        self.assertEqual(args.source, "./incoming")
        self.assertEqual(args.out, "./knowledge")
        self.assertEqual(args.provider, "offline")
        self.assertEqual(args.model, "test-model")
        self.assertEqual(args.extractor, ["idea"])
        self.assertTrue(args.force)
        self.assertTrue(args.no_semantic_log)
        self.assertEqual(args.interval, 1)
        self.assertEqual(args.settle_seconds, 0)
        self.assertTrue(args.once)

    def test_watch_discovery_ignores_unsupported_suffixes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            markdown = source / "chat.md"
            text = source / "note.txt"
            ignored = source / "image.png"
            nested = source / "nested"
            nested.mkdir()
            nested_markdown = nested / "nested.markdown"
            for path in [markdown, text, ignored, nested_markdown]:
                path.write_text("User: Does this work?", encoding="utf-8")

            files = discover_watch_files(source)

        self.assertEqual([path.name for path in files], ["chat.md", "nested.markdown", "note.txt"])

    def test_ready_file_logic_waits_until_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            chat = source / "chat.md"
            chat.write_text("User: Does this work?", encoding="utf-8")
            snapshots = {}

            first = stable_ready_files(source, snapshots, settle_seconds=2, now=10)
            second = stable_ready_files(source, snapshots, settle_seconds=2, now=11)
            third = stable_ready_files(source, snapshots, settle_seconds=2, now=12.1)
            chat.write_text("User: Does this work now?", encoding="utf-8")
            changed = stable_ready_files(source, snapshots, settle_seconds=2, now=13)

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual([path.name for path in third], ["chat.md"])
        self.assertEqual(changed, [])

    def test_watch_rejects_file_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "chat.md"
            source.write_text("User: Does this work?", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "requires a folder"):
                discover_watch_files(source)

    def test_watch_once_processes_supported_file(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            source = tmp / "incoming"
            output = tmp / "knowledge"
            source.mkdir()
            (source / "chat.md").write_text("User: Can watch preserve source links?", encoding="utf-8")

            with redirect_stdout(io.StringIO()) as stdout:
                result = main(
                    [
                        "watch",
                        "--source",
                        str(source),
                        "--out",
                        str(output),
                        "--provider",
                        "offline",
                        "--once",
                        "--interval",
                        "0.1",
                        "--settle-seconds",
                        "0",
                    ]
                )

            output_text = stdout.getvalue()

            self.assertEqual(result, 0)
            self.assertIn("ThreadSieve extract", output_text)
            self.assertTrue((output / "index.jsonl").exists())
            self.assertTrue(list(output.rglob("*.md")))

    def test_watch_once_state_store_skips_second_run(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            source = tmp / "incoming"
            output = tmp / "knowledge"
            source.mkdir()
            (source / "chat.md").write_text("User: Can watch preserve source links?", encoding="utf-8")
            command = [
                "watch",
                "--source",
                str(source),
                "--out",
                str(output),
                "--provider",
                "offline",
                "--once",
                "--interval",
                "0.1",
                "--settle-seconds",
                "0",
            ]

            with redirect_stdout(io.StringIO()):
                first = main(command)
            with redirect_stdout(io.StringIO()) as stdout:
                second = main(command)

        self.assertEqual(first, 0)
        self.assertEqual(second, 0)
        self.assertIn("Skipped: 1 file(s)", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
