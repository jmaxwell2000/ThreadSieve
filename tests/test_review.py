from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest

import _bootstrap  # noqa: F401

from threadsieve.cli import build_parser, main
from threadsieve.importers import import_markdown_chat
from threadsieve.pipeline import rebuild_index
from threadsieve.review import format_review_detail, list_review_objects, review_object_record, update_review_status
from threadsieve.writer import to_yaml_like


class ReviewTests(unittest.TestCase):
    def test_review_command_is_registered(self):
        args = build_parser().parse_args(["review", "--knowledge", "./knowledge", "--limit", "5", "--type", "idea"])

        self.assertEqual(args.command, "review")
        self.assertEqual(args.knowledge, "./knowledge")
        self.assertEqual(args.limit, 5)
        self.assertEqual(args.item_type, "idea")

    def test_review_list_finds_raw_and_needs_review_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            knowledge = Path(directory) / "knowledge"
            write_object(knowledge / "ideas" / "idea_raw.md", "idea_raw", "idea", "raw", 0.91)
            write_object(knowledge / "ideas" / "idea_accepted.md", "idea_accepted", "idea", "accepted", 0.99)
            write_object(knowledge / "_needs_review" / "idea_low.md", "idea_low", "idea", "raw", 0.42)

            records = list_review_objects(knowledge)

        self.assertEqual([record["id"] for record in records], ["idea_low", "idea_raw"])
        self.assertTrue(records[0]["needs_review"])

    def test_review_show_resolves_object_and_includes_trace_context(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            knowledge = tmp / "knowledge"
            source = tmp / "source.md"
            source.write_text(
                "# Review Source\n"
                "**Chat ID:** review-source\n\n"
                "### User (2026-01-01 00:00:00)\n"
                "Keep source links mandatory.\n",
                encoding="utf-8",
            )
            thread = import_markdown_chat(source.read_text(encoding="utf-8"), title="Review Source", source_uri=str(source))
            self.assertIsNotNone(thread)
            assert thread is not None
            message = thread.messages[0]
            object_path = write_object(
                knowledge / "ideas" / "idea_trace.md",
                "idea_trace",
                "idea",
                "raw",
                0.9,
                source_ref={
                    "message_id": message.id,
                    "start_char": 0,
                    "end_char": len(message.content),
                    "source_path": str(source),
                    "role": "user",
                },
            )
            rebuild_index(knowledge)

            record = review_object_record(knowledge, "idea_trace")
            detail = format_review_detail(knowledge, record)

        self.assertEqual(record["path"], str(object_path))
        self.assertIn("Keep source links mandatory", detail)
        self.assertIn("Trace:", detail)

    def test_review_status_update_preserves_body_and_rebuilds_index(self):
        with tempfile.TemporaryDirectory() as directory:
            knowledge = Path(directory) / "knowledge"
            object_path = write_object(knowledge / "ideas" / "idea_status.md", "idea_status", "idea", "raw", 0.88, body="Body stays here.")
            rebuild_index(knowledge)

            updated = update_review_status(knowledge, "idea_status", "accepted")

            text = object_path.read_text(encoding="utf-8")
            index_records = [json.loads(line) for line in (knowledge / "index.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(updated["status"], "accepted")
        self.assertIn("Body stays here.", text)
        self.assertEqual(index_records[0]["status"], "accepted")

    def test_review_status_cli_updates_object(self):
        with tempfile.TemporaryDirectory() as directory:
            knowledge = Path(directory) / "knowledge"
            write_object(knowledge / "ideas" / "idea_cli.md", "idea_cli", "idea", "raw", 0.88)
            rebuild_index(knowledge)

            with redirect_stdout(io.StringIO()):
                result = main(["review", "idea_cli", "--knowledge", str(knowledge), "--status", "rejected"])

            record = review_object_record(knowledge, "idea_cli")

        self.assertEqual(result, 0)
        self.assertEqual(record["status"], "rejected")


def write_object(
    path: Path,
    object_id: str,
    item_type: str,
    status: str,
    confidence: float,
    body: str = "Body.",
    source_ref: dict[str, object] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_ref = source_ref or {"message_id": "msg_1", "start_char": 0, "end_char": 4}
    path.write_text(
        "---\n"
        + to_yaml_like(
            {
                "id": object_id,
                "type": item_type,
                "title": f"{object_id} title",
                "summary": f"{object_id} summary",
                "status": status,
                "confidence": confidence,
                "source_refs": [source_ref],
            }
        )
        + "---\n\n"
        + body
        + "\n",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
