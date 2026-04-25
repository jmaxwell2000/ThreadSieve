from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from threadsieve.archive import archive_thread
from threadsieve.extractor import build_extraction_messages, extract_items, parse_model_json, repair_span
from threadsieve.importers import import_text
from threadsieve.index import index_object, index_thread, search
from threadsieve.models import KnowledgeItem
from threadsieve.pipeline import extract_sources, find_object_record, trace_object
from threadsieve.writer import write_item


class PipelineTests(unittest.TestCase):
    def test_offline_extract_writes_and_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            thread = import_text(
                "User: Can this preserve source links?\nAssistant: Yes, cite message spans in every extracted object.",
                title="Source links",
            )
            source_dir = archive_thread(tmp_path, thread)
            index_thread(tmp_path, thread, source_dir)

            items = extract_items(thread, {"provider": "offline"}, threshold=0.5)

            self.assertTrue(items)
            note_path = write_item(tmp_path, items[0], thread, source_dir)
            index_object(tmp_path, items[0], thread, note_path)

            matches = search(tmp_path, "source")
            self.assertTrue(matches)
            self.assertEqual(matches[0]["local_path"], str(note_path))

    def test_offline_extract_does_not_create_chunk_summary_note(self):
        thread = import_text("User: I have a product idea.\nAssistant: Here is a long evaluation.", title="No Chunk")

        items = extract_items(thread, {"provider": "offline"}, threshold=0.0)

        self.assertFalse(any(item.type == "project_note" for item in items))

    def test_model_json_parses_markdown_fence(self):
        parsed = parse_model_json('Sure:\n```json\n{"items": []}\n```')

        self.assertEqual(parsed, {"items": []})

    def test_extraction_messages_keep_json_protocol_when_prompt_is_custom(self):
        thread = import_text("User: Capture this idea.", title="Custom prompt")

        messages = build_extraction_messages(thread, "Extract only decisions.")

        self.assertIn("json", " ".join(message["content"].lower() for message in messages))
        payload = messages[-1]["content"]
        self.assertIn("canonical_statement", payload)
        self.assertIn("ref_type", payload)

    def test_knowledge_item_accepts_extended_schema_fields(self):
        item = KnowledgeItem.from_dict(
            {
                "type": "framework",
                "title": "Logic-Prior Mode",
                "summary": "A durable prompt framework.",
                "tags": ["llm"],
                "confidence": 0.9,
                "object_role": "artifact_spec",
                "canonical_statement": "Use a logic-prior non-anthropomorphic mode.",
                "supersedes": ["idea_old"],
                "extraction_rationale": "Merged from multiple refinement turns.",
                "thread_position": {"first_message_index": 1, "last_message_index": 3},
                "source_refs": [
                    {
                        "message_id": "msg_1",
                        "start_char": 0,
                        "end_char": 10,
                        "ref_type": "revision_instruction",
                    }
                ],
            },
            "framework_123",
        )

        self.assertEqual(item.object_role, "artifact_spec")
        self.assertEqual(item.source_refs[0].ref_type, "revision_instruction")
        self.assertEqual(item.thread_position["last_message_index"], 3)

    def test_span_repair_prefers_exact_text(self):
        content = "Alpha beta gamma delta."
        start, end = repair_span(content, {"exact_text": "gamma delta"}, 0, 5)

        self.assertEqual(content[start:end], "gamma delta")

    def test_span_repair_fuzzy_matches_whitespace(self):
        content = "Alpha beta gamma\n  delta."
        start, end = repair_span(content, {"exact_text": "gamma delta"}, 0, 5)

        self.assertEqual(" ".join(content[start:end].split()), "gamma delta")

    def test_source_out_pipeline_writes_state_index_and_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = tmp_path / "chat.md"
            output = tmp_path / "knowledge"
            source.write_text(
                "User: Can this preserve source links?\nAssistant: Yes, cite message spans in every extracted object.",
                encoding="utf-8",
            )

            summary = extract_sources(source, output, {"provider": "offline"}, threshold=0.5)

            self.assertEqual(summary["source_files_processed"], 1)
            self.assertTrue(summary["warnings"])
            self.assertIn("Provider is offline", summary["warnings"][0])
            self.assertTrue((output / ".threadsieve" / "state.json").exists())
            self.assertTrue((output / "index.jsonl").exists())
            self.assertTrue((output / "semantic_logs").exists())
            self.assertTrue(summary["semantic_logs_created"])
            self.assertTrue(summary["created_files"])

            record = find_object_record(output, Path(summary["created_files"][0]).stem)
            self.assertIsNotNone(record)
            self.assertTrue(record["source_refs"])
            traced = trace_object(output, str(record["id"]))
            self.assertIn("Source:", traced)
            self.assertIn("preserve source links", traced)

            skipped = extract_sources(source, output, {"provider": "offline"}, threshold=0.5)
            self.assertEqual(skipped["source_files_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
