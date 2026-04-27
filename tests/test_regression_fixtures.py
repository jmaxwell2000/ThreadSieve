from pathlib import Path
import re
import unittest

import _bootstrap  # noqa: F401

from threadsieve.extractor import validate_items
from threadsieve.importers import import_file
from threadsieve.cli import build_parser, find_tests_dir
from threadsieve.semantic import looks_like_artifact, offline_semantic_log, sanitize_semantic_log_text


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "regression"


def load_fixture(name: str):
    return import_file(FIXTURE_DIR / name)


def user_messages(thread):
    return [message for message in thread.messages if message.role == "user"]


class RegressionFixtureTests(unittest.TestCase):
    def test_fixture_privacy_rule_is_present(self):
        readme = (FIXTURE_DIR / "README.md").read_text(encoding="utf-8")

        self.assertIn("synthetic or sanitized", readme)
        self.assertIn("Do not commit real chat exports", readme)

    def test_regression_command_is_registered(self):
        args = build_parser().parse_args(["regression"])

        self.assertEqual(args.command, "regression")
        self.assertEqual(find_tests_dir(), Path(__file__).parent)

    def test_semantic_logs_preserve_every_user_message_verbatim(self):
        for path in sorted(FIXTURE_DIR.glob("sample-*.md")):
            with self.subTest(path=path.name):
                thread = import_file(path)
                semantic = offline_semantic_log(thread)

                for message in user_messages(thread):
                    self.assertIn(f"## {message.id} USER_STATEMENT", semantic.text)
                    self.assertIn(message.content, semantic.text)
                    transformed = next(item for item in semantic.extraction_thread.messages if item.id == message.id)
                    self.assertEqual(transformed.content, message.content)

    def test_example_continuation_stays_context(self):
        thread = load_fixture("sample-example-continuation.md")
        first_assistant = thread.messages[1]
        next_user = thread.messages[2]
        mislabeled_log = (
            "# Semantic Log: Sample Example Continuation\n\n"
            f"## {thread.messages[0].id} USER_STATEMENT\n"
            f"{thread.messages[0].content}\n\n"
            f"## {first_assistant.id} AI_ARTIFACT\n"
            "- ARTIFACT_TYPE: examples\n"
            "- ACTION: Provided examples\n"
            "- CONTENT_EXCERPT: Example One...\n"
            "- CONTENT_HASH: abc123\n"
            f"- NEXT_USER_REF: {next_user.id}\n"
            "- NEXT_USER_REACTION: User asks to continue.\n"
        )

        sanitized = sanitize_semantic_log_text(thread, mislabeled_log)

        self.assertFalse(looks_like_artifact(first_assistant.content, next_user))
        self.assertIn(f"## {first_assistant.id} AI_CONTEXT", sanitized)
        self.assertNotIn(f"## {first_assistant.id} AI_ARTIFACT", sanitized)
        for message in thread.messages:
            self.assertIn(f"## {message.id} ", sanitized)

    def test_artifact_refinement_preserves_assistant_draft_as_artifact(self):
        thread = load_fixture("sample-artifact-refinement.md")
        semantic = offline_semantic_log(thread)
        assistant_draft = thread.messages[1]

        self.assertIn(f"## {assistant_draft.id} AI_ARTIFACT", semantic.text)
        transformed = next(message for message in semantic.extraction_thread.messages if message.id == assistant_draft.id)
        self.assertTrue(transformed.metadata["semantic_artifact"])
        self.assertIn("app-preference question", thread.messages[2].content)
        self.assertIn("Ask the user which note-taking app", transformed.content)

    def test_directive_framework_strengthening_uses_synthetic_fixture(self):
        thread = load_fixture("sample-directive-framework.md")
        artifact_message = thread.messages[0]
        raw_items = [
            {
                "type": "framework",
                "title": "Compact Audio Protocol",
                "summary": "A compact response protocol.",
                "body": ["generic", "too short"],
                "canonical_statement": "A compact audio protocol.",
                "object_role": "artifact_spec",
                "tags": ["protocol"],
                "origin": "user",
                "evidence": [artifact_message.content, artifact_message.id],
                "source_refs": [
                    {
                        "message_id": artifact_message.id,
                        "start_char": 0,
                        "end_char": len(artifact_message.content),
                        "ref_type": "evidence",
                    }
                ],
                "confidence": 1.0,
            }
        ]

        items = validate_items(thread, raw_items, threshold=0.0)

        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.type, "framework")
        self.assertEqual(item.object_role, "artifact_spec")
        for directive in ["Zero Preamble", "Brevity Limit", "Signal Words", "Failure Mode"]:
            self.assertIn(directive, item.summary + "\n" + (item.body or "") + "\n" + (item.canonical_statement or ""))
        self.assertNotRegex("\n".join(item.evidence), r"^msg_[a-f0-9_]+$")
        self.assertNotIn("[", item.body or "")

    def test_no_fixture_contains_obvious_private_material(self):
        forbidden_patterns = [
            r"1066",
            r"\bGabe\b",
            r"\bTeresita\b",
            r"api[_ -]?key",
            r"sk-[A-Za-z0-9]",
        ]
        for path in sorted(FIXTURE_DIR.glob("sample-*.md")):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                for pattern in forbidden_patterns:
                    self.assertIsNone(re.search(pattern, text, re.IGNORECASE), pattern)

    def test_long_mixed_fixture_exercises_multiple_patterns(self):
        thread = load_fixture("sample-long-mixed-conversation.md")
        semantic = offline_semantic_log(thread)

        self.assertGreaterEqual(len(thread.messages), 20)
        for message in thread.messages:
            self.assertIn(f"## {message.id} ", semantic.text)
        for message in user_messages(thread):
            self.assertIn(message.content, semantic.text)

        example_assistant = next(message for message in thread.messages if "Example One:" in message.content)
        next_user = thread.messages[example_assistant.index + 1]
        self.assertEqual(next_user.role, "user")
        self.assertFalse(looks_like_artifact(example_assistant.content, next_user))

        draft_assistant = next(message for message in thread.messages if "Prompt draft:" in message.content)
        revised_semantic_message = next(message for message in semantic.extraction_thread.messages if message.id == draft_assistant.id)
        self.assertTrue(revised_semantic_message.metadata["semantic_artifact"])
        self.assertIn("Summarize the whole conversation", revised_semantic_message.content)

        protocol_message = next(message for message in thread.messages if "Compact Review Protocol" in message.content)
        raw_items = [
            {
                "type": "framework",
                "title": "Compact Review Protocol",
                "summary": "A review protocol.",
                "body": "Too generic.",
                "canonical_statement": "A review protocol.",
                "object_role": "artifact_spec",
                "tags": ["review"],
                "origin": "user",
                "evidence": [protocol_message.content, protocol_message.id],
                "source_refs": [{"message_id": protocol_message.id, "start_char": 0, "end_char": len(protocol_message.content)}],
                "confidence": 1.0,
            }
        ]
        items = validate_items(thread, raw_items, threshold=0.0)

        self.assertEqual(len(items), 1)
        for directive in ["Source First", "Small Objects", "Review Flag", "No Private Data"]:
            self.assertIn(directive, items[0].summary + "\n" + (items[0].body or "") + "\n" + (items[0].canonical_statement or ""))


if __name__ == "__main__":
    unittest.main()
