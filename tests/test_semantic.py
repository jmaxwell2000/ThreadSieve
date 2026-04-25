from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from threadsieve.importers import import_text
from threadsieve.semantic import offline_semantic_log, write_semantic_log


class SemanticLogTests(unittest.TestCase):
    def test_offline_semantic_log_preserves_user_and_compresses_assistant(self):
        thread = import_text(
            "User: I want the system to remember my ideas.\n"
            "Assistant: This suggests semantic memory, provenance, and extraction pipelines.\n"
            "User: Yes, but it should follow my flow of thought.",
            title="Memory",
        )

        semantic = offline_semantic_log(thread)

        self.assertIn("USER_STATEMENT: I want the system to remember my ideas.", semantic.text)
        self.assertIn("AI_CONTEXT:", semantic.text)
        self.assertIn("- ACTION:", semantic.text)
        self.assertIn("- CONCEPTS_INTRODUCED:", semantic.text)
        self.assertIn("- NEXT_USER_REF:", semantic.text)
        self.assertIn("- NEXT_USER_REACTION:", semantic.text)
        self.assertEqual(semantic.extraction_thread.messages[0].content, "I want the system to remember my ideas.")
        self.assertNotIn("semantic memory, provenance, and extraction pipelines", semantic.extraction_thread.messages[1].content)

    def test_offline_semantic_log_preserves_revised_assistant_artifact(self):
        thread = import_text(
            "User: Draft a future LLM prompt.\n"
            "Assistant: Prompt draft:\n"
            "1. Use non-anthropomorphic language.\n"
            "2. Ask the user five preference questions.\n"
            "3. Prefer logic-bound reasoning.\n"
            "User: Remove the questioning section and make the logic mode innate.",
            title="Artifact",
        )

        semantic = offline_semantic_log(thread)

        self.assertIn("AI_ARTIFACT:", semantic.text)
        self.assertIn("- ARTIFACT_TYPE: prompt", semantic.text)
        self.assertIn("- CONTENT_EXCERPT:", semantic.text)
        self.assertIn("- CONTENT_HASH:", semantic.text)
        self.assertIn("Ask the user five preference questions", semantic.extraction_thread.messages[1].content)

    def test_write_semantic_log(self):
        thread = import_text("User: Preserve this.", title="Write Log")
        semantic = offline_semantic_log(thread)
        with tempfile.TemporaryDirectory() as directory:
            path = write_semantic_log(Path(directory), thread, semantic.text)

            self.assertTrue(path.exists())
            self.assertIn("USER_STATEMENT", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
