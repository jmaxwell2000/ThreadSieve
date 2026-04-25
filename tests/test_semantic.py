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
        self.assertEqual(semantic.extraction_thread.messages[0].content, "I want the system to remember my ideas.")
        self.assertNotIn("semantic memory, provenance, and extraction pipelines", semantic.extraction_thread.messages[1].content)

    def test_write_semantic_log(self):
        thread = import_text("User: Preserve this.", title="Write Log")
        semantic = offline_semantic_log(thread)
        with tempfile.TemporaryDirectory() as directory:
            path = write_semantic_log(Path(directory), thread, semantic.text)

            self.assertTrue(path.exists())
            self.assertIn("USER_STATEMENT", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
