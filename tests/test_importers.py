from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from threadsieve.importers import import_file, import_text


class ImporterTests(unittest.TestCase):
    def test_import_text_role_blocks(self):
        thread = import_text("User: Hello\nAssistant: Hi", title="Demo")

        self.assertEqual(thread.title, "Demo")
        self.assertEqual(len(thread.messages), 2)
        self.assertEqual(thread.messages[0].role, "user")
        self.assertEqual(thread.messages[1].role, "assistant")
        self.assertTrue(thread.messages[0].id.startswith("msg_"))

    def test_import_chatgpt_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chatgpt.json"
            path.write_text(
                """
                {
                  "id": "abc",
                  "title": "Export",
                  "mapping": {
                    "one": {
                      "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["Build ThreadSieve."]}
                      }
                    }
                  }
                }
                """,
                encoding="utf-8",
            )

            thread = import_file(path)

        self.assertEqual(thread.source_app, "chatgpt")
        self.assertEqual(thread.title, "Export")
        self.assertEqual(thread.messages[0].content, "Build ThreadSieve.")

    def test_import_text_generic_speakers(self):
        thread = import_text("Gemini: Try this approach.\nJamie: Nice, what next?", title="Generic")

        self.assertEqual(len(thread.messages), 2)
        self.assertEqual(thread.messages[0].role, "assistant")
        self.assertEqual(thread.messages[1].role, "jamie")

    def test_import_text_dash_speaker_delimiters(self):
        thread = import_text(
            "User: - I have an app idea.\nAssistant- Great, what problem does it solve?\nUser: - Tool sharing.",
            title="Dash delimiters",
        )

        self.assertEqual(len(thread.messages), 3)
        self.assertEqual(thread.messages[0].role, "user")
        self.assertEqual(thread.messages[0].content, "I have an app idea.")
        self.assertEqual(thread.messages[1].role, "assistant")
        self.assertEqual(thread.messages[1].content, "Great, what problem does it solve?")

    def test_import_markdown_chat_headers_with_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.md"
            path.write_text(
                """# New chat
**Chat ID:** 691c85c0-a260-832d-82e0-86a6a0f48e60
**Date:** 2025-11-18

### User (2025-11-18 09:42:10)
Hi

### Assistant (2025-11-18 09:42:11)
Hello.
""",
                encoding="utf-8",
            )

            thread = import_file(path)

        self.assertEqual(thread.id, "691c85c0-a260-832d-82e0-86a6a0f48e60")
        self.assertEqual(thread.source_app, "markdown_chat")
        self.assertEqual(thread.messages[0].role, "user")
        self.assertEqual(thread.messages[0].timestamp, "2025-11-18 09:42:10")
        self.assertEqual(thread.messages[1].content, "Hello.")
        self.assertIn("start_char", thread.messages[0].metadata)


if __name__ == "__main__":
    unittest.main()
