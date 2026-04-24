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


if __name__ == "__main__":
    unittest.main()
