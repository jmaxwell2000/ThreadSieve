from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from threadsieve.importers import import_text
from threadsieve.prompts import DEFAULT_SEMANTIC_PROMPT
from threadsieve.semantic import (
    looks_like_artifact,
    offline_semantic_log,
    sanitize_semantic_log_text,
    thread_from_semantic_text,
    write_semantic_log,
)


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

    def test_continue_examples_is_not_artifact_revision(self):
        thread = import_text(
            "User: Give examples.\n"
            "Assistant: Example One: Weather status. Example Two: Schedule optimize.\n"
            "User: Continue.",
            title="Examples",
        )

        self.assertFalse(looks_like_artifact(thread.messages[1].content, thread.messages[2]))
        self.assertIn("more examples", DEFAULT_SEMANTIC_PROMPT)

    def test_sanitize_downgrades_continuation_examples(self):
        thread = import_text(
            "User: Give examples.\n"
            "Assistant: Example One: Weather status. Example Two: Schedule optimize.\n"
            "User: Continue.",
            title="Examples",
        )
        semantic_text = (
            "# Semantic Log: Examples\n\n"
            f"## {thread.messages[0].id} USER_STATEMENT\n"
            "Give examples.\n"
            f"## {thread.messages[1].id} AI_ARTIFACT\n"
            "AI_ARTIFACT:\n"
            "- ARTIFACT_TYPE: examples\n"
            "- ACTION: Provide examples\n"
            "- CONTENT_EXCERPT: Example One...\n"
            "- CONTENT_HASH: abc123\n"
            f"- NEXT_USER_REF: {thread.messages[2].id}\n"
            "- NEXT_USER_REACTION: User asks to continue.\n"
            f"## {thread.messages[2].id} USER_STATEMENT\n"
            "Continue.\n"
        )

        sanitized = sanitize_semantic_log_text(thread, semantic_text)

        self.assertIn(f"## {thread.messages[1].id} AI_CONTEXT", sanitized)
        self.assertNotIn(f"## {thread.messages[1].id} AI_ARTIFACT", sanitized)

    def test_write_semantic_log(self):
        thread = import_text("User: Preserve this.", title="Write Log")
        semantic = offline_semantic_log(thread)
        with tempfile.TemporaryDirectory() as directory:
            path = write_semantic_log(Path(directory), thread, semantic.text)

            self.assertTrue(path.exists())
            self.assertIn("USER_STATEMENT", path.read_text(encoding="utf-8"))

    def test_fenced_semantic_log_recovers_to_original_message_ids(self):
        thread = import_text(
            "User: Draft a compact protocol.\n"
            "Assistant: Example plan:\n- Step one\n- Step two\n"
            "User: Remove step two.",
            title="Fenced",
        )
        semantic_text = """# Semantic Log: Fenced

```USER_STATEMENT
Draft a compact protocol.
```
```AI_ARTIFACT
- ARTIFACT_TYPE: plan
- ACTION: Proposed plan
- CONTENT_EXCERPT: Example plan with two steps.
- CONTENT_HASH: abc123
- NEXT_USER_REF: msg_fake
- NEXT_USER_REACTION: User removes step two.
```
```USER_STATEMENT
Remove step two.
```
"""

        recovered = thread_from_semantic_text(thread, semantic_text)

        self.assertEqual([message.id for message in recovered.messages], [message.id for message in thread.messages])
        self.assertEqual(recovered.messages[0].content, "Draft a compact protocol.")
        self.assertTrue(recovered.messages[1].metadata["semantic_artifact"])
        self.assertTrue(recovered.messages[1].metadata["semantic_recovered_from_fence"])


if __name__ == "__main__":
    unittest.main()
