import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import _bootstrap  # noqa: F401

from threadsieve.cli import main
from threadsieve.config import load_config
from threadsieve.prompts import ensure_default_prompt
from threadsieve.providers import build_provider, provider_status


class ProviderTests(unittest.TestCase):
    def test_openrouter_preset_loads_api_key_from_environment(self):
        old_value = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        try:
            provider = build_provider({"provider": "openrouter"})
        finally:
            if old_value is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = old_value

        self.assertEqual(provider.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(provider.api_key, "test-key")
        self.assertIn("HTTP-Referer", provider.headers)

    def test_provider_status_does_not_expose_api_key(self):
        provider = build_provider({"provider": "openrouter", "api_key": "secret"})

        status = provider_status(provider)

        self.assertTrue(status["api_key_loaded"])
        self.assertNotIn("secret", str(status))

    def test_configure_provider_command_writes_preset(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            with redirect_stdout(StringIO()):
                result = main(["--config", str(config_path), "configure-provider", "ollama", "--model", "llama3.2"])

            self.assertEqual(result, 0)
            self.assertIn('"provider": "ollama"', config_path.read_text(encoding="utf-8"))
            self.assertIn('"model": "llama3.2"', config_path.read_text(encoding="utf-8"))

    def test_default_prompt_file_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            with redirect_stdout(StringIO()):
                result = main(["--config", str(config_path), "init", "--workspace", str(Path(directory) / "workspace")])

            config = load_config(str(config_path))
            prompt_path = ensure_default_prompt(config)
            semantic_prompt_path = ensure_default_prompt(config, "semantic")

            self.assertEqual(result, 0)
            self.assertTrue(prompt_path.exists())
            self.assertTrue(semantic_prompt_path.exists())
            self.assertIn("Return JSON only", prompt_path.read_text(encoding="utf-8"))
            self.assertIn("USER_STATEMENT", semantic_prompt_path.read_text(encoding="utf-8"))

    def test_show_and_reset_semantic_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            with redirect_stdout(StringIO()):
                main(["--config", str(config_path), "init", "--workspace", str(Path(directory) / "workspace")])

            config = load_config(str(config_path))
            semantic_prompt = ensure_default_prompt(config, "semantic")
            semantic_prompt.write_text("custom semantic prompt", encoding="utf-8")

            shown = StringIO()
            with redirect_stdout(shown):
                result = main(["--config", str(config_path), "show-prompt", "--kind", "semantic"])
            self.assertEqual(result, 0)
            self.assertIn("custom semantic prompt", shown.getvalue())

            with redirect_stdout(StringIO()):
                result = main(["--config", str(config_path), "reset-prompt", "--kind", "semantic", "--force"])
            self.assertEqual(result, 0)
            self.assertIn("USER_STATEMENT", semantic_prompt.read_text(encoding="utf-8"))

    def test_yaml_config_loads_sources_output_and_provider_name(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "threadsieve.yaml"
            config_path.write_text(
                """sources:
  - ./chats
output: ./knowledge
provider:
  name: openrouter
  model: openai/gpt-4.1-mini
extractors:
  - idea
  - task
behavior:
  needs_review_confidence_threshold: 0.75
""",
                encoding="utf-8",
            )

            config = load_config(str(config_path))

        self.assertEqual(config.raw["sources"], ["./chats"])
        self.assertEqual(config.raw["output"], "./knowledge")
        self.assertEqual(config.raw["provider"]["name"], "openrouter")
        self.assertEqual(config.raw["extractors"], ["idea", "task"])


if __name__ == "__main__":
    unittest.main()
