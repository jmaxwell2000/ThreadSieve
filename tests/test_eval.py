from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from threadsieve.cli import build_parser
from threadsieve.eval import DEFAULT_EVAL_MODELS, estimate_model_calls, eval_fixture_files, run_live_eval


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "regression"


class EvalTests(unittest.TestCase):
    def test_default_eval_models_are_current_budget_set(self):
        self.assertEqual(
            DEFAULT_EVAL_MODELS,
            [
                "openai/gpt-5-mini",
                "google/gemini-3.1-flash-lite-preview",
                "qwen/qwen3-30b-a3b",
            ],
        )

    def test_eval_fixtures_only_include_sample_markdown(self):
        files = eval_fixture_files(FIXTURE_DIR)

        self.assertTrue(files)
        self.assertTrue(all(path.name.startswith("sample-") for path in files))
        self.assertFalse(any(path.name == "README.md" for path in files))

    def test_eval_call_estimate_uses_two_stage_pipeline(self):
        files = eval_fixture_files(FIXTURE_DIR)

        self.assertEqual(estimate_model_calls(len(files), len(DEFAULT_EVAL_MODELS)), len(files) * len(DEFAULT_EVAL_MODELS) * 2)

    def test_eval_command_is_registered(self):
        args = build_parser().parse_args(["eval", "--model", "openai/gpt-5-mini", "--max-calls", "75"])

        self.assertEqual(args.command, "eval")
        self.assertEqual(args.model, ["openai/gpt-5-mini"])
        self.assertEqual(args.max_calls, 75)

    def test_eval_respects_call_budget_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "above --max-calls"):
                run_live_eval(
                    fixtures=FIXTURE_DIR,
                    output_root=Path(directory),
                    provider="openrouter",
                    models=DEFAULT_EVAL_MODELS,
                    model_config_base={"provider": "openrouter", "api_key": "not-used"},
                    threshold=0.75,
                    max_calls=1,
                )


if __name__ == "__main__":
    unittest.main()
