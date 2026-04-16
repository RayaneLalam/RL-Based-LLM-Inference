"""Unit tests for environment-driven model-pair selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_config import resolve_model_pair


class ModelConfigTests(unittest.TestCase):
    def test_default_profile_uses_two_model_m1_pair(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            draft_model, target_model, profile = resolve_model_pair()

        self.assertEqual(profile, "m1_tiny_two_models")
        self.assertEqual(draft_model, "sshleifer/tiny-gpt2")
        self.assertEqual(target_model, "distilgpt2")

    def test_profile_env_changes_pair(self) -> None:
        with patch.dict("os.environ", {"PARETOSERVE_MODEL_PROFILE": "m1_tiny_same"}, clear=True):
            draft_model, target_model, profile = resolve_model_pair()

        self.assertEqual(profile, "m1_tiny_same")
        self.assertEqual(draft_model, "sshleifer/tiny-gpt2")
        self.assertEqual(target_model, "sshleifer/tiny-gpt2")

    def test_explicit_env_overrides_profile(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "PARETOSERVE_MODEL_PROFILE": "m1_tiny_same",
                "PARETOSERVE_DRAFT_MODEL": "sshleifer/tiny-gpt2",
                "PARETOSERVE_TARGET_MODEL": "distilgpt2",
            },
            clear=True,
        ):
            draft_model, target_model, profile = resolve_model_pair()

        self.assertEqual(profile, "m1_tiny_same")
        self.assertEqual(draft_model, "sshleifer/tiny-gpt2")
        self.assertEqual(target_model, "distilgpt2")


if __name__ == "__main__":
    unittest.main()
