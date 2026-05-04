"""Environment-driven model-pair configuration for real-model runs."""

from __future__ import annotations

import os

MODEL_PROFILES: dict[str, tuple[str, str]] = {
    "m1_tiny_same": ("sshleifer/tiny-gpt2", "sshleifer/tiny-gpt2"),
    "m1_tiny_two_models": ("sshleifer/tiny-gpt2", "distilgpt2"),
    "balanced_gpt2": ("distilgpt2", "gpt2"),
    "medium_models": ("gpt2", "meta-llama/Llama-3.1-8B"),
}
DEFAULT_MODEL_PROFILE = "balanced_gpt2"


def _read_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def resolve_model_pair(
    env_prefix: str = "PARETOSERVE",
    fallback_profile: str = DEFAULT_MODEL_PROFILE,
) -> tuple[str, str, str]:
    """Resolve a draft/target model pair from environment variables.

    Priority:
    1. `<PREFIX>_DRAFT_MODEL` and `<PREFIX>_TARGET_MODEL`
    2. `<PREFIX>_MODEL_PROFILE`
    3. `fallback_profile`
    """

    profile_name = _read_env(f"{env_prefix}_MODEL_PROFILE") or fallback_profile
    profile_pair = MODEL_PROFILES.get(profile_name, MODEL_PROFILES[fallback_profile])
    draft_model = _read_env(f"{env_prefix}_DRAFT_MODEL") or profile_pair[0]
    target_model = _read_env(f"{env_prefix}_TARGET_MODEL") or profile_pair[1]
    return draft_model, target_model, profile_name
