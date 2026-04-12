"""Raw-state conventions and tabular featurization for Sprint 1.

The raw state remains a semantic dictionary shared across modules.
`featurize_state` converts that raw state into a compact discrete key that
the tabular Q-learning controller can use internally.
"""

from __future__ import annotations

import random
from typing import Any, Mapping

K_VALUES = (1, 2, 3, 4, 5)
EMA_ALPHA = 0.35


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a float into a closed interval."""
    return max(lower, min(upper, value))


def update_ema(previous: float, current: float, alpha: float = EMA_ALPHA) -> float:
    """Update the exponential moving average of acceptance history."""
    return alpha * current + (1.0 - alpha) * previous


def initialize_state(
    workload_style: str = "steady_low_load",
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Create a raw environment state with Sprint 1 semantics."""
    rng = rng or random.Random()

    if workload_style == "bursty_high_load":
        queue_length = rng.randint(3, 7)
        batch_size = rng.randint(2, 6)
        arrival_rate = rng.uniform(0.65, 1.15)
        system_load = rng.uniform(0.50, 0.78)
        sla_slack_ms = rng.uniform(45.0, 120.0)
        branching_factor = rng.randint(2, 5)
    else:
        queue_length = rng.randint(0, 2)
        batch_size = rng.randint(1, 3)
        arrival_rate = rng.uniform(0.15, 0.45)
        system_load = rng.uniform(0.15, 0.35)
        sla_slack_ms = rng.uniform(140.0, 240.0)
        branching_factor = rng.randint(2, 4)

    return {
        "last_k": 1,
        "last_acceptance_rate": 0.55,
        "ema_acceptance_rate": 0.55,
        "last_rejection_position": -1,
        "prompt_length": rng.randint(32, 384),
        "generated_length": 0,
        "queue_length": queue_length,
        "estimated_batch_size": batch_size,
        "recent_arrival_rate": arrival_rate,
        "system_load": system_load,
        "sla_slack_ms": sla_slack_ms,
        "tree_branching_factor": branching_factor,
        "workload_style": workload_style,
    }


def _bucket_acceptance(value: float) -> int:
    if value < 0.35:
        return 0
    if value < 0.65:
        return 1
    return 2


def _bucket_rejection_position(value: int) -> int:
    if value == -1:
        return 3
    if value <= 1:
        return 0
    if value <= 3:
        return 1
    return 2


def _bucket_prompt_length(value: int) -> int:
    if value < 96:
        return 0
    if value < 256:
        return 1
    return 2


def _bucket_generated_length(value: int) -> int:
    if value < 16:
        return 0
    if value < 64:
        return 1
    return 2


def _bucket_queue(value: int) -> int:
    if value <= 2:
        return 0
    if value <= 6:
        return 1
    return 2


def _bucket_batch_size(value: int) -> int:
    if value <= 1:
        return 0
    if value <= 4:
        return 1
    return 2


def _bucket_arrival_rate(value: float) -> int:
    if value < 0.40:
        return 0
    if value < 0.85:
        return 1
    return 2


def _bucket_load(value: float) -> int:
    if value < 0.35:
        return 0
    if value < 0.70:
        return 1
    return 2


def _bucket_sla_slack(value: float) -> int:
    if value < 40.0:
        return 0
    if value < 120.0:
        return 1
    return 2


def _bucket_branching_factor(value: int) -> int:
    if value <= 2:
        return 0
    if value <= 4:
        return 1
    return 2


def featurize_state(state: Mapping[str, Any]) -> tuple[int, ...]:
    """Convert the raw state into a discrete key for tabular RL.

    The output is intentionally compact and categorical. It already includes
    batching-related signals even though Sprint 1 only controls tree depth `k`.
    """

    return (
        int(state["last_k"]),
        _bucket_acceptance(float(state["last_acceptance_rate"])),
        _bucket_acceptance(float(state["ema_acceptance_rate"])),
        _bucket_rejection_position(int(state["last_rejection_position"])),
        _bucket_prompt_length(int(state["prompt_length"])),
        _bucket_generated_length(int(state["generated_length"])),
        _bucket_queue(int(state["queue_length"])),
        _bucket_batch_size(int(state["estimated_batch_size"])),
        _bucket_arrival_rate(float(state["recent_arrival_rate"])),
        _bucket_load(float(state["system_load"])),
        _bucket_sla_slack(float(state["sla_slack_ms"])),
        _bucket_branching_factor(int(state["tree_branching_factor"])),
    )
