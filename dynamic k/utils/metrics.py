"""Shared reward and rollout helpers for Sprint 1."""

from __future__ import annotations

from typing import Any, Iterable

from state.features import clamp, update_ema


def compute_reward(
    next_state: dict[str, Any],
    k: int,
    accepted_tokens: int,
    extra_info: dict[str, Any],
) -> float:
    """Compute the Sprint 1 reward from accepted work and system stress."""
    rejected_depth = max(k - accepted_tokens, 0)
    tree_cost = float(extra_info.get("normalized_tree_cost", 0.0))
    load_penalty = 1.35 * float(next_state["system_load"]) + 0.08 * int(next_state["queue_length"])
    sla_risk = max(0.0, 60.0 - float(next_state["sla_slack_ms"])) / 60.0

    return (
        2.00 * accepted_tokens
        - 1.10 * rejected_depth
        - 0.35 * tree_cost
        - 0.80 * load_penalty
        - 0.90 * sla_risk
    )


def compute_real_latency_reward(
    next_state: dict[str, Any],
    k: int,
    accepted_tokens: int,
    extra_info: dict[str, Any],
) -> float:
    """Reward real decoding rounds using measured throughput and runtime cost."""
    emitted_tokens = float(
        extra_info.get(
            "emitted_tokens",
            accepted_tokens + int(extra_info.get("used_target_fallback", False)),
        )
    )
    step_time_s = max(float(extra_info.get("step_time_s", 0.0)), 1e-6)
    throughput = emitted_tokens / step_time_s
    draft_forward_passes = float(extra_info.get("draft_forward_passes", 0.0))
    target_forward_passes = float(extra_info.get("target_forward_passes", 1.0))
    tree_cost = float(extra_info.get("normalized_tree_cost", 0.0))
    load_penalty = 1.35 * float(next_state["system_load"]) + 0.08 * int(next_state["queue_length"])
    sla_risk = max(0.0, 60.0 - float(next_state["sla_slack_ms"])) / 60.0

    return (
        1.50 * throughput
        - 0.12 * draft_forward_passes
        - 0.80 * target_forward_passes
        - 0.25 * tree_cost
        - 0.80 * load_penalty
        - 0.90 * sla_risk
    )


def update_state_from_feedback(
    state: dict[str, Any],
    k: int,
    accepted_tokens: int,
    extra_info: dict[str, Any],
) -> dict[str, Any]:
    """Update the raw state after a real decoding round."""
    next_state = dict(state)
    generated_increment = int(
        extra_info.get(
            "emitted_tokens",
            accepted_tokens + int(extra_info.get("used_target_fallback", False)),
        )
    )
    acceptance_rate = accepted_tokens / float(k)
    step_latency_ms = 1000.0 * float(extra_info.get("step_time_s", 0.0))
    workload_style = str(state.get("workload_style", "steady_low_load"))
    branching_factor = int(extra_info.get("tree_branching_factor", state.get("tree_branching_factor", 2)))

    if workload_style == "bursty_high_load":
        arrival_push = 0.90
        queue_scale = 4.0
    else:
        arrival_push = 0.35
        queue_scale = 2.5

    next_arrival_rate = clamp(
        0.65 * float(state["recent_arrival_rate"])
        + 0.25 * arrival_push
        + 0.10 * min(step_latency_ms / 180.0, 1.0),
        0.0,
        1.6,
    )
    incoming_requests = max(0, int(round(next_arrival_rate * queue_scale)))
    service_gain = generated_increment
    queue_length = max(0, int(state["queue_length"]) + incoming_requests - service_gain)

    next_load = clamp(
        0.55 * float(state["system_load"])
        + 0.25 * min(step_latency_ms / 220.0, 1.0)
        + 0.20 * min(queue_length / 10.0, 1.0),
        0.0,
        1.0,
    )
    sla_slack_ms = clamp(
        float(state["sla_slack_ms"]) + 18.0 - step_latency_ms,
        0.0,
        250.0,
    )
    batch_size = int(clamp(round(1 + 0.6 * queue_length + 1.5 * next_arrival_rate), 1, 8))

    next_state.update(
        {
            "last_k": k,
            "last_acceptance_rate": acceptance_rate,
            "ema_acceptance_rate": update_ema(float(state["ema_acceptance_rate"]), acceptance_rate),
            "last_rejection_position": int(extra_info.get("rejection_position", -1)),
            "generated_length": int(state["generated_length"]) + generated_increment,
            "queue_length": queue_length,
            "estimated_batch_size": batch_size,
            "recent_arrival_rate": next_arrival_rate,
            "system_load": next_load,
            "sla_slack_ms": sla_slack_ms,
            "tree_branching_factor": branching_factor,
        }
    )
    return next_state


def rollout_summary(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Summarize a decoding rollout."""
    rows = list(records)
    if not rows:
        return {
            "steps": 0.0,
            "avg_reward": 0.0,
            "avg_k": 0.0,
            "avg_acceptance_rate": 0.0,
            "avg_step_time_s": 0.0,
        }

    steps = float(len(rows))
    return {
        "steps": steps,
        "avg_reward": sum(row["reward"] for row in rows) / steps,
        "avg_k": sum(row["k"] for row in rows) / steps,
        "avg_acceptance_rate": sum(row["acceptance_rate"] for row in rows) / steps,
        "avg_step_time_s": sum(row["step_time_s"] for row in rows) / steps,
    }
