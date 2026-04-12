"""Simulator for tree-based speculative decoding in Sprint 1."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from state.features import clamp, initialize_state, update_ema


@dataclass(frozen=True)
class WorkloadProfile:
    """Simple workload profile used to shape arrivals and load."""

    name: str
    arrival_min: float
    arrival_max: float
    base_load_min: float
    base_load_max: float
    burst_probability: float
    burst_size_min: float
    burst_size_max: float
    branching_min: int
    branching_max: int


PROFILES = {
    "steady_low_load": WorkloadProfile(
        name="steady_low_load",
        arrival_min=0.15,
        arrival_max=0.45,
        base_load_min=0.15,
        base_load_max=0.35,
        burst_probability=0.05,
        burst_size_min=0.10,
        burst_size_max=0.35,
        branching_min=2,
        branching_max=4,
    ),
    "bursty_high_load": WorkloadProfile(
        name="bursty_high_load",
        arrival_min=0.65,
        arrival_max=1.10,
        base_load_min=0.45,
        base_load_max=0.80,
        burst_probability=0.35,
        burst_size_min=0.25,
        burst_size_max=0.80,
        branching_min=2,
        branching_max=5,
    ),
}


@dataclass(frozen=True)
class AcceptedPath:
    """Accepted path statistics on the best verified path."""

    accepted_tokens: int
    rejection_position: int
    mean_success_probability: float


class TreeSpeculativeDecodingEnv:
    """Simulate one tree-based speculative decoding round per step.

    The environment is intentionally lightweight. It is not a language-model
    simulator; it is a control simulator that captures the efficiency/risk
    tradeoff of choosing a tree depth `k` under changing workload conditions.
    """

    def __init__(
        self,
        workload_style: str = "steady_low_load",
        episode_length: int = 32,
        seed: int | None = None,
    ) -> None:
        if workload_style not in PROFILES:
            raise ValueError(f"unsupported workload_style={workload_style!r}")
        self.workload_style = workload_style
        self.episode_length = episode_length
        self._profile = PROFILES[workload_style]
        self._rng = random.Random(seed)
        self._step_count = 0

    def reset(self) -> dict[str, Any]:
        """Reset one episode and return the initial raw state."""
        self._step_count = 0
        return initialize_state(self.workload_style, rng=self._rng)

    def sample_accepted_tokens(self, k: int, state: dict[str, Any]) -> AcceptedPath:
        """Sample accepted depth on the best verified path."""
        load = float(state["system_load"])
        ema_acceptance = float(state["ema_acceptance_rate"])
        queue_pressure = clamp(int(state["queue_length"]) / 10.0, 0.0, 1.0)
        branching = int(state["tree_branching_factor"])
        prompt_penalty = clamp((int(state["prompt_length"]) - 64) / 448.0, 0.0, 1.0)
        progress_penalty = clamp(int(state["generated_length"]) / 128.0, 0.0, 1.0)

        # Wider trees provide some hedging benefit, but they also become costlier.
        branch_bonus = 0.03 * max(0, branching - 1)
        accepted = 0
        probabilities: list[float] = []

        for depth in range(1, k + 1):
            success_probability = (
                0.76
                + 0.22 * (ema_acceptance - 0.50)
                + branch_bonus
                - 0.17 * load
                - 0.12 * queue_pressure
                - 0.08 * prompt_penalty
                - 0.05 * progress_penalty
                - 0.07 * (depth - 1)
                + self._rng.uniform(-0.04, 0.04)
            )
            success_probability = clamp(success_probability, 0.05, 0.97)
            probabilities.append(success_probability)

            if self._rng.random() <= success_probability:
                accepted += 1
            else:
                return AcceptedPath(
                    accepted_tokens=accepted,
                    rejection_position=depth,
                    mean_success_probability=sum(probabilities) / len(probabilities),
                )

        return AcceptedPath(
            accepted_tokens=accepted,
            rejection_position=-1,
            mean_success_probability=sum(probabilities) / len(probabilities),
        )

    def step(self, state: dict[str, Any], k: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Simulate one speculative tree decision."""
        if k <= 0:
            raise ValueError("k must be positive")

        self._step_count += 1
        accepted = self.sample_accepted_tokens(k, state)
        branching = int(state["tree_branching_factor"])

        tree_nodes = sum(branching**depth for depth in range(1, k + 1))
        normalized_tree_cost = math.log2(tree_nodes + 1.0)
        rejected_depth = max(k - accepted.accepted_tokens, 0)
        emitted_tokens = max(1, accepted.accepted_tokens)

        incoming_rate = self._next_arrival_rate(float(state["recent_arrival_rate"]))
        incoming_requests = max(0, int(round(incoming_rate * 3.0 + self._rng.uniform(-0.6, 1.4))))

        service_boost = 1 + accepted.accepted_tokens
        queue_length = max(0, int(state["queue_length"]) + incoming_requests - service_boost)

        next_load = clamp(
            0.35 * float(state["system_load"])
            + 0.35 * clamp(queue_length / 10.0, 0.0, 1.0)
            + 0.20 * incoming_rate
            + 0.06 * normalized_tree_cost
            + self._rng.uniform(-0.04, 0.04),
            0.0,
            1.0,
        )

        batch_size = int(clamp(round(1 + queue_length * 0.6 + incoming_rate * 1.6), 1, 8))
        next_branching = self._rng.randint(
            self._profile.branching_min,
            self._profile.branching_max,
        )

        next_sla_slack = clamp(
            float(state["sla_slack_ms"])
            + 12.0
            - 26.0 * next_load
            - 3.5 * queue_length
            + 4.0 * accepted.accepted_tokens,
            0.0,
            250.0,
        )

        acceptance_rate = accepted.accepted_tokens / float(k)
        next_state = dict(state)
        next_state.update(
            {
                "last_k": k,
                "last_acceptance_rate": acceptance_rate,
                "ema_acceptance_rate": update_ema(
                    float(state["ema_acceptance_rate"]),
                    acceptance_rate,
                ),
                "last_rejection_position": accepted.rejection_position,
                "generated_length": int(state["generated_length"]) + emitted_tokens,
                "queue_length": queue_length,
                "estimated_batch_size": batch_size,
                "recent_arrival_rate": incoming_rate,
                "system_load": next_load,
                "sla_slack_ms": next_sla_slack,
                "tree_branching_factor": next_branching,
            }
        )

        load_penalty = 1.35 * next_load + 0.08 * queue_length
        sla_risk = max(0.0, 60.0 - next_sla_slack) / 60.0
        reward = (
            2.00 * accepted.accepted_tokens
            - 1.10 * rejected_depth
            - 0.35 * normalized_tree_cost
            - 0.80 * load_penalty
            - 0.90 * sla_risk
        )

        info = {
            "accepted_tokens": accepted.accepted_tokens,
            "rejected_depth": rejected_depth,
            "rejection_position": accepted.rejection_position,
            "tree_nodes": tree_nodes,
            "normalized_tree_cost": normalized_tree_cost,
            "load_penalty": load_penalty,
            "sla_risk": sla_risk,
            "mean_success_probability": accepted.mean_success_probability,
            "workload_style": self.workload_style,
        }
        done = self._step_count >= self.episode_length
        return next_state, reward, done, info

    def _next_arrival_rate(self, previous_rate: float) -> float:
        """Evolve arrival rate according to the chosen workload style."""
        target = self._rng.uniform(self._profile.arrival_min, self._profile.arrival_max)
        burst = 0.0
        if self._rng.random() < self._profile.burst_probability:
            burst = self._rng.uniform(
                self._profile.burst_size_min,
                self._profile.burst_size_max,
            )

        return clamp(
            0.55 * previous_rate + 0.45 * target + burst,
            0.0,
            1.6,
        )
