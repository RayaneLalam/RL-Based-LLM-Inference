"""LLM-backed environment using real speculative decoding interactions."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from decoding.speculative import HuggingFaceTreeSpeculativeDecoder
from state.features import clamp, initialize_state, update_ema
from prompt_dataset import PromptDataset


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


class LLMTreeSpeculativeDecodingEnv:
    """Environment that uses real speculative decoding to drive transitions."""

    def __init__(
        self,
        decoder: HuggingFaceTreeSpeculativeDecoder,
        prompt_dataset: PromptDataset,
        workload_style: str = "steady_low_load",
        episode_length: int = 32,
        seed: int | None = None,
        sampling_overrides: dict[str, Any] | None = None,
    ) -> None:
        if workload_style not in PROFILES:
            raise ValueError(f"unsupported workload_style={workload_style!r}")
        self.decoder = decoder
        self.prompt_dataset = prompt_dataset
        self.workload_style = workload_style
        self.episode_length = episode_length
        self._profile = PROFILES[workload_style]
        self._rng = random.Random(seed)
        self._step_count = 0
        self._input_ids = None
        self._prompt_id = None
        self._prompt_text = None
        self._prompt_length = 0
        self._sampling_overrides = sampling_overrides or {}

    def reset(self) -> dict[str, Any]:
        """Reset one episode with a real prompt."""
        self._step_count = 0
        prompt_record = self.prompt_dataset.sample_prompt()
        self._prompt_id = prompt_record.prompt_id
        self._prompt_text = prompt_record.prompt
        self._input_ids = self.decoder.encode_prompt(prompt_record.prompt)
        
        # Reserve 128 tokens for generation and speculative tree nodes
        max_len = int(getattr(self.decoder.tokenizer, "model_max_length", 1024)) - 128
        if self._input_ids.shape[-1] > max_len:
            # Keep the most recent tokens to stay within the model context window.
            self._input_ids = self._input_ids[:, -max_len:]
        self._prompt_length = int(self._input_ids.shape[1])

        if not hasattr(self, "_episode_count"):
            self._episode_count = 0
        self._episode_count += 1
        print(f"[Env] Starting episode {self._episode_count} with prompt length {self._prompt_length}")

        state = initialize_state(self.workload_style, rng=self._rng)
        state["prompt_length"] = self._prompt_length
        state["generated_length"] = 0
        state.update(self._sampling_overrides)
        return state

    def step(self, state: dict[str, Any], k: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Run one speculative decoding step and update the environment state."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self._input_ids is None:
            raise RuntimeError("reset() must be called before step()")

        self._step_count += 1
        new_input_ids, _accepted, extra = self.decoder.speculative_decode_step(
            self._input_ids,
            k,
            state=state,
        )
        self._input_ids = new_input_ids

        accepted_tokens = int(extra["accepted_tokens"])
        emitted_tokens = int(extra["emitted_tokens"])
        acceptance_rate = float(extra["acceptance_rate"])
        rejection_position = int(extra["rejection_position"])
        normalized_tree_cost = float(extra["normalized_tree_cost"])

        incoming_rate = self._next_arrival_rate(float(state["recent_arrival_rate"]))
        incoming_requests = max(0, int(round(incoming_rate * 3.0 + self._rng.uniform(-0.6, 1.4))))
        service_boost = emitted_tokens
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
        next_branching = self._rng.randint(self._profile.branching_min, self._profile.branching_max)

        next_sla_slack = clamp(
            float(state["sla_slack_ms"])
            + 12.0
            - 26.0 * next_load
            - 3.5 * queue_length
            + 4.0 * accepted_tokens,
            0.0,
            250.0,
        )

        rejected_depth = max(k - accepted_tokens, 0)
        load_penalty = 1.35 * next_load + 0.08 * queue_length
        sla_risk = max(0.0, 60.0 - next_sla_slack) / 60.0
        reward = (
            2.00 * accepted_tokens
            - 1.10 * rejected_depth
            - 0.35 * normalized_tree_cost
            - 0.80 * load_penalty
            - 0.90 * sla_risk
        )

        next_state = dict(state)
        next_state.update(
            {
                "last_k": k,
                "last_acceptance_rate": acceptance_rate,
                "ema_acceptance_rate": update_ema(float(state["ema_acceptance_rate"]), acceptance_rate),
                "last_rejection_position": rejection_position,
                "generated_length": int(state["generated_length"]) + emitted_tokens,
                "queue_length": queue_length,
                "estimated_batch_size": batch_size,
                "recent_arrival_rate": incoming_rate,
                "system_load": next_load,
                "sla_slack_ms": next_sla_slack,
                "tree_branching_factor": next_branching,
                "prompt_length": self._prompt_length,
                "workload_style": self.workload_style,
            }
        )

        info = {
            "prompt_id": self._prompt_id,
            "prompt_length": self._prompt_length,
            "accepted_tokens": accepted_tokens,
            "emitted_tokens": emitted_tokens,
            "acceptance_rate": acceptance_rate,
            "rejection_position": rejection_position,
            "tree_nodes": int(extra["tree_nodes"]),
            "normalized_tree_cost": normalized_tree_cost,
            "target_logprob_mean": float(extra["target_logprob_mean"]),
            "draft_forward_passes": int(extra["draft_forward_passes"]),
            "target_forward_passes": int(extra["target_forward_passes"]),
            "draft_time_s": float(extra["draft_time_s"]),
            "verify_time_s": float(extra["verify_time_s"]),
            "step_time_s": float(extra["step_time_s"]),
            "simulated_queue_length": queue_length,
            "simulated_system_load": next_load,
            "simulated_arrival_rate": incoming_rate,
        }

        print(f"      Episode {self._episode_count} | Step {self._step_count:02d}/{self.episode_length} | "
              f"K: {k:02d} | Accepted: {accepted_tokens:02d} | Cost: {normalized_tree_cost:.3f}")

        done = self._step_count >= self.episode_length
        return next_state, reward, done, info

    def _next_arrival_rate(self, previous_rate: float) -> float:
        """Evolve arrival rate according to the chosen workload style."""
        target = self._rng.uniform(self._profile.arrival_min, self._profile.arrival_max)
        burst = 0.0
        if self._rng.random() < self._profile.burst_probability:
            burst = self._rng.uniform(self._profile.burst_size_min, self._profile.burst_size_max)

        return clamp(
            0.55 * previous_rate + 0.45 * target + burst,
            0.0,
            1.6,
        )
