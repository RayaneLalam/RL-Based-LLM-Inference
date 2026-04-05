"""Single-file Sprint 1 implementation for dynamic tree-depth control.

This file contains:
- virtualenv bootstrap logic
- state design and featurization
- tree-aware simulator
- tabular Q-learning controller
- real Hugging Face tree-based speculative decoding
- batch benchmarks with plots

Typical usage:

1. Create the virtual environment and install dependencies:
   python3.12 "speculative devoding dynamic k.py" bootstrap-venv

2. Activate the environment:
   . .venv/bin/activate

3. Train the simulator controller:
   python "speculative devoding dynamic k.py" train --episodes 250

4. Run real decoding:
   python "speculative devoding dynamic k.py" decode --prompt "Adaptive inference control"

5. Run benchmarks:
   python "speculative devoding dynamic k.py" benchmark

6. Show command help:
   python "speculative devoding dynamic k.py" --help
   python "speculative devoding dynamic k.py" train --help
   python "speculative devoding dynamic k.py" decode --help
   python "speculative devoding dynamic k.py" benchmark --help
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
import venv
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


K_VALUES = (1, 2, 3, 4, 5)
EMA_ALPHA = 0.35
DEFAULT_PROMPTS = (
    "Adaptive inference control",
    "ParetoServe chooses speculative depth",
    "Queue pressure rises during a traffic burst",
)
REQUIREMENTS = (
    "torch",
    "transformers",
    "accelerate",
    "sentencepiece",
    "safetensors",
    "matplotlib",
    "pandas",
    "nbformat",
)


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a float into a closed interval."""
    return max(lower, min(upper, value))


def update_ema(previous: float, current: float, alpha: float = EMA_ALPHA) -> float:
    """Update the exponential moving average of acceptance history."""
    return alpha * current + (1.0 - alpha) * previous


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory when needed."""
    parent = Path(path).parent
    if str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)


def bootstrap_venv(venv_dir: str = ".venv", python_cmd: str = "python3.12") -> None:
    """Create a local virtualenv and install all required packages."""
    target = Path(venv_dir)
    if not target.exists():
        subprocess.run([python_cmd, "-m", "venv", str(target)], check=True)

    if sys.platform == "win32":
        python_bin = target / "Scripts" / "python"
    else:
        python_bin = target / "bin" / "python"

    subprocess.run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python_bin), "-m", "pip", "install", *REQUIREMENTS], check=True)

    print(f"virtualenv ready at {target}")
    print(f"activate with: . {target}/bin/activate")


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


def featurize_state(state: dict[str, Any]) -> tuple[int, ...]:
    """Convert the raw state into a discrete tabular key."""
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
    """Simulator for one tree-based speculative decoding round per step."""

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
        next_branching = self._rng.randint(self._profile.branching_min, self._profile.branching_max)
        next_sla_slack = clamp(
            float(state["sla_slack_ms"]) + 12.0 - 26.0 * next_load - 3.5 * queue_length + 4.0 * accepted.accepted_tokens,
            0.0,
            250.0,
        )

        acceptance_rate = accepted.accepted_tokens / float(k)
        next_state = dict(state)
        next_state.update(
            {
                "last_k": k,
                "last_acceptance_rate": acceptance_rate,
                "ema_acceptance_rate": update_ema(float(state["ema_acceptance_rate"]), acceptance_rate),
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
        reward = 2.00 * accepted.accepted_tokens - 1.10 * rejected_depth - 0.35 * normalized_tree_cost - 0.80 * load_penalty - 0.90 * sla_risk

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
        target = self._rng.uniform(self._profile.arrival_min, self._profile.arrival_max)
        burst = 0.0
        if self._rng.random() < self._profile.burst_probability:
            burst = self._rng.uniform(self._profile.burst_size_min, self._profile.burst_size_max)
        return clamp(0.55 * previous_rate + 0.45 * target + burst, 0.0, 1.6)


@dataclass(frozen=True)
class EpisodeSummary:
    """Training summary for one episode."""

    episode: int
    total_reward: float
    average_reward: float
    average_k: float
    epsilon: float
    steps: int


class QLearningController:
    """Simple tabular controller over the discrete action space of `k` values."""

    def __init__(
        self,
        actions: Iterable[int] = K_VALUES,
        learning_rate: float = 0.18,
        discount: float = 0.92,
        epsilon: float = 0.25,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        self.actions = tuple(actions)
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table: dict[tuple[int, ...], dict[int, float]] = {}
        self._rng = random.Random(seed)

    def choose_k(self, state: dict[str, Any], greedy: bool = False) -> int:
        """Choose a tree depth with epsilon-greedy exploration."""
        feature_state = featurize_state(state)
        self._ensure_state(feature_state)
        if not greedy and self._rng.random() < self.epsilon:
            return self._rng.choice(self.actions)

        values = self.q_table[feature_state]
        max_value = max(values.values())
        best_actions = [action for action, value in values.items() if value == max_value]
        return self._rng.choice(best_actions)

    def update_policy(self, state: dict[str, Any], action: int, reward: float, next_state: dict[str, Any]) -> None:
        """Apply the tabular Q-learning update."""
        state_key = featurize_state(state)
        next_state_key = featurize_state(next_state)
        self._ensure_state(state_key)
        self._ensure_state(next_state_key)
        current_q = self.q_table[state_key][action]
        next_q = max(self.q_table[next_state_key].values())
        td_target = reward + self.discount * next_q
        td_error = td_target - current_q
        self.q_table[state_key][action] = current_q + self.learning_rate * td_error

    def train(self, env: Any, episodes: int = 250) -> list[EpisodeSummary]:
        """Train on the simulator for a fixed number of episodes."""
        history: list[EpisodeSummary] = []
        for episode in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0.0
            actions: list[int] = []
            steps = 0
            while True:
                action = self.choose_k(state)
                next_state, reward, done, _ = env.step(state, action)
                self.update_policy(state, action, reward, next_state)
                total_reward += reward
                actions.append(action)
                steps += 1
                state = next_state
                if done:
                    break

            history.append(
                EpisodeSummary(
                    episode=episode,
                    total_reward=total_reward,
                    average_reward=total_reward / max(steps, 1),
                    average_k=sum(actions) / max(len(actions), 1),
                    epsilon=self.epsilon,
                    steps=steps,
                )
            )
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return history

    def evaluate(self, env: Any, episodes: int = 20) -> dict[str, float]:
        """Run a greedy evaluation without exploration."""
        total_reward = 0.0
        total_actions = 0
        total_acceptance = 0.0
        total_steps = 0

        for _ in range(episodes):
            state = env.reset()
            while True:
                action = self.choose_k(state, greedy=True)
                next_state, reward, done, _ = env.step(state, action)
                total_reward += reward
                total_actions += action
                total_acceptance += next_state["last_acceptance_rate"]
                total_steps += 1
                state = next_state
                if done:
                    break

        return {
            "episodes": float(episodes),
            "avg_reward": total_reward / max(total_steps, 1),
            "avg_k": total_actions / max(total_steps, 1),
            "avg_acceptance_rate": total_acceptance / max(total_steps, 1),
        }

    def save(self, path: str) -> None:
        """Persist the Q-table as JSON."""
        payload = {
            "actions": list(self.actions),
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "epsilon": self.epsilon,
            "min_epsilon": self.min_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "q_table": {
                "|".join(str(part) for part in state_key): values
                for state_key, values in self.q_table.items()
            },
        }
        ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "QLearningController":
        """Restore a controller from a saved Q-table."""
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        controller = cls(
            actions=payload["actions"],
            learning_rate=payload["learning_rate"],
            discount=payload["discount"],
            epsilon=payload["epsilon"],
            min_epsilon=payload["min_epsilon"],
            epsilon_decay=payload["epsilon_decay"],
        )
        controller.q_table = {
            tuple(int(part) for part in key.split("|")): {
                int(action): float(value) for action, value in values.items()
            }
            for key, values in payload["q_table"].items()
        }
        return controller

    def _ensure_state(self, state_key: tuple[int, ...]) -> None:
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.actions}


def compute_reward(next_state: dict[str, Any], k: int, accepted_tokens: int, extra_info: dict[str, Any]) -> float:
    """Compute the Sprint 1 reward from accepted work and system stress."""
    rejected_depth = max(k - accepted_tokens, 0)
    tree_cost = float(extra_info.get("normalized_tree_cost", 0.0))
    load_penalty = 1.35 * float(next_state["system_load"]) + 0.08 * int(next_state["queue_length"])
    sla_risk = max(0.0, 60.0 - float(next_state["sla_slack_ms"])) / 60.0
    return 2.00 * accepted_tokens - 1.10 * rejected_depth - 0.35 * tree_cost - 0.80 * load_penalty - 0.90 * sla_risk


def update_state_from_feedback(state: dict[str, Any], k: int, accepted_tokens: int, extra_info: dict[str, Any]) -> dict[str, Any]:
    """Update the raw state after a real decoding round."""
    next_state = dict(state)
    generated_increment = accepted_tokens + int(extra_info.get("used_target_fallback", False))
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
        0.65 * float(state["recent_arrival_rate"]) + 0.25 * arrival_push + 0.10 * min(step_latency_ms / 180.0, 1.0),
        0.0,
        1.6,
    )
    incoming_requests = max(0, int(round(next_arrival_rate * queue_scale)))
    service_gain = 1 + accepted_tokens
    queue_length = max(0, int(state["queue_length"]) + incoming_requests - service_gain)
    next_load = clamp(
        0.55 * float(state["system_load"]) + 0.25 * min(step_latency_ms / 220.0, 1.0) + 0.20 * min(queue_length / 10.0, 1.0),
        0.0,
        1.0,
    )
    sla_slack_ms = clamp(float(state["sla_slack_ms"]) + 18.0 - step_latency_ms, 0.0, 250.0)
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


@dataclass
class TreeNode:
    """Node inside the speculative draft tree."""

    token_id: int | None
    depth: int
    path_token_ids: tuple[int, ...] = ()
    draft_logprob: float = 0.0
    children: list["TreeNode"] = field(default_factory=list)


class HuggingFaceTreeSpeculativeDecoder:
    """Tree-based speculative decoder with real Hugging Face models."""

    def __init__(self, draft_model_name: str, target_model_name: str) -> None:
        import torch
        from accelerate import Accelerator
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.accelerator = Accelerator()
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.draft_model_name = draft_model_name
        self.target_model_name = target_model_name
        self.draft_model = self.accelerator.prepare_model(
            AutoModelForCausalLM.from_pretrained(draft_model_name),
            evaluation_mode=True,
        )
        self.target_model = self.accelerator.prepare_model(
            AutoModelForCausalLM.from_pretrained(target_model_name),
            evaluation_mode=True,
        )
        self.draft_model.eval()
        self.target_model.eval()

        self.device = self.accelerator.device
        self.target_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.draft_model.config.pad_token_id = self.tokenizer.pad_token_id

    def encode_prompt(self, prompt: str):
        """Tokenize a text prompt for inference."""
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return encoded["input_ids"].to(self.device)

    def decode_tokens(self, input_ids) -> str:
        """Convert model tokens back to text."""
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def speculative_decode_step(
        self,
        input_ids,
        k: int,
        state: dict[str, Any] | None = None,
        branching_factor: int | None = None,
    ):
        """Run one tree-based speculative decoding round."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape [1, seq_len]")
        if k <= 0:
            raise ValueError("k must be positive")

        branch = branching_factor
        if branch is None and state is not None:
            branch = int(state.get("tree_branching_factor", 2))
        branch = max(1, int(branch or 2))

        draft_start = time.perf_counter()
        root, tree_node_count = self._build_tree(input_ids, k=k, branching_factor=branch)
        draft_time_s = time.perf_counter() - draft_start

        verify_start = time.perf_counter()
        verified = self._verify_tree(input_ids, root, k=k, branching_factor=branch)
        verify_time_s = time.perf_counter() - verify_start

        step_time_s = draft_time_s + verify_time_s
        normalized_tree_cost = math.log2(tree_node_count + 1.0)
        acceptance_rate = verified["accepted_tokens"] / float(k)
        extra_info = {
            "draft_model_name": self.draft_model_name,
            "target_model_name": self.target_model_name,
            "accepted_token_ids": verified["accepted_token_ids"],
            "accepted_tokens": verified["accepted_tokens"],
            "acceptance_rate": acceptance_rate,
            "rejection_position": verified["rejection_position"],
            "used_target_fallback": verified["used_target_fallback"],
            "fallback_token_id": verified["fallback_token_id"],
            "tree_branching_factor": branch,
            "tree_nodes": tree_node_count,
            "normalized_tree_cost": normalized_tree_cost,
            "draft_time_s": draft_time_s,
            "verify_time_s": verify_time_s,
            "step_time_s": step_time_s,
            "target_logprob_mean": verified["target_logprob_mean"],
        }
        return verified["new_sequence"], verified["accepted_tokens"], extra_info

    def _build_tree(self, input_ids, k: int, branching_factor: int) -> tuple[TreeNode, int]:
        root = TreeNode(token_id=None, depth=0, path_token_ids=())
        frontier = [root]
        total_nodes = 0

        for depth in range(1, k + 1):
            next_frontier: list[TreeNode] = []
            for node in frontier:
                draft_input = self._append_path(input_ids, node.path_token_ids)
                log_probs = self._next_token_log_probs(self.draft_model, draft_input)
                top_log_probs, top_token_ids = self.torch.topk(log_probs, k=branching_factor, dim=-1)

                for branch_idx in range(branching_factor):
                    token_id = int(top_token_ids[0, branch_idx].item())
                    child = TreeNode(
                        token_id=token_id,
                        depth=depth,
                        path_token_ids=node.path_token_ids + (token_id,),
                        draft_logprob=float(top_log_probs[0, branch_idx].item()),
                    )
                    node.children.append(child)
                    next_frontier.append(child)
                    total_nodes += 1
            frontier = next_frontier

        return root, total_nodes

    def _verify_tree(self, input_ids, root: TreeNode, k: int, branching_factor: int) -> dict[str, Any]:
        current_node = root
        current_sequence = input_ids.clone()
        accepted_token_ids: list[int] = []
        target_logprobs: list[float] = []
        rejection_position = -1
        fallback_token_id: int | None = None
        used_target_fallback = False

        for depth in range(1, k + 1):
            if not current_node.children:
                rejection_position = depth
                break

            log_probs = self._next_token_log_probs(self.target_model, current_sequence)
            target_token_id = int(self.torch.argmax(log_probs, dim=-1).item())
            target_logprobs.append(float(log_probs[0, target_token_id].item()))

            candidates = {child.token_id: child for child in current_node.children}
            if target_token_id not in candidates:
                rejection_position = depth
                fallback_token_id = target_token_id
                current_sequence = self._append_tokens(current_sequence, (target_token_id,))
                used_target_fallback = True
                break

            accepted_token_ids.append(target_token_id)
            current_sequence = self._append_tokens(current_sequence, (target_token_id,))
            current_node = candidates[target_token_id]

        return {
            "new_sequence": current_sequence,
            "accepted_tokens": len(accepted_token_ids),
            "accepted_token_ids": accepted_token_ids,
            "rejection_position": rejection_position,
            "fallback_token_id": fallback_token_id,
            "used_target_fallback": used_target_fallback,
            "target_logprob_mean": sum(target_logprobs) / len(target_logprobs) if target_logprobs else float("nan"),
            "tree_branching_factor": branching_factor,
        }

    def _next_token_log_probs(self, model, input_ids):
        with self.torch.inference_mode():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            return self.torch.log_softmax(logits, dim=-1)

    def _append_path(self, input_ids, path_token_ids: tuple[int, ...]):
        if not path_token_ids:
            return input_ids
        extension = self.torch.tensor([list(path_token_ids)], device=self.device, dtype=input_ids.dtype)
        return self.torch.cat([input_ids, extension], dim=1)

    def _append_tokens(self, input_ids, token_ids: tuple[int, ...]):
        if not token_ids:
            return input_ids
        extension = self.torch.tensor([list(token_ids)], device=self.device, dtype=input_ids.dtype)
        return self.torch.cat([input_ids, extension], dim=1)


@dataclass(frozen=True)
class PolicySpec:
    """Policy used in the benchmark runner."""

    name: str
    mode: str
    fixed_k: int | None = None


POLICIES = (
    PolicySpec(name="fixed_k_1", mode="fixed", fixed_k=1),
    PolicySpec(name="fixed_k_3", mode="fixed", fixed_k=3),
    PolicySpec(name="fixed_k_5", mode="fixed", fixed_k=5),
    PolicySpec(name="adaptive_rl", mode="adaptive", fixed_k=None),
)


def ensure_controller(path: str, workload: str, episodes: int, episode_length: int, seed: int) -> QLearningController:
    """Load or train the controller checkpoint."""
    if os.path.exists(path):
        return QLearningController.load(path)

    env = TreeSpeculativeDecodingEnv(workload_style=workload, episode_length=episode_length, seed=seed)
    controller = QLearningController(seed=seed)
    controller.train(env, episodes=episodes)
    controller.save(path)
    return controller


def run_train(args: argparse.Namespace) -> None:
    """Train the simulator controller and save the checkpoint."""
    env = TreeSpeculativeDecodingEnv(
        workload_style=args.workload,
        episode_length=args.episode_length,
        seed=args.seed,
    )
    controller = QLearningController(seed=args.seed)
    history = controller.train(env, episodes=args.episodes)
    evaluation = controller.evaluate(env, episodes=args.eval_episodes)
    controller.save(args.checkpoint)

    print("Training complete")
    print(f"episodes: {args.episodes}")
    print(f"final_epsilon: {controller.epsilon:.4f}")
    print(f"last_episode: {json.dumps(asdict(history[-1]), indent=2)}")
    print("evaluation:")
    print(json.dumps(evaluation, indent=2))
    print(f"checkpoint: {args.checkpoint}")


def run_decode(args: argparse.Namespace) -> None:
    """Run real-model decoding with the adaptive controller."""
    controller = ensure_controller(
        path=args.checkpoint,
        workload=args.workload,
        episodes=args.train_episodes,
        episode_length=args.episode_length,
        seed=args.seed,
    )
    decoder = HuggingFaceTreeSpeculativeDecoder(
        draft_model_name=args.draft_model,
        target_model_name=args.target_model,
    )
    input_ids = decoder.encode_prompt(args.prompt)
    state = initialize_state(workload_style=args.workload)
    state["prompt_length"] = int(input_ids.shape[1])
    state["tree_branching_factor"] = args.branching_factor
    rollout: list[dict[str, float]] = []
    start_length = int(input_ids.shape[1])

    while int(input_ids.shape[1]) - start_length < args.max_new_tokens:
        k = controller.choose_k(state, greedy=True)
        new_input_ids, accepted_tokens, extra_info = decoder.speculative_decode_step(
            input_ids=input_ids,
            k=k,
            state=state,
            branching_factor=args.branching_factor,
        )
        next_state = update_state_from_feedback(state, k, accepted_tokens, extra_info)
        reward = compute_reward(next_state, k, accepted_tokens, extra_info)
        controller.update_policy(state, k, reward, next_state)

        rollout.append(
            {
                "k": float(k),
                "accepted_tokens": float(accepted_tokens),
                "acceptance_rate": float(extra_info["acceptance_rate"]),
                "reward": float(reward),
                "step_time_s": float(extra_info["step_time_s"]),
            }
        )
        input_ids = new_input_ids
        state = next_state
        if int(input_ids.shape[1]) - start_length >= args.max_new_tokens:
            break

    summary = rollout_summary(rollout)
    generated_text = decoder.decode_tokens(input_ids)
    controller.save(args.checkpoint)

    print("Real decoding run complete")
    print(json.dumps(summary, indent=2))
    print("generated_text:")
    print(generated_text)
    print(f"checkpoint: {args.checkpoint}")


def evaluate_fixed_policy_sim(workload: str, fixed_k: int, episodes: int, episode_length: int, seed: int) -> dict[str, float]:
    """Evaluate a fixed-k baseline in the simulator."""
    env = TreeSpeculativeDecodingEnv(workload_style=workload, episode_length=episode_length, seed=seed)
    total_reward = 0.0
    total_acceptance = 0.0
    total_steps = 0
    for _ in range(episodes):
        state = env.reset()
        while True:
            next_state, reward, done, _ = env.step(state, fixed_k)
            total_reward += reward
            total_acceptance += next_state["last_acceptance_rate"]
            total_steps += 1
            state = next_state
            if done:
                break
    return {
        "avg_reward": total_reward / max(total_steps, 1),
        "avg_k": float(fixed_k),
        "avg_acceptance_rate": total_acceptance / max(total_steps, 1),
        "steps": float(total_steps),
    }


def run_simulator_benchmark(args: argparse.Namespace, output_dir: Path):
    """Run simulator baselines and adaptive evaluation."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    workloads = ("steady_low_load", "bursty_high_load")
    for workload in workloads:
        controller = QLearningController(seed=args.seed)
        env = TreeSpeculativeDecodingEnv(workload_style=workload, episode_length=args.episode_length, seed=args.seed)
        controller.train(env, episodes=args.train_episodes)
        adaptive_metrics = controller.evaluate(env, episodes=args.sim_eval_episodes)
        rows.append({"benchmark": "simulator", "workload": workload, "policy": "adaptive_rl", **adaptive_metrics})

        for policy in POLICIES:
            if policy.mode != "fixed":
                continue
            metrics = evaluate_fixed_policy_sim(
                workload=workload,
                fixed_k=int(policy.fixed_k),
                episodes=args.sim_eval_episodes,
                episode_length=args.episode_length,
                seed=args.seed,
            )
            rows.append({"benchmark": "simulator", "workload": workload, "policy": policy.name, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "simulator_results.csv", index=False)
    with open(output_dir / "simulator_results.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    return df


def _pick_k(policy: PolicySpec, controller: QLearningController, state: dict[str, Any]) -> int:
    if policy.mode == "fixed":
        return int(policy.fixed_k)
    return controller.choose_k(state, greedy=True)


def run_real_policy(
    decoder: HuggingFaceTreeSpeculativeDecoder,
    controller: QLearningController,
    policy: PolicySpec,
    prompt: str,
    workload: str,
    branching_factor: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run one real-model rollout for one policy and prompt."""
    input_ids = decoder.encode_prompt(prompt)
    state = initialize_state(workload_style=workload)
    state["prompt_length"] = int(input_ids.shape[1])
    state["tree_branching_factor"] = branching_factor
    start_length = int(input_ids.shape[1])
    rollout: list[dict[str, float]] = []

    while int(input_ids.shape[1]) - start_length < max_new_tokens:
        k = _pick_k(policy, controller, state)
        new_input_ids, accepted_tokens, extra_info = decoder.speculative_decode_step(
            input_ids=input_ids,
            k=k,
            state=state,
            branching_factor=branching_factor,
        )
        next_state = update_state_from_feedback(state, k, accepted_tokens, extra_info)
        reward = compute_reward(next_state, k, accepted_tokens, extra_info)
        if policy.mode == "adaptive":
            controller.update_policy(state, k, reward, next_state)

        rollout.append(
            {
                "k": float(k),
                "accepted_tokens": float(accepted_tokens),
                "acceptance_rate": float(extra_info["acceptance_rate"]),
                "reward": float(reward),
                "step_time_s": float(extra_info["step_time_s"]),
            }
        )
        input_ids = new_input_ids
        state = next_state
        if int(input_ids.shape[1]) - start_length >= max_new_tokens:
            break

    summary = rollout_summary(rollout)
    summary["generated_length_delta"] = float(int(input_ids.shape[1]) - start_length)
    summary["generated_text"] = decoder.decode_tokens(input_ids)
    return summary


def run_real_benchmark(args: argparse.Namespace, output_dir: Path):
    """Run the real-model policy comparison."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    workloads = ("steady_low_load", "bursty_high_load")
    controllers = {
        workload: ensure_controller(
            path=str(output_dir / f"{workload}_controller.json"),
            workload=workload,
            episodes=args.train_episodes,
            episode_length=args.episode_length,
            seed=args.seed,
        )
        for workload in workloads
    }
    decoder = HuggingFaceTreeSpeculativeDecoder(
        draft_model_name=args.draft_model,
        target_model_name=args.target_model,
    )

    for workload in workloads:
        for prompt in DEFAULT_PROMPTS:
            for policy in POLICIES:
                result = run_real_policy(
                    decoder=decoder,
                    controller=controllers[workload],
                    policy=policy,
                    prompt=prompt,
                    workload=workload,
                    branching_factor=args.branching_factor,
                    max_new_tokens=args.max_new_tokens,
                )
                rows.append(
                    {
                        "benchmark": "real_models",
                        "workload": workload,
                        "prompt": prompt,
                        "policy": policy.name,
                        "draft_model": args.draft_model,
                        "target_model": args.target_model,
                        "branching_factor": args.branching_factor,
                        "max_new_tokens": args.max_new_tokens,
                        **result,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "real_results.csv", index=False)
    with open(output_dir / "real_results.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    return df


def plot_results(sim_df, real_df, output_dir: Path) -> None:
    """Create benchmark plots from the result tables."""
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sim_plot = sim_df.pivot(index="policy", columns="workload", values="avg_reward").reindex([p.name for p in POLICIES])
    ax = sim_plot.plot(kind="bar", figsize=(10, 6), title="Simulator Average Reward by Policy")
    ax.set_ylabel("Average reward")
    ax.figure.tight_layout()
    ax.figure.savefig(plots_dir / "simulator_avg_reward.png", dpi=180)
    plt.close(ax.figure)

    real_grouped = real_df.groupby(["policy", "workload"], as_index=False)[["avg_reward", "avg_acceptance_rate", "avg_step_time_s"]].mean()
    real_plot = real_grouped.pivot(index="policy", columns="workload", values="avg_reward").reindex([p.name for p in POLICIES])
    ax = real_plot.plot(kind="bar", figsize=(10, 6), title="Real-Model Average Reward by Policy")
    ax.set_ylabel("Average reward")
    ax.figure.tight_layout()
    ax.figure.savefig(plots_dir / "real_avg_reward.png", dpi=180)
    plt.close(ax.figure)

    acceptance_plot = real_grouped.pivot(index="policy", columns="workload", values="avg_acceptance_rate").reindex([p.name for p in POLICIES])
    ax = acceptance_plot.plot(kind="bar", figsize=(10, 6), title="Real-Model Acceptance Rate by Policy")
    ax.set_ylabel("Average acceptance rate")
    ax.figure.tight_layout()
    ax.figure.savefig(plots_dir / "real_acceptance_rate.png", dpi=180)
    plt.close(ax.figure)


def run_benchmark(args: argparse.Namespace) -> None:
    """Run simulator and real-model benchmarks, then save artifacts."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_df = run_simulator_benchmark(args, output_dir)
    real_df = run_real_benchmark(args, output_dir)
    plot_results(sim_df, real_df, output_dir)

    summary = {
        "simulator_rows": int(len(sim_df)),
        "real_rows": int(len(real_df)),
        "prompts": list(DEFAULT_PROMPTS),
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("Benchmark run complete")
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for the single-file implementation."""
    parser = argparse.ArgumentParser(description="Single-file dynamic speculative decoding implementation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap-venv", help="create the virtualenv and install dependencies")
    bootstrap.add_argument("--venv-dir", default=".venv", help="virtualenv directory")
    bootstrap.add_argument("--python", default="python3.12", help="python executable used to create the venv")

    train = subparsers.add_parser("train", help="train the simulator controller")
    train.add_argument("--episodes", type=int, default=250, help="training episodes")
    train.add_argument("--eval-episodes", type=int, default=20, help="evaluation episodes after training")
    train.add_argument("--episode-length", type=int, default=32, help="steps per episode")
    train.add_argument("--workload", choices=("steady_low_load", "bursty_high_load"), default="steady_low_load")
    train.add_argument("--checkpoint", default="artifacts/q_table.json", help="checkpoint path")
    train.add_argument("--seed", type=int, default=7, help="random seed")

    decode = subparsers.add_parser("decode", help="run real-model decoding")
    decode.add_argument("--prompt", required=True, help="prompt to decode from")
    decode.add_argument("--target-model", default="gpt2", help="target Hugging Face causal LM")
    decode.add_argument("--draft-model", default="distilgpt2", help="draft Hugging Face causal LM")
    decode.add_argument("--max-new-tokens", type=int, default=12, help="maximum tokens to generate")
    decode.add_argument("--workload", choices=("steady_low_load", "bursty_high_load"), default="steady_low_load")
    decode.add_argument("--checkpoint", default="artifacts/q_table.json", help="controller checkpoint path")
    decode.add_argument("--train-episodes", type=int, default=150, help="episodes to train if checkpoint is missing")
    decode.add_argument("--episode-length", type=int, default=32, help="episode length used when training missing checkpoint")
    decode.add_argument("--branching-factor", type=int, default=2, help="tree branching factor")
    decode.add_argument("--seed", type=int, default=7, help="random seed")

    bench = subparsers.add_parser("benchmark", help="run simulator and real-model benchmarks")
    bench.add_argument("--output-dir", default="artifacts/benchmarks", help="artifact directory")
    bench.add_argument("--seed", type=int, default=7, help="random seed")
    bench.add_argument("--train-episodes", type=int, default=100, help="simulator training episodes")
    bench.add_argument("--sim-eval-episodes", type=int, default=30, help="simulator evaluation episodes")
    bench.add_argument("--episode-length", type=int, default=16, help="episode length")
    bench.add_argument("--draft-model", default="distilgpt2", help="draft model for real evaluation")
    bench.add_argument("--target-model", default="gpt2", help="target model for real evaluation")
    bench.add_argument("--branching-factor", type=int, default=2, help="tree branching factor")
    bench.add_argument("--max-new-tokens", type=int, default=2, help="real decoding token budget")

    return parser


def main() -> None:
    """Entry point for the single-file implementation."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "bootstrap-venv":
        bootstrap_venv(venv_dir=args.venv_dir, python_cmd=args.python)
        return
    if args.command == "train":
        run_train(args)
        return
    if args.command == "decode":
        run_decode(args)
        return
    if args.command == "benchmark":
        run_benchmark(args)
        return

    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
