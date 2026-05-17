"""Batch experiment runner for ParetoServe Sprint 1.

Outputs:
- JSON summaries
- CSV tables
- PNG plots
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decoding.speculative import HuggingFaceTreeSpeculativeDecoder
from simulated_qlearning.controller import QLearningController
from simulated_qlearning.env import TreeSpeculativeDecodingEnv
from state.features import initialize_state
from utils.model_config import resolve_model_pair
from utils.metrics import compute_real_latency_reward, compute_reward, rollout_summary, update_state_from_feedback


DEFAULT_PROMPTS = (
    "Adaptive inference control",
    "ParetoServe chooses speculative depth",
    "Queue pressure rises during a traffic burst",
)
DEFAULT_DRAFT_MODEL, DEFAULT_TARGET_MODEL, DEFAULT_MODEL_PROFILE = resolve_model_pair()


@dataclass(frozen=True)
class PolicySpec:
    name: str
    mode: str
    fixed_k: int | None = None


POLICIES = (
    PolicySpec(name="fixed_k_1", mode="fixed", fixed_k=1),
    PolicySpec(name="fixed_k_3", mode="fixed", fixed_k=3),
    PolicySpec(name="fixed_k_5", mode="fixed", fixed_k=5),
    PolicySpec(name="adaptive_rl", mode="adaptive", fixed_k=None),
)
REAL_POLICY_ORDER = ["autoregressive_cached", *[p.name for p in POLICIES]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ParetoServe Sprint 1 benchmarks.")
    parser.add_argument("--output-dir", default="artifacts/benchmarks", help="artifact directory")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--train-episodes", type=int, default=120, help="simulator training episodes")
    parser.add_argument("--sim-eval-episodes", type=int, default=40, help="simulator evaluation episodes")
    parser.add_argument("--episode-length", type=int, default=16, help="episode length")
    parser.add_argument(
        "--draft-model",
        default=DEFAULT_DRAFT_MODEL,
        help=(
            "draft model for real evaluation; defaults come from "
            "PARETOSERVE_DRAFT_MODEL or PARETOSERVE_MODEL_PROFILE "
            f"(current profile: {DEFAULT_MODEL_PROFILE})"
        ),
    )
    parser.add_argument(
        "--target-model",
        default=DEFAULT_TARGET_MODEL,
        help=(
            "target model for real evaluation; defaults come from "
            "PARETOSERVE_TARGET_MODEL or PARETOSERVE_MODEL_PROFILE "
            f"(current profile: {DEFAULT_MODEL_PROFILE})"
        ),
    )
    parser.add_argument("--branching-factor", type=int, default=2, help="real decoding branching factor")
    parser.add_argument("--max-new-tokens", type=int, default=3, help="real decoding token budget")
    parser.add_argument(
        "--reward-mode",
        choices=("simulator", "latency_aware"),
        default="latency_aware",
        help="reward used for real-model adaptive RL updates",
    )
    return parser


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def train_controller(workload: str, episodes: int, episode_length: int, seed: int) -> QLearningController:
    env = TreeSpeculativeDecodingEnv(
        workload_style=workload,
        episode_length=episode_length,
        seed=seed,
    )
    controller = QLearningController(seed=seed)
    controller.train(env, episodes=episodes)
    return controller


def evaluate_fixed_policy_sim(
    workload: str,
    fixed_k: int,
    episodes: int,
    episode_length: int,
    seed: int,
) -> dict[str, float]:
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


def run_simulator_benchmark(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    workloads = ("steady_low_load", "bursty_high_load")

    for workload in workloads:
        controller = train_controller(
            workload=workload,
            episodes=args.train_episodes,
            episode_length=args.episode_length,
            seed=args.seed,
        )
        adaptive_env = TreeSpeculativeDecodingEnv(
            workload_style=workload,
            episode_length=args.episode_length,
            seed=args.seed,
        )
        adaptive_metrics = controller.evaluate(adaptive_env, episodes=args.sim_eval_episodes)
        rows.append(
            {
                "benchmark": "simulator",
                "workload": workload,
                "policy": "adaptive_rl",
                **adaptive_metrics,
            }
        )

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
            rows.append(
                {
                    "benchmark": "simulator",
                    "workload": workload,
                    "policy": policy.name,
                    **metrics,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "simulator_results.csv", index=False)
    with open(output_dir / "simulator_results.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    return df


def _pick_k(policy: PolicySpec, controller: QLearningController, state: dict[str, Any]) -> int:
    if policy.mode == "fixed":
        return int(policy.fixed_k)
    return controller.choose_k(state, greedy=True)


def run_autoregressive_policy(
    decoder: HuggingFaceTreeSpeculativeDecoder,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    input_ids = decoder.encode_prompt(prompt)
    generated = input_ids.clone()
    current_input = input_ids
    past_key_values = None
    rollout: list[dict[str, float]] = []
    start = time.perf_counter()

    for _ in range(max_new_tokens):
        step_start = time.perf_counter()
        outputs = decoder.target_model(
            input_ids=current_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        step_time_s = time.perf_counter() - step_start
        next_token_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        generated = decoder._append_tokens(generated, (next_token_id,))
        past_key_values = outputs.past_key_values
        current_input = decoder._append_tokens(current_input[:, :0], (next_token_id,))
        rollout.append(
            {
                "k": 0.0,
                "accepted_tokens": 0.0,
                "acceptance_rate": 0.0,
                "reward": 0.0,
                "step_time_s": step_time_s,
            }
        )

    summary = rollout_summary(rollout)
    generated_length_delta = float(int(generated.shape[1]) - int(input_ids.shape[1]))
    total_time_s = time.perf_counter() - start
    summary.update(
        {
            "generated_length_delta": generated_length_delta,
            "generated_text": decoder.decode_tokens(generated),
            "target_forward_passes": float(max_new_tokens),
            "draft_forward_passes": 0.0,
            "tokens_per_s": generated_length_delta / max(total_time_s, 1e-6),
        }
    )
    return summary


def run_real_policy(
    decoder: HuggingFaceTreeSpeculativeDecoder,
    controller: QLearningController,
    policy: PolicySpec,
    prompt: str,
    workload: str,
    branching_factor: int,
    max_new_tokens: int,
    reward_mode: str,
) -> dict[str, Any]:
    input_ids = decoder.encode_prompt(prompt)
    state = initialize_state(workload_style=workload)
    state["prompt_length"] = int(input_ids.shape[1])
    state["tree_branching_factor"] = branching_factor
    start_length = int(input_ids.shape[1])
    rollout: list[dict[str, float]] = []
    total_target_forward_passes = 0.0
    total_draft_forward_passes = 0.0
    reward_fn = compute_real_latency_reward if reward_mode == "latency_aware" else compute_reward

    while int(input_ids.shape[1]) - start_length < max_new_tokens:
        k = _pick_k(policy, controller, state)
        new_input_ids, accepted_tokens, extra_info = decoder.speculative_decode_step(
            input_ids=input_ids,
            k=k,
            state=state,
            branching_factor=branching_factor,
        )
        next_state = update_state_from_feedback(state, k, accepted_tokens, extra_info)
        reward = reward_fn(next_state, k, accepted_tokens, extra_info)
        if policy.mode == "adaptive":
            controller.update_policy(state, k, reward, next_state)
        total_target_forward_passes += float(extra_info.get("target_forward_passes", 1.0))
        total_draft_forward_passes += float(extra_info.get("draft_forward_passes", 0.0))

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
    total_time_s = sum(row["step_time_s"] for row in rollout)
    summary["target_forward_passes"] = total_target_forward_passes
    summary["draft_forward_passes"] = total_draft_forward_passes
    summary["tokens_per_s"] = summary["generated_length_delta"] / max(total_time_s, 1e-6)
    return summary


def run_real_benchmark(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    workloads = ("steady_low_load", "bursty_high_load")
    controllers = {
        workload: train_controller(
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
            baseline = run_autoregressive_policy(
                decoder=decoder,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )
            rows.append(
                {
                    "benchmark": "real_models",
                    "workload": workload,
                    "prompt": prompt,
                    "policy": "autoregressive_cached",
                    "draft_model": "",
                    "target_model": args.target_model,
                    "branching_factor": 0,
                    "max_new_tokens": args.max_new_tokens,
                    **baseline,
                }
            )
            for policy in POLICIES:
                result = run_real_policy(
                    decoder=decoder,
                    controller=controllers[workload],
                    policy=policy,
                    prompt=prompt,
                    workload=workload,
                    branching_factor=args.branching_factor,
                    max_new_tokens=args.max_new_tokens,
                    reward_mode=args.reward_mode,
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


def plot_results(sim_df: pd.DataFrame, real_df: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = ensure_dir(output_dir / "plots")

    sim_plot = (
        sim_df.pivot(index="policy", columns="workload", values="avg_reward")
        .reindex([p.name for p in POLICIES if p.mode == "fixed"] + ["adaptive_rl"])
    )
    ax = sim_plot.plot(kind="bar", figsize=(10, 6), title="Simulator Average Reward by Policy")
    ax.set_ylabel("Average reward")
    ax.figure.tight_layout()
    ax.figure.savefig(plots_dir / "simulator_avg_reward.png", dpi=180)
    plt.close(ax.figure)

    real_grouped = (
        real_df.groupby(
            ["policy", "workload"],
            as_index=False,
        )[["avg_reward", "avg_acceptance_rate", "avg_step_time_s", "tokens_per_s"]]
        .mean()
    )
    real_plot = real_grouped.pivot(index="policy", columns="workload", values="avg_reward").reindex(REAL_POLICY_ORDER)
    ax = real_plot.plot(kind="bar", figsize=(10, 6), title="Real-Model Average Reward by Policy")
    ax.set_ylabel("Average reward")
    ax.figure.tight_layout()
    ax.figure.savefig(plots_dir / "real_avg_reward.png", dpi=180)
    plt.close(ax.figure)

    acceptance_plot = real_grouped.pivot(index="policy", columns="workload", values="avg_acceptance_rate").reindex(
        REAL_POLICY_ORDER
    )
    ax = acceptance_plot.plot(kind="bar", figsize=(10, 6), title="Real-Model Acceptance Rate by Policy")
    ax.set_ylabel("Average acceptance rate")
    ax.figure.tight_layout()
    ax.figure.savefig(plots_dir / "real_acceptance_rate.png", dpi=180)
    plt.close(ax.figure)

    throughput_plot = real_grouped.pivot(index="policy", columns="workload", values="tokens_per_s").reindex(
        REAL_POLICY_ORDER
    )
    ax = throughput_plot.plot(kind="bar", figsize=(10, 6), title="Real-Model Throughput by Policy")
    ax.set_ylabel("Tokens / second")
    ax.figure.tight_layout()
    ax.figure.savefig(plots_dir / "real_tokens_per_s.png", dpi=180)
    plt.close(ax.figure)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)

    sim_df = run_simulator_benchmark(args, output_dir)
    real_df = run_real_benchmark(args, output_dir)
    plot_results(sim_df, real_df, output_dir)

    summary = {
        "simulator_rows": int(len(sim_df)),
        "real_rows": int(len(real_df)),
        "prompts": list(DEFAULT_PROMPTS),
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "reward_mode": args.reward_mode,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Benchmark run complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
