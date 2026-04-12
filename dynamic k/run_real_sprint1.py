"""End-to-end Sprint 1 runner using Rayane's tree-based decoding engine."""

from __future__ import annotations

import argparse
import os
import sys

# Add the tree-based decoding module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Tree-Based-Speculative-Decoding"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from draft import DraftModel
from target import TargetModel
from decoding import speculative_decode_step

from rl.controller import QLearningController
from simulation.env import TreeSpeculativeDecodingEnv
from state.features import initialize_state
from utils.metrics import compute_reward, rollout_summary, update_state_from_feedback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full Sprint 1 stack on real models.")
    parser.add_argument("--prompt", required=True, help="prompt to decode from")
    parser.add_argument(
        "--target-model",
        default="gpt2-medium",
        help="target Hugging Face causal LM",
    )
    parser.add_argument(
        "--draft-model",
        default="gpt2",
        help="draft Hugging Face causal LM",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="maximum tokens to generate",
    )
    parser.add_argument(
        "--workload",
        choices=("steady_low_load", "bursty_high_load"),
        default="steady_low_load",
        help="serving regime used for controller state",
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/q_table.json",
        help="controller checkpoint path",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=150,
        help="episodes to train if the checkpoint does not exist",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=32,
        help="episode length used when training a missing controller",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="random seed",
    )
    return parser


def ensure_controller(path: str, workload: str, episodes: int, episode_length: int, seed: int) -> QLearningController:
    """Load or train the controller checkpoint."""
    if os.path.exists(path):
        print(f"Loading controller from {path}")
        return QLearningController.load(path)

    print(f"No checkpoint found. Training controller for {episodes} episodes...")
    env = TreeSpeculativeDecodingEnv(
        workload_style=workload,
        episode_length=episode_length,
        seed=seed,
    )
    controller = QLearningController(seed=seed)
    controller.train(env, episodes=episodes)
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    controller.save(path)
    print(f"Controller trained and saved to {path}")
    return controller


def main() -> None:
    args = build_parser().parse_args()

    # ── Load RL controller ──────────────────────────────────────────────
    controller = ensure_controller(
        path=args.checkpoint,
        workload=args.workload,
        episodes=args.train_episodes,
        episode_length=args.episode_length,
        seed=args.seed,
    )

    # ── Load models (Rayane's engine) ───────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading draft model: {args.draft_model}")
    draft_hf = AutoModelForCausalLM.from_pretrained(
        args.draft_model, torch_dtype=torch.float32
    ).to(device)

    print(f"Loading target model: {args.target_model}")
    target_hf = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.float32
    ).to(device)

    draft_model = DraftModel(draft_hf)
    target_model = TargetModel(target_hf)

    # ── Tokenize prompt ─────────────────────────────────────────────────
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    start_length = input_ids.shape[1]
    print(f"Prompt ({start_length} tokens): {args.prompt!r}\n")

    # ── Initialize state ────────────────────────────────────────────────
    state = initialize_state(workload_style=args.workload)
    state["prompt_length"] = start_length
    state["tree_branching_factor"] = 2

    # ── Main decoding loop (RL controls k each round) ───────────────────
    rollout: list[dict[str, float]] = []

    print("Starting RL-controlled speculative decoding...")
    print(f"{'Step':>4}  {'k':>2}  {'Accepted':>8}  {'Rate':>6}  {'Reward':>7}  {'Time':>6}")
    print("-" * 50)

    while int(input_ids.shape[1]) - start_length < args.max_new_tokens:
        k = controller.choose_k(state, greedy=True)

        new_input_ids, accepted_tokens, extra_info = speculative_decode_step(
            input_ids=input_ids,
            k=k,
            draft_model=draft_model,
            target_model=target_model,
            state=state,
        )

        next_state = update_state_from_feedback(state, k, accepted_tokens, extra_info)
        reward = compute_reward(next_state, k, accepted_tokens, extra_info)
        controller.update_policy(state, k, reward, next_state)

        step_num = len(rollout) + 1
        print(f"{step_num:>4}  {k:>2}  {accepted_tokens:>8}  "
              f"{extra_info['acceptance_rate']:>5.1%}  {reward:>7.2f}  "
              f"{extra_info['step_time_s']:>5.1f}s")

        rollout.append({
            "k": float(k),
            "accepted_tokens": float(accepted_tokens),
            "acceptance_rate": float(extra_info["acceptance_rate"]),
            "reward": float(reward),
            "step_time_s": float(extra_info["step_time_s"]),
        })

        input_ids = new_input_ids
        state = next_state

    # ── Results ─────────────────────────────────────────────────────────
    generated_text = tokenizer.decode(
        input_ids[0, start_length:], skip_special_tokens=True
    )
    summary = rollout_summary(rollout)

    print("\n" + "=" * 50)
    print("Run complete")
    print(f"  target_model : {args.target_model}")
    print(f"  draft_model  : {args.draft_model}")
    print(f"  checkpoint   : {args.checkpoint}")
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")
    print(f"\nGenerated text:\n  {generated_text!r}")

    controller.save(args.checkpoint)


if __name__ == "__main__":
    main()
