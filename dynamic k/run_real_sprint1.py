"""End-to-end Sprint 1 runner with real Hugging Face models."""

from __future__ import annotations

import argparse
import os

from decoding.speculative import HuggingFaceTreeSpeculativeDecoder
from rl.controller import QLearningController
from simulation.env import TreeSpeculativeDecodingEnv
from state.features import initialize_state
from utils.metrics import compute_reward, rollout_summary, update_state_from_feedback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full Sprint 1 stack on real models.")
    parser.add_argument("--prompt", required=True, help="prompt to decode from")
    parser.add_argument(
        "--target-model",
        default="distilgpt2",
        help="target Hugging Face causal LM",
    )
    parser.add_argument(
        "--draft-model",
        default="sshleifer/tiny-gpt2",
        help="draft Hugging Face causal LM",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=12,
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
        "--branching-factor",
        type=int,
        default=3,
        help="tree branching factor used during real decoding",
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
        return QLearningController.load(path)

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
    return controller


def main() -> None:
    args = build_parser().parse_args()

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
        if rollout and int(input_ids.shape[1]) - start_length >= args.max_new_tokens:
            break

    generated_text = decoder.decode_tokens(input_ids)
    summary = rollout_summary(rollout)

    print("Real decoding run complete")
    print(f"target_model: {args.target_model}")
    print(f"draft_model: {args.draft_model}")
    print(f"checkpoint: {args.checkpoint}")
    print("summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")
    print("generated_text:")
    print(generated_text)

    controller.save(args.checkpoint)


if __name__ == "__main__":
    main()
