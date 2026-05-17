"""Small runnable entrypoint for the Sprint 1 simulator-first prototype."""

from __future__ import annotations

import argparse
import os

from simulated_qlearning.controller import QLearningController
from simulated_qlearning.env import TreeSpeculativeDecodingEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Sprint 1 RL prototype.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=250,
        help="number of training episodes",
    )
    parser.add_argument(
        "--workload",
        choices=("steady_low_load", "bursty_high_load"),
        default="steady_low_load",
        help="workload style for training",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=32,
        help="steps per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="random seed",
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/q_table.json",
        help="where to save the learned Q-table",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    env = TreeSpeculativeDecodingEnv(
        workload_style=args.workload,
        episode_length=args.episode_length,
        seed=args.seed,
    )
    controller = QLearningController(seed=args.seed)

    history = controller.train(env, episodes=args.episodes)
    evaluation = controller.evaluate(env, episodes=20)
    checkpoint_dir = os.path.dirname(args.checkpoint)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    controller.save(args.checkpoint)

    print("Training complete")
    print(f"episodes: {args.episodes}")
    print(f"final_epsilon: {controller.epsilon:.4f}")
    print(f"last_episode_avg_reward: {history[-1].average_reward:.3f}")
    print(f"last_episode_avg_k: {history[-1].average_k:.3f}")
    print(f"checkpoint: {args.checkpoint}")
    print("evaluation:")
    for key, value in evaluation.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
