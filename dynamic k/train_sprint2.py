"""Sprint 2 training entrypoint: Dyna-Q (Model-Based RL).

This script trains a :class:`~rl.controller.DynaQController` against the
same :class:`~simulation.env.TreeSpeculativeDecodingEnv` used in Sprint 1,
but the controller now:

* treats the environment as a **black box** (no formula access),
* learns transition dynamics P(s'|s,a) and reward R(s,a) online,
* performs ``--planning-steps`` simulated Q-updates after every real step.

Convergence comparison
----------------------
The script optionally trains a plain :class:`~rl.controller.QLearningController`
(``--compare``) and prints a side-by-side summary so you can see how quickly
Dyna-Q converges relative to model-free Q-learning.

Typical usage
-------------
::

    # Train Dyna-Q only (steady workload, 20 planning steps):
    python train_sprint2.py

    # Train Dyna-Q + compare against Q-learning (bursty workload):
    python train_sprint2.py --workload bursty_high_load --compare

    # Sweep planning steps to study sample efficiency:
    python train_sprint2.py --planning-steps 5 --episodes 100
    python train_sprint2.py --planning-steps 50 --episodes 100

"""

from __future__ import annotations

import argparse
import os
import time
from typing import Union

from rl.controller import (
    DynaEpisodeSummary,
    DynaQController,
    EpisodeSummary,
    QLearningController,
)
from simulation.env import TreeSpeculativeDecodingEnv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Sprint 2 Dyna-Q prototype.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        help="workload style for both train and evaluation environments",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=32,
        help="number of steps per episode",
    )
    parser.add_argument(
        "--planning-steps",
        type=int,
        default=20,
        help="simulated Q-updates per real environment step (Dyna-Q N)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/dynaq_table.json",
        help="where to save the Dyna-Q Q-table",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also train a plain Q-learning baseline for comparison",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="print a progress line every N episodes",
    )
    return parser


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _convergence_episode(
    history: list[Union[DynaEpisodeSummary, EpisodeSummary]],
    threshold: float = 1.5,
    window: int = 10,
) -> int | None:
    """Return the first episode at which average reward exceeds *threshold*.

    Uses a rolling mean over *window* episodes to smooth noise.
    Returns ``None`` if the threshold was never reached.
    """
    for i in range(window - 1, len(history)):
        window_avg = sum(h.average_reward for h in history[i - window + 1 : i + 1]) / window
        if window_avg >= threshold:
            return history[i].episode
    return None


def train_dynaq(
    args: argparse.Namespace,
) -> tuple[DynaQController, list[DynaEpisodeSummary], float]:
    """Instantiate and train the Dyna-Q controller.

    Returns
    -------
    controller, history, wall_clock_seconds
    """
    env = TreeSpeculativeDecodingEnv(
        workload_style=args.workload,
        episode_length=args.episode_length,
        seed=args.seed,
    )
    controller = DynaQController(
        planning_steps=args.planning_steps,
        seed=args.seed,
    )

    print(
        f"\n{'='*60}\n"
        f"  Sprint 2 — Dyna-Q  (planning_steps={args.planning_steps})\n"
        f"{'='*60}"
    )
    print(f"  workload      : {args.workload}")
    print(f"  episodes      : {args.episodes}")
    print(f"  episode_length: {args.episode_length}")
    print(f"  seed          : {args.seed}\n")

    t0 = time.perf_counter()
    history = controller.train(env, episodes=args.episodes)
    elapsed = time.perf_counter() - t0

    _print_progress(history, args.log_interval, label="DynaQ")
    return controller, history, elapsed


def train_qlearning(
    args: argparse.Namespace,
) -> tuple[QLearningController, list[EpisodeSummary], float]:
    """Instantiate and train the Sprint 1 Q-learning baseline.

    Returns
    -------
    controller, history, wall_clock_seconds
    """
    env = TreeSpeculativeDecodingEnv(
        workload_style=args.workload,
        episode_length=args.episode_length,
        seed=args.seed,
    )
    controller = QLearningController(seed=args.seed)

    print(
        f"\n{'='*60}\n"
        f"  Sprint 1 baseline — Q-learning  (no planning)\n"
        f"{'='*60}"
    )

    t0 = time.perf_counter()
    history = controller.train(env, episodes=args.episodes)
    elapsed = time.perf_counter() - t0

    _print_progress(history, args.log_interval, label="QL    ")
    return controller, history, elapsed


def _print_progress(
    history: list[Union[DynaEpisodeSummary, EpisodeSummary]],
    interval: int,
    label: str,
) -> None:
    """Print one progress line every *interval* episodes."""
    for h in history:
        if h.episode % interval == 0 or h.episode == 1:
            extra = ""
            if isinstance(h, DynaEpisodeSummary):
                extra = (
                    f"  plan_updates={h.planning_updates:4d}"
                    f"  model_cov={h.model_coverage:4d}"
                )
            print(
                f"  [{label}] ep={h.episode:4d}"
                f"  avg_r={h.average_reward:+6.3f}"
                f"  avg_k={h.average_k:.2f}"
                f"  eps={h.epsilon:.4f}"
                + extra
            )


# ---------------------------------------------------------------------------
# Evaluation & comparison report
# ---------------------------------------------------------------------------


def evaluate_controller(
    controller: QLearningController,
    args: argparse.Namespace,
    episodes: int = 20,
) -> dict[str, float]:
    """Run a greedy evaluation on a fresh environment instance."""
    eval_env = TreeSpeculativeDecodingEnv(
        workload_style=args.workload,
        episode_length=args.episode_length,
        seed=args.seed + 999,  # different seed to avoid overfitting to train env
    )
    return controller.evaluate(eval_env, episodes=episodes)


def _print_comparison(
    dynaq_history: list[DynaEpisodeSummary],
    ql_history: list[EpisodeSummary],
    dynaq_eval: dict[str, float],
    ql_eval: dict[str, float],
    dynaq_elapsed: float,
    ql_elapsed: float,
) -> None:
    """Print a side-by-side convergence & evaluation comparison."""
    dynaq_conv = _convergence_episode(dynaq_history)
    ql_conv = _convergence_episode(ql_history)

    dynaq_final_avg = sum(h.average_reward for h in dynaq_history[-20:]) / 20
    ql_final_avg = sum(h.average_reward for h in ql_history[-20:]) / 20

    print(f"\n{'='*60}")
    print("  Convergence & Sample-Efficiency Comparison")
    print(f"{'='*60}")
    print(f"  {'Metric':<35} {'Dyna-Q':>10} {'Q-Learning':>12}")
    print(f"  {'-'*57}")
    print(
        f"  {'Convergence episode (avg_r >= 1.5)':<35}"
        f" {str(dynaq_conv) if dynaq_conv else 'N/A':>10}"
        f" {str(ql_conv) if ql_conv else 'N/A':>12}"
    )
    print(
        f"  {'Final 20-ep avg reward':<35}"
        f" {dynaq_final_avg:>10.4f}"
        f" {ql_final_avg:>12.4f}"
    )
    print(
        f"  {'Eval avg reward (greedy)':<35}"
        f" {dynaq_eval['avg_reward']:>10.4f}"
        f" {ql_eval['avg_reward']:>12.4f}"
    )
    print(
        f"  {'Eval avg k (greedy)':<35}"
        f" {dynaq_eval['avg_k']:>10.4f}"
        f" {ql_eval['avg_k']:>12.4f}"
    )
    print(
        f"  {'Eval avg acceptance rate':<35}"
        f" {dynaq_eval['avg_acceptance_rate']:>10.4f}"
        f" {ql_eval['avg_acceptance_rate']:>12.4f}"
    )
    print(
        f"  {'Wall-clock training time (s)':<35}"
        f" {dynaq_elapsed:>10.2f}"
        f" {ql_elapsed:>12.2f}"
    )
    print()

    # Highlight sample-efficiency insight.
    if dynaq_conv and ql_conv:
        ratio = ql_conv / dynaq_conv if dynaq_conv > 0 else float("inf")
        print(
            f"  >>> Dyna-Q converged {ratio:.1f}x faster than pure Q-learning "
            f"({dynaq_conv} vs {ql_conv} episodes)."
        )
    elif dynaq_conv and not ql_conv:
        print(
            f"  >>> Dyna-Q converged at episode {dynaq_conv}; "
            "Q-learning never reached the threshold."
        )
    else:
        print("  >>> Neither controller reached the reward threshold.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()

    # --- Train Dyna-Q ---
    dynaq_ctrl, dynaq_history, dynaq_elapsed = train_dynaq(args)

    # --- Optional Q-learning baseline ---
    ql_ctrl: QLearningController | None = None
    ql_history: list[EpisodeSummary] | None = None
    ql_elapsed: float = 0.0

    if args.compare:
        ql_ctrl, ql_history, ql_elapsed = train_qlearning(args)

    # --- Evaluate ---
    print("\nRunning greedy evaluation …")
    dynaq_eval = evaluate_controller(dynaq_ctrl, args)

    # --- Save checkpoint ---
    checkpoint_dir = os.path.dirname(args.checkpoint)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    dynaq_ctrl.save(args.checkpoint)

    # --- Final report ---
    print(f"\n{'='*60}")
    print("  Sprint 2 — Dyna-Q Results")
    print(f"{'='*60}")
    print(f"  planning_steps       : {args.planning_steps}")
    print(f"  episodes             : {args.episodes}")
    print(f"  final epsilon        : {dynaq_ctrl.epsilon:.4f}")
    print(f"  model coverage (s,a) : {dynaq_ctrl.env_model.num_observed_pairs}")
    print(f"  wall-clock time      : {dynaq_elapsed:.2f}s")
    print(f"  last-ep avg reward   : {dynaq_history[-1].average_reward:.3f}")
    print(f"  last-ep avg k        : {dynaq_history[-1].average_k:.3f}")
    print(f"  checkpoint           : {args.checkpoint}")
    print("  evaluation (greedy)  :")
    for key, value in dynaq_eval.items():
        print(f"    {key}: {value:.4f}")

    if args.compare and ql_ctrl is not None and ql_history is not None:
        ql_eval = evaluate_controller(ql_ctrl, args)
        _print_comparison(
            dynaq_history, ql_history,
            dynaq_eval, ql_eval,
            dynaq_elapsed, ql_elapsed,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
