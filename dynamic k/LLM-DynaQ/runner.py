"""Minimal runner for LLM-backed Dyna-Q."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_DYNAQ_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(LLM_DYNAQ_ROOT))

from decoding.speculative import HuggingFaceTreeSpeculativeDecoder
from rl.controller import DynaQController

from llm_env import LLMTreeSpeculativeDecodingEnv
from prompt_dataset import PromptDataset


DATASET_PATH = "hf://datasets/LLMs/Alpaca-ShareGPT/alpaca_sharedgpt_data.json"
DRAFT_MODEL = "distilgpt2"
TARGET_MODEL = "gpt2"
WORKLOAD = "steady_low_load"
SEED = 7
EPISODES = 2500
EPISODE_LENGTH = 8
PLANNING_STEPS = 5


def main() -> None:
    prompt_dataset = PromptDataset.from_hf_json(
        DATASET_PATH,
        seed=SEED,
        max_prompts=max(200, EPISODES),
        sample_strategy="sequential",
    )
    decoder = HuggingFaceTreeSpeculativeDecoder(
        draft_model_name=DRAFT_MODEL,
        target_model_name=TARGET_MODEL,
    )
    env = LLMTreeSpeculativeDecodingEnv(
        decoder=decoder,
        prompt_dataset=prompt_dataset,
        workload_style=WORKLOAD,
        episode_length=EPISODE_LENGTH,
        seed=SEED,
    )

    controller = DynaQController(planning_steps=PLANNING_STEPS, seed=SEED)
    history = controller.train(env, episodes=EPISODES)

    eval_metrics = controller.evaluate(env, episodes=3)

    print("Training complete.")
    print(f"Episodes: {EPISODES}")
    print(f"Last avg reward: {history[-1].average_reward:.4f}")
    print(f"Last avg k: {history[-1].average_k:.3f}")
    print(f"Eval avg acceptance: {eval_metrics['avg_acceptance_rate']:.4f}")
    print(f"Final epsilon: {history[-1].epsilon:.4f}")


if __name__ == "__main__":
    main()
