"""Dyna-Q Controller logic exclusively interacting with the LLM environment."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from state.features import K_VALUES, featurize_state
from model import TabularEnvironmentModel


@dataclass(frozen=True)
class DynaEpisodeSummary:
    """Extended training summary for one Dyna-Q episode."""

    episode: int
    total_reward: float
    average_reward: float
    average_k: float
    epsilon: float
    steps: int
    planning_updates: int
    model_coverage: int  # unique (s,a) pairs observed by the world model


class DynaQController:
    """Dyna-Q controller — model-based RL using real LLMs.

    Architecture
    ------------
    Every *real* environment step triggers three activities:

    1. **Direct Q-update** — identical to a standard Q-learning update.
    2. **Model update** — the (s, a, r, s') tuple is fed into
       ``TabularEnvironmentModel``, which refines its MLE estimates.
    3. **Planning** — ``planning_steps`` simulated experiences are drawn from
       the model and used for additional Q-updates.
    """

    def __init__(
        self,
        actions: Iterable[int] = K_VALUES,
        learning_rate: float = 0.18,
        discount: float = 0.92,
        epsilon: float = 0.25,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.995,
        planning_steps: int = 20,
        seed: int | None = None,
    ) -> None:
        self.actions = tuple(actions)
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.planning_steps = planning_steps

        self.q_table: dict[tuple[int, ...], dict[int, float]] = {}
        self._rng = random.Random(seed)
        self._env_model = TabularEnvironmentModel(actions=tuple(actions), seed=seed)

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

    def update_policy(
        self,
        state: dict[str, Any],
        action: int,
        reward: float,
        next_state: dict[str, Any],
    ) -> None:
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

    def step(
        self,
        state: dict[str, Any],
        action: int,
        reward: float,
        next_state: dict[str, Any],
    ) -> int:
        """Process one real (s, a, r, s') transition."""
        # 1. Direct Q-learning update
        self.update_policy(state, action, reward, next_state)

        # 2. Feed real experience into the world model
        self._env_model.update(state, action, reward, next_state)

        # 3. Planning phase
        planning_done = 0
        for _ in range(self.planning_steps):
            experience = self._env_model.sample_random_experience()
            if experience is None:
                break  # model not populated yet
            sim_state_key, sim_action, sim_reward, sim_next_key = experience
            self._q_update_from_keys(sim_state_key, sim_action, sim_reward, sim_next_key)
            planning_done += 1

        return planning_done

    def train(self, env: Any, episodes: int = 250) -> list[DynaEpisodeSummary]:
        """Train with the Dyna-Q algorithm for a fixed number of episodes."""
        history: list[DynaEpisodeSummary] = []

        for episode in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0.0
            episode_actions: list[int] = []
            episode_planning_updates = 0
            steps = 0

            while True:
                action = self.choose_k(state)
                next_state, reward, done, _ = env.step(state, action)

                planning_done = self.step(state, action, reward, next_state)

                total_reward += reward
                episode_actions.append(action)
                episode_planning_updates += planning_done
                steps += 1
                state = next_state

                if done:
                    break

            history.append(
                DynaEpisodeSummary(
                    episode=episode,
                    total_reward=total_reward,
                    average_reward=total_reward / max(steps, 1),
                    average_k=sum(episode_actions) / max(len(episode_actions), 1),
                    epsilon=self.epsilon,
                    steps=steps,
                    planning_updates=episode_planning_updates,
                    model_coverage=self._env_model.num_observed_pairs,
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
                next_state, reward, done, info = env.step(state, action)

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
            "planning_steps": self.planning_steps,
            "q_table": {
                "|".join(str(part) for part in state_key): values
                for state_key, values in self.q_table.items()
            },
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "DynaQController":
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
            planning_steps=payload.get("planning_steps", 20),
        )
        controller.q_table = {
            tuple(int(part) for part in key.split("|")): {int(action): float(value) for action, value in values.items()}
            for key, values in payload["q_table"].items()
        }
        return controller

    @property
    def env_model(self):
        """Expose the underlying model."""
        return self._env_model

    def _q_update_from_keys(
        self,
        state_key: tuple[int, ...],
        action: int,
        reward: float,
        next_state_key: tuple[int, ...],
    ) -> None:
        self._ensure_state(state_key)
        self._ensure_state(next_state_key)

        current_q = self.q_table[state_key][action]
        next_q = max(self.q_table[next_state_key].values())
        td_target = reward + self.discount * next_q
        td_error = td_target - current_q
        self.q_table[state_key][action] = current_q + self.learning_rate * td_error

    def _ensure_state(self, state_key: tuple[int, ...]) -> None:
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.actions}
