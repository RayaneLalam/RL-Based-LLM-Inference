"""Tabular environment model learner for Sprint 2 (Model-Based RL).

This module implements a counting-based / Maximum Likelihood Estimation (MLE)
world model that learns transition dynamics P(s' | s, a) and a mean reward
function R(s, a) purely from empirical (s, a, r, s') tuples observed during
real environment interaction.

The learned model is used by the DynaQController (rl/controller.py) to
generate *simulated* experience for planning steps, dramatically improving
sample efficiency over pure model-free Q-learning (Sprint 1).

No part of this module has access to the formulas inside simulation/env.py.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from state.features import K_VALUES, featurize_state

# Type alias for the discrete featurized state key.
StateKey = tuple[int, ...]


class TabularEnvironmentModel:
    """MLE-based tabular model that learns P(s' | s, a) and R(s, a).

    Internals
    ---------
    For every (state_key, action) pair the model maintains:

    * ``_transition_counts[state_key][action][next_state_key]``  — how many
      times the agent arrived at *next_state_key* after taking *action* in
      *state_key*.
    * ``_reward_sum[state_key][action]``   — running sum of observed rewards.
    * ``_visit_count[state_key][action]``  — total number of observations for
      the (state_key, action) pair.

    Transition sampling draws a next state from the empirical categorical
    distribution over *next_state_key*.  The predicted reward is the running
    mean (maximum-likelihood estimate under a Gaussian noise model).

    Parameters
    ----------
    actions:
        Iterable of valid action values (defaults to ``K_VALUES``).
    seed:
        Optional random seed for reproducible planning rollouts.
    """

    def __init__(
        self,
        actions: tuple[int, ...] = K_VALUES,
        seed: int | None = None,
    ) -> None:
        self.actions = tuple(actions)
        self._rng = random.Random(seed)

        # transition_counts[(s, a)][s'] -> count
        self._transition_counts: dict[
            tuple[StateKey, int], dict[StateKey, int]
        ] = defaultdict(lambda: defaultdict(int))

        # reward_sum[(s, a)] -> float
        self._reward_sum: dict[tuple[StateKey, int], float] = defaultdict(float)

        # visit_count[(s, a)] -> int
        self._visit_count: dict[tuple[StateKey, int], int] = defaultdict(int)

        # Cache of all (state_key, action) pairs seen so far.
        # Maintained as a list for O(1) random sampling.
        self._observed_pairs: list[tuple[StateKey, int]] = []
        self._observed_pairs_set: set[tuple[StateKey, int]] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        state: dict[str, Any],
        action: int,
        reward: float,
        next_state: dict[str, Any],
    ) -> None:
        """Record one real (s, a, r, s') transition.

        Parameters
        ----------
        state:
            Raw environment state *before* the action.
        action:
            The action (k value) that was taken.
        reward:
            Scalar reward received after the action.
        next_state:
            Raw environment state *after* the action.
        """
        state_key = featurize_state(state)
        next_state_key = featurize_state(next_state)
        sa_pair = (state_key, action)

        self._transition_counts[sa_pair][next_state_key] += 1
        self._reward_sum[sa_pair] += reward
        self._visit_count[sa_pair] += 1

        if sa_pair not in self._observed_pairs_set:
            self._observed_pairs.append(sa_pair)
            self._observed_pairs_set.add(sa_pair)

    def sample(
        self,
        state_key: StateKey,
        action: int,
    ) -> tuple[StateKey, float] | None:
        """Generate a simulated (next_state_key, reward) from the learned model.

        Uses the empirical transition distribution P(s' | s, a) and the MLE
        mean reward R̂(s, a).

        Parameters
        ----------
        state_key:
            Discrete featurized state (output of ``featurize_state``).
        action:
            Action to simulate.

        Returns
        -------
        ``(next_state_key, reward)`` drawn from the learned distributions, or
        ``None`` if this (state_key, action) pair has never been observed.
        """
        sa_pair = (state_key, action)
        visit_count = self._visit_count[sa_pair]
        if visit_count == 0:
            return None

        # --- Sample next state from the empirical categorical distribution ---
        counts = self._transition_counts[sa_pair]
        population = list(counts.keys())
        weights = [counts[ns] for ns in population]
        (next_state_key,) = self._rng.choices(population, weights=weights, k=1)

        # --- MLE mean reward ---
        mean_reward = self._reward_sum[sa_pair] / visit_count

        return next_state_key, mean_reward

    def sample_random_experience(self) -> tuple[StateKey, int, float, StateKey] | None:
        """Return a randomly sampled simulated experience tuple.

        Uniformly samples a previously observed (s, a) pair from the replay
        buffer, then queries ``sample`` to produce (s, a, r̂, s').

        Returns
        -------
        ``(state_key, action, reward, next_state_key)`` or ``None`` if no
        transitions have been observed yet.
        """
        if not self._observed_pairs:
            return None

        state_key, action = self._rng.choice(self._observed_pairs)
        result = self.sample(state_key, action)
        if result is None:
            return None

        next_state_key, reward = result
        return state_key, action, reward, next_state_key

    # ------------------------------------------------------------------
    # Introspection helpers (for logging / debugging)
    # ------------------------------------------------------------------

    @property
    def num_observed_pairs(self) -> int:
        """Number of unique (state, action) pairs seen so far."""
        return len(self._observed_pairs)

    def mean_reward(self, state_key: StateKey, action: int) -> float | None:
        """Return the MLE mean reward for (state_key, action), or None."""
        sa_pair = (state_key, action)
        if self._visit_count[sa_pair] == 0:
            return None
        return self._reward_sum[sa_pair] / self._visit_count[sa_pair]

    def visit_count(self, state_key: StateKey, action: int) -> int:
        """Return the number of times (state_key, action) was observed."""
        return self._visit_count[(state_key, action)]
