"""Prompt loading helpers for LLM-backed Dyna-Q."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PromptRecord:
    """One prompt entry plus optional metadata."""

    prompt_id: str
    prompt: str
    metadata: dict[str, Any]


class PromptDataset:
    """Deterministic prompt sampler backed by a pandas DataFrame."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        seed: int = 0,
        max_prompts: int | None = None,
        sample_strategy: str = "sequential",
    ) -> None:
        self._df = dataframe.reset_index(drop=True)
        if max_prompts is not None:
            self._df = self._df.iloc[: max_prompts].reset_index(drop=True)
        self._size = len(self._df)
        if self._size == 0:
            raise ValueError("prompt dataset is empty after filtering")

        self._seed = int(seed)
        self._sample_strategy = sample_strategy
        self._cursor = 0
        self._order = list(range(self._size))
        if sample_strategy == "shuffle":
            rng = pd.Series(self._order).sample(frac=1.0, random_state=self._seed).tolist()
            self._order = list(rng)
        elif sample_strategy != "sequential":
            raise ValueError(f"unsupported sample_strategy={sample_strategy!r}")

    @classmethod
    def from_hf_json(
        cls,
        path: str,
        seed: int = 0,
        max_prompts: int | None = None,
        sample_strategy: str = "sequential",
    ) -> "PromptDataset":
        df = pd.read_json(path)
        return cls(df, seed=seed, max_prompts=max_prompts, sample_strategy=sample_strategy)

    @property
    def size(self) -> int:
        return self._size

    def sample_prompt(self) -> PromptRecord:
        index = self._order[self._cursor % self._size]
        self._cursor += 1
        row = self._df.iloc[index]
        prompt = self._extract_prompt(row)
        prompt_id = str(row.get("id", f"prompt_{index}"))
        metadata = self._extract_metadata(row)
        return PromptRecord(prompt_id=prompt_id, prompt=prompt, metadata=metadata)

    def _extract_prompt(self, row: pd.Series) -> str:
        for key in ("prompt", "text", "instruction", "question", "input"):
            if key in row and isinstance(row[key], str) and row[key].strip():
                if key == "instruction" and "input" in row and isinstance(row.get("input"), str):
                    combined = f"{row[key].strip()}\n\n{row.get('input').strip()}"
                    return combined.strip()
                return row[key].strip()

        for value in row.values:
            if isinstance(value, str) and value.strip():
                return value.strip()

        raise ValueError("no usable prompt text found in dataset row")

    def _extract_metadata(self, row: pd.Series) -> dict[str, Any]:
        metadata = {}
        for key in ("source", "domain", "category", "language"):
            if key in row:
                metadata[key] = row[key]
        return metadata
