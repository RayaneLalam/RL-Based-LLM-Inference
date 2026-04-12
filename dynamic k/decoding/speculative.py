"""Tree-based speculative decoding using Hugging Face models.

This module is intentionally research-oriented rather than production-oriented.
It uses `transformers` and `accelerate` to build a real decoding path suitable
for Sprint 1 experiments without relying on serving-specific frameworks.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer


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

    def __init__(
        self,
        draft_model_name: str,
        target_model_name: str,
    ) -> None:
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

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize a text prompt for inference."""
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return encoded["input_ids"].to(self.device)

    def decode_tokens(self, input_ids: torch.Tensor) -> str:
        """Convert model tokens back to text."""
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def speculative_decode_step(
        self,
        input_ids: torch.Tensor,
        k: int,
        state: dict[str, Any] | None = None,
        branching_factor: int | None = None,
    ) -> tuple[torch.Tensor, int, dict[str, Any]]:
        """Run one tree-based speculative decoding round.

        The draft model builds a token tree up to depth `k`.
        The target model then greedily verifies that tree and keeps the longest
        verified path. If verification fails before depth `k`, the target model
        still contributes one fallback token so generation can continue.
        """

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

    def _build_tree(self, input_ids: torch.Tensor, k: int, branching_factor: int) -> tuple[TreeNode, int]:
        """Expand a draft tree of candidate continuations."""
        root = TreeNode(token_id=None, depth=0, path_token_ids=())
        frontier = [root]
        total_nodes = 0

        for depth in range(1, k + 1):
            next_frontier: list[TreeNode] = []
            for node in frontier:
                draft_input = self._append_path(input_ids, node.path_token_ids)
                log_probs = self._next_token_log_probs(self.draft_model, draft_input)
                top_log_probs, top_token_ids = torch.topk(
                    log_probs,
                    k=branching_factor,
                    dim=-1,
                )

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

    def _verify_tree(
        self,
        input_ids: torch.Tensor,
        root: TreeNode,
        k: int,
        branching_factor: int,
    ) -> dict[str, Any]:
        """Verify the tree using greedy target-model decisions."""
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
            target_token_id = int(torch.argmax(log_probs, dim=-1).item())
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
            "target_logprob_mean": (
                sum(target_logprobs) / len(target_logprobs) if target_logprobs else float("nan")
            ),
            "tree_branching_factor": branching_factor,
        }

    @torch.inference_mode()
    def _next_token_log_probs(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
        return torch.log_softmax(logits, dim=-1)

    def _append_path(self, input_ids: torch.Tensor, path_token_ids: tuple[int, ...]) -> torch.Tensor:
        if not path_token_ids:
            return input_ids
        extension = torch.tensor([list(path_token_ids)], device=self.device, dtype=input_ids.dtype)
        return torch.cat([input_ids, extension], dim=1)

    def _append_tokens(self, input_ids: torch.Tensor, token_ids: tuple[int, ...]) -> torch.Tensor:
        if not token_ids:
            return input_ids
        extension = torch.tensor([list(token_ids)], device=self.device, dtype=input_ids.dtype)
        return torch.cat([input_ids, extension], dim=1)
