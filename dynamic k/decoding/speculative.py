"""Tree-based speculative decoding using Hugging Face models.

This module implements a research-oriented variant of SpecInfer-style greedy
tree verification on top of the standard `transformers` stack. The verifier
packs the whole speculative tree into one target-model forward pass by using a
topology-aware attention mask and GPT-2 absolute position remapping.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


@dataclass
class TreeNode:
    """Node inside the speculative draft tree."""

    node_id: int
    token_id: int | None
    depth: int
    parent_id: int | None = None
    path_token_ids: tuple[int, ...] = ()
    path_node_ids: tuple[int, ...] = ()
    draft_logprob: float = 0.0
    children: list["TreeNode"] = field(default_factory=list)


@dataclass(frozen=True)
class PackedTreeInputs:
    """Packed representation used by the single-pass tree verifier."""

    packed_input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    flat_nodes: tuple[TreeNode, ...]
    output_positions: dict[int, int]
    prompt_length: int


@dataclass(frozen=True)
class TreeExpansionPlan:
    """Planner output for the anisotropic speculative tree."""

    depth_widths: tuple[int, ...]
    node_budget: int


@dataclass
class DraftFrontierEntry:
    """One expandable node plus its cached draft-model continuation."""

    node: TreeNode
    batch_index: int
    next_log_probs: torch.Tensor


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
        self._ensure_supported_architecture(self.draft_model, draft_model_name)
        self._ensure_supported_architecture(self.target_model, target_model_name)
        self.draft_model.config._attn_implementation = "eager"
        self.target_model.config._attn_implementation = "eager"
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

        The draft model builds a fixed-width token tree up to depth `k`. The
        target model then verifies the whole tree in a single forward pass by
        following SpecInfer's greedy `VerifyGreedy` procedure over the returned
        node-wise outputs.
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
        root, tree_node_count, draft_forward_passes, plan = self._build_tree(
            input_ids,
            k=k,
            branching_factor=branch,
            state=state,
        )
        draft_time_s = time.perf_counter() - draft_start

        verify_start = time.perf_counter()
        verified = self._verify_tree(input_ids, root)
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
            "terminal_token_id": verified["terminal_token_id"],
            "emitted_token_ids": verified["emitted_token_ids"],
            "emitted_tokens": verified["emitted_tokens"],
            "tree_branching_factor": branch,
            "tree_nodes": tree_node_count,
            "normalized_tree_cost": normalized_tree_cost,
            "draft_time_s": draft_time_s,
            "verify_time_s": verify_time_s,
            "step_time_s": step_time_s,
            "target_logprob_mean": verified["target_logprob_mean"],
            "verification_mode": "specinfer_greedy",
            "draft_forward_passes": draft_forward_passes,
            "target_forward_passes": 1,
            "dfs_tree_nodes": verified["dfs_tree_nodes"],
            "verified_nodes": verified["verified_nodes"],
            "planned_node_budget": plan.node_budget,
            "depth_widths": list(plan.depth_widths),
        }
        return verified["new_sequence"], verified["accepted_tokens"], extra_info

    def _build_tree(
        self,
        input_ids: torch.Tensor,
        k: int,
        branching_factor: int,
        state: dict[str, Any] | None = None,
    ) -> tuple[TreeNode, int, int, TreeExpansionPlan]:
        """Expand a draft tree using batched anisotropic expansion."""
        root = TreeNode(
            node_id=0,
            token_id=None,
            depth=0,
            parent_id=None,
            path_token_ids=(),
            path_node_ids=(),
        )
        plan = self._plan_tree_shape(k=k, branching_factor=branching_factor, state=state)
        root_log_probs, current_cache = self._draft_prefill(input_ids)
        frontier = [DraftFrontierEntry(node=root, batch_index=0, next_log_probs=root_log_probs[0])]
        total_nodes = 0
        next_node_id = 1
        draft_forward_passes = 1

        for depth, width in enumerate(plan.depth_widths, start=1):
            if not frontier or total_nodes >= plan.node_budget or width <= 0:
                break

            next_frontier_nodes: list[TreeNode] = []
            parent_indices: list[int] = []
            child_token_ids: list[int] = []
            frontier.sort(
                key=lambda entry: float(torch.max(entry.next_log_probs).item()) + entry.node.draft_logprob,
                reverse=True,
            )

            for entry in frontier:
                remaining_budget = plan.node_budget - total_nodes - len(next_frontier_nodes)
                if remaining_budget <= 0:
                    break

                local_width = min(width, remaining_budget, int(entry.next_log_probs.shape[-1]))
                top_log_probs, top_token_ids = torch.topk(
                    entry.next_log_probs,
                    k=local_width,
                    dim=-1,
                )

                for branch_idx in range(local_width):
                    token_id = int(top_token_ids[branch_idx].item())
                    child = TreeNode(
                        node_id=next_node_id,
                        token_id=token_id,
                        depth=depth,
                        parent_id=entry.node.node_id,
                        path_token_ids=entry.node.path_token_ids + (token_id,),
                        path_node_ids=entry.node.path_node_ids + (next_node_id,),
                        draft_logprob=entry.node.draft_logprob + float(top_log_probs[branch_idx].item()),
                    )
                    entry.node.children.append(child)
                    next_frontier_nodes.append(child)
                    parent_indices.append(entry.batch_index)
                    child_token_ids.append(token_id)
                    next_node_id += 1
            total_nodes += len(next_frontier_nodes)

            if depth == k or not next_frontier_nodes:
                break

            current_cache.batch_select_indices(
                torch.tensor(parent_indices, device=self.device, dtype=torch.long)
            )
            batch_token_ids = torch.tensor(
                child_token_ids,
                device=self.device,
                dtype=input_ids.dtype,
            ).unsqueeze(1)
            outputs = self.draft_model(
                input_ids=batch_token_ids,
                past_key_values=current_cache,
                use_cache=True,
            )
            draft_forward_passes += 1
            current_cache = outputs.past_key_values
            next_log_probs_batch = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
            frontier = [
                DraftFrontierEntry(node=child, batch_index=batch_index, next_log_probs=next_log_probs_batch[batch_index])
                for batch_index, child in enumerate(next_frontier_nodes)
            ]

        return root, total_nodes, draft_forward_passes, plan

    def _verify_tree(
        self,
        input_ids: torch.Tensor,
        root: TreeNode,
    ) -> dict[str, Any]:
        """Verify the tree using a single packed target-model pass."""
        packed = self._flatten_tree_for_verification(input_ids, root)
        log_probs = self._tree_parallel_decode(
            packed_input_ids=packed.packed_input_ids,
            position_ids=packed.position_ids,
            attention_mask=packed.attention_mask,
        )
        return self._verify_greedy_outputs(
            input_ids=input_ids,
            root=root,
            packed=packed,
            log_probs=log_probs,
        )

    def _flatten_tree_for_verification(
        self,
        input_ids: torch.Tensor,
        root: TreeNode,
    ) -> PackedTreeInputs:
        """Flatten the speculative tree into one packed decoding pass."""
        flat_nodes: list[TreeNode] = []

        def dfs(node: TreeNode) -> None:
            for child in node.children:
                flat_nodes.append(child)
                dfs(child)

        dfs(root)
        prompt_length = int(input_ids.shape[1])
        token_ids = [int(node.token_id) for node in flat_nodes]
        packed_input_ids = self._append_tokens(input_ids, tuple(token_ids))

        logical_positions = list(range(prompt_length))
        logical_positions.extend(prompt_length + node.depth - 1 for node in flat_nodes)
        position_ids = torch.tensor([logical_positions], device=self.device, dtype=torch.long)

        total_length = prompt_length + len(flat_nodes)
        mask = self._make_topology_attention_mask(
            total_length=total_length,
            prompt_length=prompt_length,
            flat_nodes=tuple(flat_nodes),
        )
        output_positions = {node.node_id: prompt_length + index for index, node in enumerate(flat_nodes)}

        return PackedTreeInputs(
            packed_input_ids=packed_input_ids,
            position_ids=position_ids,
            attention_mask=mask,
            flat_nodes=tuple(flat_nodes),
            output_positions=output_positions,
            prompt_length=prompt_length,
        )

    def _make_topology_attention_mask(
        self,
        total_length: int,
        prompt_length: int,
        flat_nodes: tuple[TreeNode, ...],
    ) -> torch.Tensor:
        """Build the topology-aware additive mask used by tree verification."""
        min_value = torch.finfo(torch.float32).min
        mask = torch.full(
            (1, 1, total_length, total_length),
            fill_value=min_value,
            device=self.device,
            dtype=torch.float32,
        )

        for query_index in range(prompt_length):
            mask[0, 0, query_index, : query_index + 1] = 0.0

        node_positions = {node.node_id: prompt_length + index for index, node in enumerate(flat_nodes)}
        for index, node in enumerate(flat_nodes, start=prompt_length):
            mask[0, 0, index, :prompt_length] = 0.0
            for ancestor_id in node.path_node_ids:
                mask[0, 0, index, node_positions[ancestor_id]] = 0.0

        return mask

    @torch.inference_mode()
    def _tree_parallel_decode(
        self,
        packed_input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute all verifier outputs for the packed speculative tree."""
        outputs = self.target_model(
            input_ids=packed_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        return torch.log_softmax(outputs.logits[0], dim=-1)

    def _verify_greedy_outputs(
        self,
        input_ids: torch.Tensor,
        root: TreeNode,
        packed: PackedTreeInputs,
        log_probs: torch.Tensor,
    ) -> dict[str, Any]:
        """Apply SpecInfer's greedy verification procedure over packed outputs."""
        current_node = root
        accepted_token_ids: list[int] = []
        output_logprobs: list[float] = []
        rejection_position = -1
        fallback_token_id: int | None = None
        used_target_fallback = False
        verified_nodes = 0

        while True:
            if current_node.node_id == root.node_id:
                output_position = packed.prompt_length - 1
            else:
                output_position = packed.output_positions[current_node.node_id]

            node_log_probs = log_probs[output_position]
            output_token_id = int(torch.argmax(node_log_probs, dim=-1).item())
            output_logprobs.append(float(node_log_probs[output_token_id].item()))

            candidates = {child.token_id: child for child in current_node.children}
            if output_token_id in candidates:
                current_node = candidates[output_token_id]
                accepted_token_ids.append(output_token_id)
                verified_nodes += 1
                continue

            terminal_token_id = output_token_id
            if current_node.children:
                rejection_position = current_node.depth + 1
                fallback_token_id = terminal_token_id
                used_target_fallback = True
            break

        emitted_token_ids = accepted_token_ids + [terminal_token_id]
        new_sequence = self._append_tokens(input_ids, tuple(emitted_token_ids))

        return {
            "new_sequence": new_sequence,
            "accepted_tokens": len(accepted_token_ids),
            "accepted_token_ids": accepted_token_ids,
            "rejection_position": rejection_position,
            "fallback_token_id": fallback_token_id,
            "used_target_fallback": used_target_fallback,
            "terminal_token_id": terminal_token_id,
            "emitted_token_ids": emitted_token_ids,
            "emitted_tokens": len(emitted_token_ids),
            "target_logprob_mean": sum(output_logprobs) / len(output_logprobs),
            "dfs_tree_nodes": len(packed.flat_nodes),
            "verified_nodes": verified_nodes,
        }

    def _plan_tree_shape(
        self,
        k: int,
        branching_factor: int,
        state: dict[str, Any] | None,
    ) -> TreeExpansionPlan:
        """Choose an anisotropic tree shape that is cheap enough to win in practice."""
        ema_acceptance = float(state.get("ema_acceptance_rate", 0.55)) if state is not None else 0.55
        system_load = float(state.get("system_load", 0.30)) if state is not None else 0.30
        queue_length = int(state.get("queue_length", 0)) if state is not None else 0
        prompt_length = int(state.get("prompt_length", 64)) if state is not None else 64

        widths: list[int] = []
        root_width = min(branching_factor, 2 if prompt_length < 256 else 1)
        second_width = (
            min(branching_factor, 2)
            if ema_acceptance >= 0.60 and system_load <= 0.55 and queue_length <= 4
            else 1
        )
        third_width = (
            min(branching_factor, 2)
            if ema_acceptance >= 0.78 and system_load <= 0.35 and queue_length <= 2
            else 1
        )

        for depth in range(1, k + 1):
            if depth == 1:
                widths.append(max(1, root_width))
            elif depth == 2:
                widths.append(max(1, second_width))
            elif depth == 3:
                widths.append(max(1, third_width))
            else:
                widths.append(1)

        estimated_nodes = self._estimate_tree_nodes(widths)
        if system_load >= 0.70 or queue_length >= 6:
            node_budget = min(estimated_nodes, max(k + 1, 2 * k))
        elif ema_acceptance < 0.40:
            node_budget = min(estimated_nodes, max(k + branching_factor, 2 * k))
        else:
            node_budget = estimated_nodes

        return TreeExpansionPlan(depth_widths=tuple(widths), node_budget=max(1, node_budget))

    def _estimate_tree_nodes(self, depth_widths: list[int]) -> int:
        """Estimate total expanded nodes for a level-wise width schedule."""
        frontier_size = 1
        total_nodes = 0
        for width in depth_widths:
            frontier_size *= max(1, width)
            total_nodes += frontier_size
        return total_nodes

    @torch.inference_mode()
    def _draft_prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, DynamicCache]:
        """Run one prompt prefill for the draft model and return root logits plus cache."""
        cache = DynamicCache(config=self.draft_model.config)
        outputs = self.draft_model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
        )
        log_probs = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
        return log_probs, outputs.past_key_values

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

    def _ensure_supported_architecture(self, model: AutoModelForCausalLM, model_name: str) -> None:
        model_type = getattr(model.config, "model_type", None)
        if model_type != "gpt2":
            raise NotImplementedError(
                "SpecInfer-style packed verification currently supports GPT-2-family models only; "
                f"got {model_name!r} with model_type={model_type!r}."
            )
