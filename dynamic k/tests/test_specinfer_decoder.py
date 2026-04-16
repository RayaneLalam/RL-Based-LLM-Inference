"""Regression tests for the SpecInfer-style greedy tree verifier."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_config import resolve_model_pair

MODULE_PATH = PROJECT_ROOT / "decoding" / "speculative.py"
MODULE_SPEC = importlib.util.spec_from_file_location("speculative_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)

HuggingFaceTreeSpeculativeDecoder = MODULE.HuggingFaceTreeSpeculativeDecoder
TreeNode = MODULE.TreeNode
DEFAULT_REAL_DRAFT_MODEL, DEFAULT_REAL_TARGET_MODEL, _ = resolve_model_pair(
    env_prefix="SPECINFER_REAL_TEST",
)
REAL_DRAFT_MODEL = os.environ.get("SPECINFER_REAL_TEST_DRAFT_MODEL", DEFAULT_REAL_DRAFT_MODEL)
REAL_TARGET_MODEL = os.environ.get("SPECINFER_REAL_TEST_TARGET_MODEL", DEFAULT_REAL_TARGET_MODEL)


class CountingLM(torch.nn.Module):
    """Wrap a GPT-2 LM and record forward-pass usage."""

    def __init__(self, model: GPT2LMHeadModel) -> None:
        super().__init__()
        self.model = model
        self.config = model.config
        self.calls = 0
        self.seen_shapes: list[tuple[int, ...]] = []

    def forward(self, *args, **kwargs):
        self.calls += 1
        input_ids = kwargs.get("input_ids")
        if input_ids is not None:
            self.seen_shapes.append(tuple(input_ids.shape))
        return self.model(*args, **kwargs)


def build_decoder(target_model: torch.nn.Module | None = None) -> HuggingFaceTreeSpeculativeDecoder:
    """Construct a decoder instance without loading remote models."""
    if target_model is None:
        config = GPT2Config(
            vocab_size=32,
            n_positions=32,
            n_ctx=32,
            n_embd=24,
            n_layer=2,
            n_head=2,
        )
        config._attn_implementation = "eager"
        target_model = GPT2LMHeadModel(config)
        target_model.eval()

    decoder = HuggingFaceTreeSpeculativeDecoder.__new__(HuggingFaceTreeSpeculativeDecoder)
    decoder.device = torch.device("cpu")
    decoder.target_model = target_model
    decoder.draft_model = target_model
    decoder.target_model_name = "unit-target"
    decoder.draft_model_name = "unit-draft"
    return decoder


def attach(parent: TreeNode, node_id: int, token_id: int) -> TreeNode:
    """Create one child node and attach it to the given parent."""
    child = TreeNode(
        node_id=node_id,
        token_id=token_id,
        depth=parent.depth + 1,
        parent_id=parent.node_id,
        path_token_ids=parent.path_token_ids + (token_id,),
        path_node_ids=parent.path_node_ids + (node_id,),
    )
    parent.children.append(child)
    return child


def build_manual_tree() -> tuple[TreeNode, TreeNode, TreeNode, TreeNode, TreeNode]:
    """Build a small hand-crafted tree with one branch and one sibling."""
    root = TreeNode(node_id=0, token_id=None, depth=0)
    left = attach(root, node_id=1, token_id=4)
    left_left = attach(left, node_id=2, token_id=5)
    left_right = attach(left, node_id=3, token_id=6)
    right = attach(root, node_id=4, token_id=7)
    return root, left, left_left, left_right, right


def make_log_probs(total_length: int, vocab_size: int, winners: dict[int, int]) -> torch.Tensor:
    """Create deterministic per-position log-probabilities."""
    log_probs = torch.full((total_length, vocab_size), fill_value=-100.0)
    for position, token_id in winners.items():
        log_probs[position, token_id] = 0.0
    return log_probs


class SpecInferDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.decoder = build_decoder()
        self.input_ids = torch.tensor([[10, 11]])
        self.root, self.left, self.left_left, self.left_right, self.right = build_manual_tree()

    def test_topology_mask_uses_prompt_and_ancestor_chain_only(self) -> None:
        packed = self.decoder._flatten_tree_for_verification(self.input_ids, self.root)

        self.assertEqual(packed.packed_input_ids.tolist(), [[10, 11, 4, 5, 6, 7]])
        self.assertEqual(packed.position_ids.tolist(), [[0, 1, 2, 3, 3, 2]])

        mask = packed.attention_mask[0, 0]

        def allowed(query_index: int) -> set[int]:
            return {index for index in range(mask.shape[1]) if mask[query_index, index].item() == 0.0}

        self.assertEqual(allowed(0), {0})
        self.assertEqual(allowed(1), {0, 1})
        self.assertEqual(allowed(packed.output_positions[self.left.node_id]), {0, 1, 2})
        self.assertEqual(allowed(packed.output_positions[self.left_left.node_id]), {0, 1, 2, 3})
        self.assertEqual(allowed(packed.output_positions[self.left_right.node_id]), {0, 1, 2, 4})
        self.assertEqual(allowed(packed.output_positions[self.right.node_id]), {0, 1, 5})

    def test_verify_greedy_root_mismatch_appends_verifier_token(self) -> None:
        packed = self.decoder._flatten_tree_for_verification(self.input_ids, self.root)
        log_probs = make_log_probs(total_length=6, vocab_size=32, winners={1: 9})

        result = self.decoder._verify_greedy_outputs(self.input_ids, self.root, packed, log_probs)

        self.assertEqual(result["accepted_tokens"], 0)
        self.assertEqual(result["emitted_token_ids"], [9])
        self.assertEqual(result["new_sequence"].tolist(), [[10, 11, 9]])
        self.assertEqual(result["rejection_position"], 1)
        self.assertTrue(result["used_target_fallback"])
        self.assertEqual(result["fallback_token_id"], 9)

    def test_verify_greedy_accepts_prefix_then_falls_back(self) -> None:
        packed = self.decoder._flatten_tree_for_verification(self.input_ids, self.root)
        log_probs = make_log_probs(
            total_length=6,
            vocab_size=32,
            winners={
                1: 4,
                packed.output_positions[self.left.node_id]: 8,
            },
        )

        result = self.decoder._verify_greedy_outputs(self.input_ids, self.root, packed, log_probs)

        self.assertEqual(result["accepted_tokens"], 1)
        self.assertEqual(result["accepted_token_ids"], [4])
        self.assertEqual(result["emitted_token_ids"], [4, 8])
        self.assertEqual(result["new_sequence"].tolist(), [[10, 11, 4, 8]])
        self.assertEqual(result["rejection_position"], 2)
        self.assertTrue(result["used_target_fallback"])
        self.assertEqual(result["fallback_token_id"], 8)

    def test_verify_greedy_accepts_leaf_and_appends_terminal_token(self) -> None:
        packed = self.decoder._flatten_tree_for_verification(self.input_ids, self.root)
        log_probs = make_log_probs(
            total_length=6,
            vocab_size=32,
            winners={
                1: 4,
                packed.output_positions[self.left.node_id]: 5,
                packed.output_positions[self.left_left.node_id]: 12,
            },
        )

        result = self.decoder._verify_greedy_outputs(self.input_ids, self.root, packed, log_probs)

        self.assertEqual(result["accepted_tokens"], 2)
        self.assertEqual(result["accepted_token_ids"], [4, 5])
        self.assertEqual(result["emitted_token_ids"], [4, 5, 12])
        self.assertEqual(result["new_sequence"].tolist(), [[10, 11, 4, 5, 12]])
        self.assertEqual(result["rejection_position"], -1)
        self.assertFalse(result["used_target_fallback"])
        self.assertIsNone(result["fallback_token_id"])

    def test_tree_parallel_decode_matches_individual_scoring(self) -> None:
        packed = self.decoder._flatten_tree_for_verification(self.input_ids, self.root)
        parallel_log_probs = self.decoder._tree_parallel_decode(
            packed_input_ids=packed.packed_input_ids,
            position_ids=packed.position_ids,
            attention_mask=packed.attention_mask,
        )

        def individual_log_probs(path_tokens: tuple[int, ...]) -> torch.Tensor:
            context = self.decoder._append_tokens(self.input_ids, path_tokens)
            outputs = self.decoder.target_model(input_ids=context, use_cache=False)
            return torch.log_softmax(outputs.logits[0, -1], dim=-1)

        torch.testing.assert_close(parallel_log_probs[1], individual_log_probs(()), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            parallel_log_probs[packed.output_positions[self.left.node_id]],
            individual_log_probs((4,)),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            parallel_log_probs[packed.output_positions[self.left_left.node_id]],
            individual_log_probs((4, 5)),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            parallel_log_probs[packed.output_positions[self.right.node_id]],
            individual_log_probs((7,)),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_verify_tree_uses_single_packed_target_forward(self) -> None:
        base_model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=32,
                n_positions=32,
                n_ctx=32,
                n_embd=24,
                n_layer=2,
                n_head=2,
            )
        )
        base_model.config._attn_implementation = "eager"
        base_model.eval()
        counting_model = CountingLM(base_model)
        decoder = build_decoder(counting_model)

        packed = decoder._flatten_tree_for_verification(self.input_ids, self.root)
        result = decoder._verify_tree(self.input_ids, self.root)

        self.assertEqual(counting_model.calls, 1)
        self.assertEqual(counting_model.seen_shapes, [(1, packed.packed_input_ids.shape[1])])
        self.assertEqual(result["dfs_tree_nodes"], len(packed.flat_nodes))
        self.assertGreaterEqual(result["emitted_tokens"], 1)

    def test_build_tree_batches_draft_expansion(self) -> None:
        root, tree_nodes, draft_forward_passes, plan = self.decoder._build_tree(
            self.input_ids,
            k=5,
            branching_factor=2,
            state={
                "ema_acceptance_rate": 0.85,
                "system_load": 0.20,
                "queue_length": 1,
                "prompt_length": int(self.input_ids.shape[1]),
            },
        )

        self.assertGreater(tree_nodes, 0)
        self.assertLess(draft_forward_passes, tree_nodes)
        self.assertEqual(plan.depth_widths, (2, 2, 2, 1, 1))
        self.assertLessEqual(tree_nodes, plan.node_budget)
        self.assertGreaterEqual(len(root.children), 1)


class SpecInferRealModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.decoder = HuggingFaceTreeSpeculativeDecoder(
                draft_model_name=REAL_DRAFT_MODEL,
                target_model_name=REAL_TARGET_MODEL,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"unable to load tiny real-model smoke test: {exc}") from exc

    def test_real_tiny_model_speculative_decode_step(self) -> None:
        input_ids = self.decoder.encode_prompt("Adaptive inference control")
        state = {
            "tree_branching_factor": 2,
            "ema_acceptance_rate": 0.75,
            "system_load": 0.25,
            "queue_length": 1,
            "prompt_length": int(input_ids.shape[1]),
        }

        new_input_ids, accepted_tokens, extra_info = self.decoder.speculative_decode_step(
            input_ids=input_ids,
            k=2,
            state=state,
            branching_factor=2,
        )

        self.assertGreaterEqual(int(new_input_ids.shape[1]), int(input_ids.shape[1]) + 1)
        self.assertGreaterEqual(accepted_tokens, 0)
        self.assertEqual(extra_info["verification_mode"], "specinfer_greedy")
        self.assertEqual(extra_info["target_forward_passes"], 1)
        self.assertEqual(extra_info["draft_model_name"], REAL_DRAFT_MODEL)
        self.assertEqual(extra_info["target_model_name"], REAL_TARGET_MODEL)
        self.assertGreaterEqual(extra_info["tree_nodes"], 1)
        self.assertGreaterEqual(extra_info["emitted_tokens"], 1)
        self.assertEqual(len(extra_info["depth_widths"]), 2)


if __name__ == "__main__":
    unittest.main()
