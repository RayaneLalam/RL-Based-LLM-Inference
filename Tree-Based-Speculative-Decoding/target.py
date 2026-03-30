"""
target.py  –  TargetModel: verifies a speculative token tree against the LLM.

Two verification backends
--------------------------

1. "batched"  (default, works with ANY HuggingFace decoder model)
   ─────────────────────────────────────────────────────────────
   Build N+1 sequences (one per node, including virtual root) and run a
   single batched forward pass.  No redundant KV computation is shared
   across branches, but it works with GPT-2, Falcon, Phi, LLaMA, etc.

2. "tree_parallel"  (Section 4 of the SpecInfer paper)
   ─────────────────────────────────────────────────────────────
   ONE forward pass over the full [prompt | BFS_tree_tokens] sequence using
   a topology-aware causal mask.  Each draft token attends to the prompt and
   to its own ancestors only – NOT to sibling branches.

   This is the paper's core contribution: shared KV for common prefixes,
   zero redundant attention computation.

   Compatibility: requires the model to use our 4-D additive attention mask
   verbatim.  Tested with LlamaForCausalLM (attn_implementation="eager") and
   MistralForCausalLM.  NOT compatible with GPT-2 (GPT2Model reshapes the
   attention_mask and ignores custom 4-D inputs).

Return value (both modes)
--------------------------
A dict  id(node) → prob_vector  (vocab_size,)

The prob at id(node) is the target model's distribution over the token that
should come AFTER node in the sequence, i.e.:
    P(next | prompt + draft_token_sequence(node),  target_model)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

from transformers import PreTrainedModel
from tree import TreeNode, TokenTree


class TargetModel:
    """
    Thin wrapper around a large HuggingFace causal LM for tree verification.

    Parameters
    ----------
    model : any AutoModelForCausalLM-compatible model
    """

    def __init__(self, model: PreTrainedModel) -> None:
        self.model = model.eval()
        self.device = next(model.parameters()).device

    # ================================================================ public

    @torch.no_grad()
    def verify_tree(
        self,
        prompt_ids: torch.Tensor,   # (1, P)
        tree: TokenTree,
        mode: str = "batched",      # "batched" | "tree_parallel"
    ) -> Dict[int, torch.Tensor]:
        """
        Compute P(next | context) for every node in the tree.

        The virtual root's entry gives P(next | prompt).
        A draft node's entry gives P(next | prompt + path_to_node).
        """
        if mode == "batched":
            return self._verify_batched(prompt_ids, tree)
        if mode == "tree_parallel":
            return self._verify_tree_parallel(prompt_ids, tree)
        raise ValueError(f"Unknown mode {mode!r}. Choose 'batched' or 'tree_parallel'.")

    # ================================================================ backend 1: batched sequences

    @torch.no_grad()
    def _verify_batched(
        self,
        prompt_ids: torch.Tensor,
        tree: TokenTree,
    ) -> Dict[int, torch.Tensor]:
        """
        Batch all (N+1) sequences into ONE padded forward pass.

        Each sequence is:   prompt + draft_token_sequence(node)
        For the virtual root that reduces to just the prompt.
        """
        nodes = tree.bfs_nodes()
        all_nodes = [tree.root] + nodes   # virtual root first

        seqs: List[torch.Tensor] = []
        last_positions: List[int] = []

        for node in all_nodes:
            draft = node.draft_token_sequence()
            if draft:
                extra = torch.tensor(
                    draft, dtype=torch.long, device=self.device
                ).unsqueeze(0)
                seq = torch.cat([prompt_ids, extra], dim=1)[0]
            else:
                seq = prompt_ids[0]
            seqs.append(seq)
            last_positions.append(seq.size(0) - 1)

        # Right-pad into a batch
        max_len = max(s.size(0) for s in seqs)
        B = len(seqs)
        batch_ids = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
        attn_mask = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
        for i, seq in enumerate(seqs):
            L = seq.size(0)
            batch_ids[i, :L] = seq
            attn_mask[i, :L] = 1

        outputs = self.model(input_ids=batch_ids, attention_mask=attn_mask)
        # logits: (B, max_len, vocab)

        result: Dict[int, torch.Tensor] = {}
        for i, (node, last_pos) in enumerate(zip(all_nodes, last_positions)):
            result[id(node)] = F.softmax(outputs.logits[i, last_pos], dim=-1)
        return result

    # ================================================================ backend 2: tree-parallel (single forward pass)

    @torch.no_grad()
    def _verify_tree_parallel(
        self,
        prompt_ids: torch.Tensor,
        tree: TokenTree,
    ) -> Dict[int, torch.Tensor]:
        """
        SpecInfer's single-pass tree verification (Section 4).

        Input to the LLM
        ─────────────────
            [ prompt_tokens (P) | BFS_ordered_draft_tokens (N) ]

        Attention mask (topology-aware causal)
        ──────────────────────────────────────
        Prompt tokens   → standard lower-triangular causal mask.
        Draft token i   → attends to ALL prompt positions AND to draft
                          position j iff node_j is an ancestor-or-self
                          of node_i.  Sibling branches are masked out.

        Position IDs
        ─────────────
        All draft nodes at depth d share position ID  P + d - 1.
        This ensures rotary / absolute positional embeddings are correct
        for every branch, regardless of BFS ordering in the flat sequence.

        Output
        ──────
        logit at prompt position P-1  →  P(next | prompt)              [virtual root]
        logit at position P+i         →  P(next | path to draft_node_i)
        """
        nodes = tree.bfs_nodes()
        if not nodes:
            # Empty tree – just return the prompt distribution
            out = self.model(input_ids=prompt_ids)
            return {id(tree.root): F.softmax(out.logits[0, -1], dim=-1)}

        P = prompt_ids.size(1)
        N = len(nodes)
        node_to_idx: Dict[int, int] = {id(n): i for i, n in enumerate(nodes)}

        # ---- input ids -------------------------------------------------------
        draft_ids = torch.tensor(
            [n.token_id for n in nodes],
            dtype=torch.long, device=self.device,
        ).unsqueeze(0)                                          # (1, N)
        full_ids = torch.cat([prompt_ids, draft_ids], dim=1)   # (1, P+N)

        # ---- topology-aware causal mask --------------------------------------
        attn_mask = self._build_tree_mask(P, nodes, node_to_idx)   # (1,1,P+N,P+N)

        # ---- position ids (critical for RoPE / absolute PE) -----------------
        pos_ids = self._build_position_ids(P, nodes)               # (1, P+N)

        # ---- SINGLE forward pass over the whole tree -------------------------
        outputs = self.model(
            input_ids=full_ids,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            use_cache=False,
        )
        # logits: (1, P+N, vocab)

        # ---- extract per-node distributions ----------------------------------
        result: Dict[int, torch.Tensor] = {}
        # Virtual root: logit at last prompt position
        result[id(tree.root)] = F.softmax(outputs.logits[0, P - 1], dim=-1)
        # Each draft node: logit at its BFS position
        for i, node in enumerate(nodes):
            result[id(node)] = F.softmax(outputs.logits[0, P + i], dim=-1)

        return result

    # ================================================================ helpers

    def _build_tree_mask(
        self,
        P: int,
        nodes: List[TreeNode],
        node_to_idx: Dict[int, int],
    ) -> torch.Tensor:
        """
        Build a (1, 1, P+N, P+N) additive attention mask.

        Convention (matches HuggingFace eager attention):
            0.0   → position is visible (attend)
           -inf   → position is masked  (ignore)

        This mask is passed directly to the model without further
        modification when using attn_implementation="eager".
        """
        N = len(nodes)
        total = P + N
        # Start with everything masked
        mask = torch.full((total, total), float("-inf"), device=self.device)

        # ── Prompt: standard lower-triangular causal mask ──────────────────
        for i in range(P):
            mask[i, : i + 1] = 0.0

        # ── Draft nodes: topology-aware ─────────────────────────────────────
        for i, node in enumerate(nodes):
            row = P + i
            # Every draft token can see the entire prompt
            mask[row, :P] = 0.0
            # Each draft token can see itself and all its ancestors
            curr: Optional[TreeNode] = node
            while curr is not None and not curr.is_virtual_root():
                j = node_to_idx[id(curr)]
                mask[row, P + j] = 0.0
                curr = curr.parent  # walk up toward virtual root

        return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, P+N, P+N)

    def _build_position_ids(
        self,
        P: int,
        nodes: List[TreeNode],
    ) -> torch.Tensor:
        """
        Position IDs for [prompt | tree] sequence.

        Prompt      : 0, 1, …, P-1  (standard)
        Draft depth d: P + d - 1

        All nodes at the same tree depth share a position ID because they
        represent the same sequence position (just different branch choices).
        This keeps RoPE / absolute positional embeddings consistent.
        """
        prompt_pos = torch.arange(P, device=self.device)
        tree_pos = torch.tensor(
            [P + node.depth - 1 for node in nodes],
            dtype=torch.long, device=self.device,
        )
        return torch.cat([prompt_pos, tree_pos]).unsqueeze(0)   # (1, P+N)
