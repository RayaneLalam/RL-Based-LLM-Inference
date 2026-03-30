"""
draft.py  –  DraftModel: builds a speculative token tree from a small LM.

Strategy: expansion-based, static tree (Section 3 of the SpecInfer paper).
---------------------------------------------------------------------------
• Process tree levels top-down (breadth-first).
• At each level, batch ALL parent contexts into ONE forward pass.
• For every parent take the top-k (= branching_factor) tokens as children.

This yields a complete k-ary tree of depth D with
    k¹ + k² + … + kᴰ   draft nodes total.

The draft model runs autoregressively for each node, but the batching
across siblings at the same depth means only `depth` forward passes are
needed (one per level), not one per node.
"""

import torch
import torch.nn.functional as F
from typing import List

from transformers import PreTrainedModel
from tree import TreeNode, TokenTree


class DraftModel:
    """
    Thin wrapper around a small HuggingFace causal LM.

    Parameters
    ----------
    model : any AutoModelForCausalLM-compatible model
    """

    def __init__(self, model: PreTrainedModel) -> None:
        self.model = model.eval()
        self.device = next(model.parameters()).device

    # ================================================================ public

    @torch.no_grad()
    def build_tree(
        self,
        prompt_ids: torch.Tensor,   # (1, P)
        branching_factor: int = 2,
        depth: int = 3,
    ) -> TokenTree:
        """
        Build a speculative token tree.

        Algorithm
        ---------
        1. Start with the virtual root as the only "parent" at level 0.
        2. For each depth level 1 … D:
           a. Collect contexts for every current parent
              (context = prompt + draft_token_sequence of parent).
           b. One batched forward pass → next-token probs for every parent.
           c. Take top-k tokens per parent → attach as children.
        3. Return the completed tree.

        Parameters
        ----------
        prompt_ids       : tokenised prompt, shape (1, P)
        branching_factor : k – how many children to add per node
        depth            : D – how many draft levels to grow

        Returns
        -------
        TokenTree with branching_factor^d nodes at depth d.
        """
        tree = TokenTree()
        current_parents: List[TreeNode] = [tree.root]

        for _ in range(depth):
            # One batched forward pass covers the entire level
            probs_list = self._level_probs(prompt_ids, current_parents)

            next_parents: List[TreeNode] = []
            for parent, probs in zip(current_parents, probs_list):
                topk_p, topk_ids = torch.topk(probs, branching_factor)
                children = tree.expand(
                    parent,
                    topk_ids.tolist(),
                    topk_p.tolist(),
                )
                next_parents.extend(children)

            current_parents = next_parents

        return tree

    # ================================================================ private

    @torch.no_grad()
    def _level_probs(
        self,
        prompt_ids: torch.Tensor,   # (1, P)
        parents: List[TreeNode],
    ) -> List[torch.Tensor]:
        """
        Batched forward pass: compute next-token probability distributions
        for every node in `parents`.

        Context for node n:
            prompt_tokens  +  draft_token_sequence(n)
        For the virtual root:
            draft_token_sequence = []  →  context = prompt only.

        Sequences of different lengths are right-padded with 0; the
        attention mask ensures padding positions are ignored.
        """
        # ---- build one sequence per parent --------------------------------
        seqs: List[torch.Tensor] = []
        for node in parents:
            draft = node.draft_token_sequence()   # [] for virtual root
            if draft:
                extra = torch.tensor(
                    draft, dtype=torch.long, device=self.device
                ).unsqueeze(0)                           # (1, d)
                seq = torch.cat([prompt_ids, extra], dim=1)[0]  # (P+d,)
            else:
                seq = prompt_ids[0]                      # (P,)
            seqs.append(seq)

        # ---- pad to uniform length ----------------------------------------
        max_len = max(s.size(0) for s in seqs)
        B = len(seqs)
        batch_ids = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
        attn_mask = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
        for i, seq in enumerate(seqs):
            L = seq.size(0)
            batch_ids[i, :L] = seq
            attn_mask[i, :L] = 1

        # ---- single batched forward pass ----------------------------------
        outputs = self.model(input_ids=batch_ids, attention_mask=attn_mask)
        # logits: (B, max_len, vocab_size)

        # ---- extract probs at the last real position for each sequence ----
        result: List[torch.Tensor] = []
        for i, seq in enumerate(seqs):
            last = seq.size(0) - 1
            probs = F.softmax(outputs.logits[i, last], dim=-1)
            result.append(probs)

        return result
