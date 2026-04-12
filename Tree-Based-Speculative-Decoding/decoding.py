"""
decoding.py  –  Acceptance logic and the speculative decoding loop.

Two acceptance algorithms (both from Algorithm 2 / Section 4.3 of SpecInfer)
─────────────────────────────────────────────────────────────────────────────

greedy_accept
    Walk the tree greedily: accept child v iff v.token_id == argmax P_target.
    Always appends at least one new token directly from the target model.

stochastic_accept  (Multi-Step Speculative Sampling, "MSS")
    At each node, try children in random order.  Accept child s (token x_s)
    with probability  min(1, P_target(x_s) / P_draft(x_s)).
    On rejection update the residual target distribution:
        P_target(x) ← normalize(max(0, P_target(x) − P_draft(x)))
    If all children are rejected, sample from the residual distribution.
    Always appends at least one new token.

    Theorem 4.2 (SpecInfer): MSS is distributionally equivalent to sampling
    directly from the target model.  It maximises the expected number of
    accepted tokens while being strictly lossless.

generate
    Outer speculative decoding loop:
        while tokens_generated < max_new_tokens:
            1. draft_model.build_tree(prompt)
            2. target_model.verify_tree(tree)          ← 1 forward pass
            3. accept / reject with greedy or MSS
            4. append accepted tokens
"""

import time
import math
import torch
from typing import Dict, List, Optional, Tuple

from tree import TreeNode, TokenTree
from draft import DraftModel
from target import TargetModel


# ======================================================================== acceptance

def greedy_accept(
    tree: TokenTree,
    target_probs: Dict[int, torch.Tensor],
) -> List[int]:
    """
    Greedy acceptance (VerifyGreedy, Algorithm 2).

    Traverse the tree from the virtual root.  At each node u:
        target_token = argmax P_target(· | context at u)
        if any child has token_id == target_token  →  accept, descend
        else                                        →  append target_token, stop

    The final token in the returned list is always drawn from the target model
    (either the mismatched argmax, or the argmax at a leaf).
    """
    accepted: List[int] = []
    node: TreeNode = tree.root

    while True:
        probs = target_probs[id(node)]
        target_token = int(probs.argmax())

        # Search for a child that matches the target's greedy pick
        matched: Optional[TreeNode] = next(
            (c for c in node.children if c.token_id == target_token), None
        )

        # Always append the target token (draft hit or not)
        accepted.append(target_token)

        if matched is not None:
            node = matched          # draft was correct – go one level deeper
        else:
            break                   # draft diverged – stop and return

    return accepted


def stochastic_accept(
    tree: TokenTree,
    target_probs: Dict[int, torch.Tensor],
) -> List[int]:
    """
    Multi-step speculative sampling (VerifyStochastic / MSS, Algorithm 2).

    Traverse the tree from the virtual root.

    At each node u (with children H):
      • Shuffle H to avoid order bias.
      • For each candidate s ∈ H (token x_s):
            p_t = P_target(x_s | context at u)
            p_d = P_draft(x_s  | context at u)   ← stored in child.draft_prob
            alpha = min(1, p_t / p_d)
            if Uniform(0,1) ≤ alpha:   ACCEPT s, descend, stop inner loop
            else:                       REJECT  s, update residual
                residual[x_s] = max(0, residual[x_s] - p_d)
      • If H is exhausted (all rejected):  break outer loop

    After the outer loop (leaf reached OR all-rejection):
      • Sample one MORE token from the residual (if all-rejection) or from
        P_target at the leaf.  This guarantees at least one new token per call
        and ensures distributional equivalence with the target model.

    Distributional correctness (Theorem 4.2):
        P_SpecInfer(x | context) == P_target(x | context)   for all x, context
    """
    accepted: List[int] = []
    node: TreeNode = tree.root
    final_residual: Optional[torch.Tensor] = None

    # ── Traverse tree ────────────────────────────────────────────────────────
    while node.children:
        probs = target_probs[id(node)]

        # Shuffle children – random order is required for unbiased acceptance
        children = node.children[:]
        perm = torch.randperm(len(children)).tolist()
        children = [children[i] for i in perm]

        # residual starts as a copy of P_target; shrinks on each rejection
        residual = probs.clone()
        accepted_child: Optional[TreeNode] = None

        for child in children:
            x   = child.token_id
            p_t = probs[x].item()               # target probability for x
            p_d = child.draft_prob              # draft  probability for x

            # Rejection-sampling acceptance criterion
            alpha = min(1.0, p_t / (p_d + 1e-12))

            if torch.rand(1).item() <= alpha:
                # ── ACCEPT ──
                accepted.append(x)
                accepted_child = child
                break
            else:
                # ── REJECT: subtract draft contribution from residual ──
                # After this, residual represents the "leftover" probability
                # mass that the draft model didn't cover.
                residual[x] = max(0.0, residual[x].item() - p_d)

        if accepted_child is not None:
            node = accepted_child               # go one level deeper
        else:
            # All children rejected; save residual for final sampling
            final_residual = residual
            break

    # ── Always sample one extra token (Algorithm 2, lines 39–40) ─────────────
    # This token either:
    #   (a) comes from the residual distribution (some level had all-rejection)
    #   (b) comes from P_target at a leaf (all draft tokens accepted)
    # Either way, it guarantees progress and maintains distributional equivalence.
    if final_residual is not None:
        sp = torch.clamp(final_residual, min=0.0)
        total = sp.sum()
        sp = sp / total if total > 1e-12 else target_probs[id(node)]
    else:
        sp = target_probs[id(node)]

    # Numerical safety: clamp, replace any nan/inf, and re-normalise
    sp = torch.clamp(sp, min=0.0)
    sp = torch.nan_to_num(sp, nan=0.0, posinf=0.0, neginf=0.0)
    total = sp.sum()
    if total < 1e-12:
        # Fallback to uniform over vocab if distribution collapsed
        sp = torch.ones_like(sp)
    sp = sp / sp.sum()

    next_token = int(torch.multinomial(sp, num_samples=1))
    accepted.append(next_token)

    return accepted


# ======================================================================== decoding loop

@torch.no_grad()
def generate(
    prompt_ids: torch.Tensor,               # (1, P) on the correct device
    draft_model: DraftModel,
    target_model: TargetModel,
    max_new_tokens: int = 100,
    branching_factor: int = 2,              # k – children per node
    depth: int = 3,                         # D – tree depth
    use_greedy: bool = False,               # False → stochastic MSS sampling
    verify_mode: str = "batched",           # "batched" | "tree_parallel"
    eos_token_id: Optional[int] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Tree-based speculative decoding loop.

    One iteration
    ─────────────
    1. DraftModel.build_tree  → k-ary token tree (kᴰ leaf nodes)
    2. TargetModel.verify_tree→ target probs for all N nodes in ≤1 pass
    3. Accept tokens (greedy or MSS)
    4. Append accepted tokens to the running sequence
    5. Repeat from the new sequence

    Expected gain: each iteration accepts > 1 token on average, so the
    total number of target-model forward passes is reduced by the tree's
    acceptance rate.

    Parameters
    ----------
    prompt_ids       : tokenised prompt, shape (1, P)
    draft_model      : DraftModel wrapper
    target_model     : TargetModel wrapper
    max_new_tokens   : stop after this many new tokens
    branching_factor : k in the paper's ⟨k₁, k₂, …, kₘ⟩ config
    depth            : D – how many draft levels to speculate
    use_greedy       : True → greedy acceptance; False → MSS
    verify_mode      : "batched" (any model) or "tree_parallel" (LLaMA etc.)
    eos_token_id     : stop on this token if provided
    verbose          : print acceptance stats at the end
    """
    generated = prompt_ids.clone()
    total_generated = 0
    total_drafted   = 0
    total_accepted  = 0
    iterations      = 0

    while total_generated < max_new_tokens:
        # ── Step 1: build speculative token tree ─────────────────────────
        tree = draft_model.build_tree(
            prompt_ids=generated,
            branching_factor=branching_factor,
            depth=depth,
        )
        total_drafted += tree.size()

        # ── Step 2: verify entire tree with target model (1 forward pass) ─
        target_probs = target_model.verify_tree(
            prompt_ids=generated,
            tree=tree,
            mode=verify_mode,
        )

        # ── Step 3: accept / reject ───────────────────────────────────────
        if use_greedy:
            new_tokens = greedy_accept(tree, target_probs)
        else:
            new_tokens = stochastic_accept(tree, target_probs)

        total_accepted += len(new_tokens)
        iterations     += 1

        # ── Step 4: append accepted tokens ───────────────────────────────
        for tok in new_tokens:
            tok_t = torch.tensor(
                [[tok]], dtype=torch.long, device=generated.device
            )
            generated = torch.cat([generated, tok_t], dim=1)
            total_generated += 1

            if eos_token_id is not None and tok == eos_token_id:
                if verbose:
                    _print_stats(total_accepted, total_drafted, iterations)
                return generated

            if total_generated >= max_new_tokens:
                break

    if verbose:
        _print_stats(total_accepted, total_drafted, iterations)
    return generated


def _print_stats(accepted: int, drafted: int, iters: int) -> None:
    rate = accepted / max(drafted, 1)
    avg  = accepted / max(iters, 1)
    print(
        f"\n[SpecInfer stats]\n"
        f"  iterations      : {iters}\n"
        f"  tokens accepted : {accepted}\n"
        f"  tokens drafted  : {drafted}\n"
        f"  acceptance rate : {rate:.1%}\n"
        f"  avg tokens/iter : {avg:.2f}  "
        f"(baseline = 1.00 for standard decoding)\n"
    )
