# SpecInfer – Minimal Tree-based Speculative Decoding

A clean, hackable PyTorch implementation of the core ideas from:  
**SpecInfer: Accelerating LLM Serving with Tree-based Speculative Inference and Verification** (ASPLOS '24)

---

## Files

| File | Role |
|---|---|
| `tree.py` | `TreeNode` + `TokenTree` – the data structure |
| `draft.py` | `DraftModel` – builds speculative token trees |
| `target.py` | `TargetModel` – verifies trees against the LLM |
| `decoding.py` | Acceptance logic + `generate()` loop |
| `main.py` | Runnable example with GPT-2 |

---

## Quick Start

```bash
pip install torch transformers accelerate
python main.py
```

---

## How It Works

### 1. Tree Expansion (draft.py)

The draft model builds a **k-ary tree of depth D** level by level.

```
Virtual root (depth 0)
├── T_a  (depth 1, P_draft = 0.6)
│   ├── T_c  (depth 2)
│   └── T_d  (depth 2)
└── T_b  (depth 1, P_draft = 0.4)
    ├── T_e  (depth 2)
    └── T_f  (depth 2)
```

At each depth level all parent contexts are **batched into one forward pass**
(one padded batch per level, not one pass per node).

### 2. Tree Verification (target.py)

Two backends are provided:

**`batched` (default)**  
Packs all N+1 sequences (one per node) into a single padded batch.  
Works with any HuggingFace decoder model including GPT-2.

**`tree_parallel` (paper's contribution)**  
A **single** forward pass over `[prompt | BFS_tree_tokens]` using a
topology-aware causal mask:

```
Prompt tokens  → standard lower-triangular causal mask
Draft token i  → attends to ALL prompt tokens
                + attends to draft token j  iff  j is ancestor-or-self of i
                  (sibling branches are masked with -inf)
```

Position IDs: all draft nodes at depth d share position `P + d − 1`.
This makes rotary / absolute embeddings correct across all branches.

Compatible with: LlamaForCausalLM (`attn_implementation="eager"`), Mistral.  
Not compatible with: GPT-2 (it reshapes the attention_mask internally).

### 3. Acceptance Logic (decoding.py)

**Greedy (`greedy_accept`)**  
Walk the tree: accept child `v` iff `v.token_id == argmax P_target`.  
Always append one target token at the end.

**Stochastic MSS (`stochastic_accept`)**  
At each node, try children in random order.  
Accept child `s` (token `x`) with probability `min(1, P_target(x) / P_draft(x))`.  
On rejection, update residual:

```python
residual[x] = max(0, residual[x] - P_draft[x])
```

If all children are rejected, sample from the normalized residual.  
Always append one extra token (at a leaf OR from the residual).

**Theorem 4.2 (SpecInfer):** MSS is *distributionally equivalent* to  
sampling directly from the target model (zero quality loss).

---

## Extending This Code

This codebase is intentionally minimal and easy to modify:

```python
# Dynamic tree shapes
# Change build_tree to vary branching_factor per depth level:
tree = draft_model.build_tree(prompt_ids, branching_factor=[3,2,2], depth=3)

# Learned acceptance predictor
# Replace the min(1, p_t/p_d) criterion with a trained model:
alpha = acceptance_predictor(node_embedding, target_probs)

# Adaptive depth
# Stop expanding when draft confidence drops below a threshold:
if node.draft_prob < 0.1:
    continue  # don't expand this node
```

---

## Expected Output

```
[SpecInfer stats]
  iterations      : 18
  tokens accepted : 86
  tokens drafted  : 90
  acceptance rate : 95.6%
  avg tokens/iter : 4.78   (baseline = 1.00 for standard decoding)
```

Wall-clock speedup is most visible on **GPU** with model offloading.  
On CPU the draft model overhead can dominate; the real gain is in  
reducing target-model forward passes.
