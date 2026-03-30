"""
main.py  –  Minimal runnable example of tree-based speculative decoding.

Models used (auto-downloaded from HuggingFace on first run):
    draft  : gpt2         (~117 M params)
    target : gpt2-medium  (~345 M params)

Verification mode: "batched"
    Packs all N+1 sequences into one padded forward pass.
    Works with any HuggingFace decoder model.

To use the true single-pass tree-parallel backend (Section 4 of the paper),
switch to LLaMA-family models and set  verify_mode="tree_parallel":

    draft_name  = "JackFram/llama-68m"
    target_name = "meta-llama/Llama-2-7b-hf"
    ...
    _target_hf = AutoModelForCausalLM.from_pretrained(
        target_name,
        torch_dtype=torch.float16,
        attn_implementation="eager",   # ← required for 4-D mask support
    ).to(device)
    ...
    output_ids = generate(..., verify_mode="tree_parallel")

Requirements:
    pip install torch transformers accelerate
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from draft    import DraftModel
from target   import TargetModel
from decoding import generate


# ────────────────────────────────────────────────────────────── configuration
DRAFT_NAME   = "gpt2"          # small speculative model
TARGET_NAME  = "gpt2-medium"   # large target / verifier

BRANCHING    = 2     # k – children per tree node
DEPTH        = 4     # D – tree depth  →  2⁴-1 = 15 draft nodes per iteration
MAX_TOKENS   = 80
USE_GREEDY   = False  # False → stochastic MSS  |  True → greedy
VERIFY_MODE  = "batched"

PROMPT = (
    "The key insight behind speculative decoding is that a small draft model "
    "can predict multiple tokens ahead, and a large language model verifies "
    "all of them in a single forward pass. This works because"
)


# ────────────────────────────────────────────────────────────── helpers
def baseline_generate(model, tokenizer, prompt_ids, max_new_tokens, device):
    """Standard autoregressive decoding – one token per forward pass."""
    generated = prompt_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids=generated)
        if USE_GREEDY:
            next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        else:
            probs = torch.softmax(out.logits[0, -1], dim=-1)
            next_tok = torch.multinomial(probs, 1).unsqueeze(0)
        generated = torch.cat([generated, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break
    return generated


# ────────────────────────────────────────────────────────────── main
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── load models ──────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading draft  model : {DRAFT_NAME}")
    _draft_hf = AutoModelForCausalLM.from_pretrained(
        DRAFT_NAME, torch_dtype=torch.float32
    ).to(device)

    print(f"Loading target model : {TARGET_NAME}")
    _target_hf = AutoModelForCausalLM.from_pretrained(
        TARGET_NAME, torch_dtype=torch.float32
    ).to(device)

    draft_model  = DraftModel(_draft_hf)
    target_model = TargetModel(_target_hf)

    # ── tokenise prompt ───────────────────────────────────────────────────────
    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    print(f"\nPrompt ({prompt_ids.size(1)} tokens):\n  {PROMPT!r}\n")

    # ── speculative decoding ──────────────────────────────────────────────────
    print("─" * 60)
    print(f"Tree-based speculative decoding  "
          f"(branching={BRANCHING}, depth={DEPTH}, mode={VERIFY_MODE})")
    print("─" * 60)

    t0 = time.perf_counter()
    spec_ids = generate(
        prompt_ids      = prompt_ids,
        draft_model     = draft_model,
        target_model    = target_model,
        max_new_tokens  = MAX_TOKENS,
        branching_factor= BRANCHING,
        depth           = DEPTH,
        use_greedy      = USE_GREEDY,
        verify_mode     = VERIFY_MODE,
        eos_token_id    = tokenizer.eos_token_id,
        verbose         = True,
    )
    spec_time = time.perf_counter() - t0

    spec_text = tokenizer.decode(
        spec_ids[0, prompt_ids.size(1):], skip_special_tokens=True
    )
    print(f"Generated ({spec_time:.2f}s):\n  {spec_text!r}\n")

    # ── baseline (target model, no speculation) ───────────────────────────────
    print("─" * 60)
    print("Baseline: standard autoregressive decoding (target model only)")
    print("─" * 60)

    t0 = time.perf_counter()
    base_ids = baseline_generate(
        _target_hf, tokenizer, prompt_ids, MAX_TOKENS, device
    )
    base_time = time.perf_counter() - t0

    base_text = tokenizer.decode(
        base_ids[0, prompt_ids.size(1):], skip_special_tokens=True
    )
    print(f"Generated ({base_time:.2f}s):\n  {base_text!r}\n")

    # ── summary ───────────────────────────────────────────────────────────────
    if base_time > 0:
        speedup = base_time / spec_time
        print(f"Wall-clock speedup: {speedup:.2f}×")
        print("(Note: on CPU the draft-model overhead dominates; "
              "speedup is most visible on GPU with model offloading.)")


if __name__ == "__main__":
    main()
