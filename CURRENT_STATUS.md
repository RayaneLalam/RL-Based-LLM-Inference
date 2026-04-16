# Current Status

This document explains what is currently implemented in the `dynamic k`
prototype, what results we have measured so far, and what still needs to be
tested next.

## What Is Implemented

The current codebase includes a real tree-based speculative decoding path, not
just a tree data structure.

Implemented pieces:

- SpecInfer-style greedy tree verification in one packed target-model forward
  pass
- speculative tree construction on the draft model
- batched draft expansion with cache reuse instead of one draft forward per node
- anisotropic tree planning with a node budget
- tabular RL controller that chooses `k` online
- latency-aware reward for real-model online RL updates
- cached autoregressive baseline inside the benchmark runner
- environment-driven model selection so model pairs can be changed per machine
  without editing code
- unit tests plus lightweight real-model smoke tests

Current implementation limits:

- GPT-2-family models only
- greedy verification only
- no stochastic verifier path
- no custom serving kernel or fused runtime
- performance still depends heavily on draft/target compatibility

## Results We Have So Far

### 1. Same-Model M1 Smoke Benchmark

Model pair:

- draft: `sshleifer/tiny-gpt2`
- target: `sshleifer/tiny-gpt2`

Observed behavior:

- speculative acceptance was effectively perfect in that setup
- adaptive RL beat cached autoregressive throughput
- this confirmed the implementation works end to end on a lightweight machine

Interpretation:

- good smoke test
- not a realistic production-style claim because the draft and target are the
  same model

### 2. Two-Model M1 Benchmark

Model pair:

- draft: `sshleifer/tiny-gpt2`
- target: `distilgpt2`

Observed behavior:

- average speculative acceptance was `0.0`
- cached autoregressive stayed faster than every speculative policy
- adaptive RL was slower because it paid draft-tree overhead without getting
  accepted speculative tokens

Representative numbers:

- steady low load:
  - autoregressive: `24.84` tokens/s
  - adaptive RL: `15.21` tokens/s
- bursty high load:
  - autoregressive: `41.89` tokens/s
  - adaptive RL: `13.84` tokens/s

Interpretation:

- the implementation is functioning
- the RL controller is functioning
- this lightweight mismatched pair is not good enough for speculation to win

### 3. Earlier Larger Pair Benchmark

We also measured a larger pair earlier:

- draft: `distilgpt2`
- target: `gpt2`

That run improved substantially after the batching and cache-reuse changes, but
it still did not beat cached autoregressive in the measured setup.

Interpretation:

- the current implementation is no longer failing because of the old sequential
  verifier bug
- the remaining issue is real performance efficiency and acceptance on practical
  model pairs

## What This Means

The project has moved past the initial implementation problem. The tree method
is now real, RL is choosing `k` online, and the benchmark pipeline can compare
against autoregressive decoding.

The main open question is no longer whether the algorithm is wired correctly.
The main question is whether we can find a model pair and runtime setting where
speculation actually wins end to end.

## What We Need To Try Next

The next important step is testing with larger models.

Why:

- larger targets make each saved target forward pass more valuable
- tiny-model pairs are useful for smoke tests but not strong evidence for LLM
  inference optimization
- a very weak draft paired with a somewhat larger target may still have too low
  acceptance

Recommended next experiments:

- `distilgpt2 -> gpt2`
- a larger GPT-2-family target if the machine allows it
- machine-specific env profiles for laptop-safe, medium, and larger runs

Success condition for the next stage:

- a mismatched draft/target pair with nontrivial acceptance
- lower end-to-end latency or higher tokens/s than cached autoregressive
- reproducible results through the benchmark runner

## Where To Look

- run commands: [HOW_TO_RUN.md](/Users/youcefs/Desktop/school/RL/PROJECT/HOW_TO_RUN.md)
- project overview: [README.md](/Users/youcefs/Desktop/school/RL/PROJECT/README.md)
- codebase overview: [dynamic k/README.md](/Users/youcefs/Desktop/school/RL/PROJECT/dynamic%20k/README.md)
