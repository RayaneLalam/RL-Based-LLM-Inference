# RL Project

This repository contains the Sprint 1 research material and the runnable
`dynamic k` prototype for RL-controlled speculative decoding.

## Top Level

- `SpecInfer.pdf`: reference paper used for the greedy tree verification path.
- `RL-Tasks-Sprint-1.pdf`: sprint task definition.
- `dynamic k/`: the actual Python project.

## What The Code Does

The `dynamic k` project trains a tabular RL controller that chooses speculative
tree depth `k` online, then runs a SpecInfer-style greedy tree verifier with
Hugging Face GPT-2-family models.

Use [HOW_TO_RUN.md](/Users/youcefs/Desktop/school/RL/PROJECT/HOW_TO_RUN.md) for
the current commands.
