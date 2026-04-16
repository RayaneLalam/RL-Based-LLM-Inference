# How To Run

These commands use the current project entrypoints inside `dynamic k/`.

## 1. Activate The Environment

```bash
cd "/Users/youcefs/Desktop/school/RL/PROJECT/dynamic k"
source .venv/bin/activate
```

## 2. Configure Models From Env

The project now resolves model pairs from environment variables so changing
machines does not require code edits.

Default lightweight two-model profile:

```bash
export PARETOSERVE_MODEL_PROFILE=m1_tiny_two_models
```

Explicit override:

```bash
export PARETOSERVE_DRAFT_MODEL=sshleifer/tiny-gpt2
export PARETOSERVE_TARGET_MODEL=distilgpt2
```

You can also source the example file:

```bash
set -a
source .env.example
set +a
```

## 3. Train The RL Controller

Steady workload:

```bash
python3 train_sprint1.py \
  --episodes 250 \
  --workload steady_low_load \
  --episode-length 32 \
  --checkpoint artifacts/q_table_steady.json \
  --seed 7
```

Bursty workload:

```bash
python3 train_sprint1.py \
  --episodes 250 \
  --workload bursty_high_load \
  --episode-length 32 \
  --checkpoint artifacts/q_table_bursty.json \
  --seed 7
```

## 4. Run Real Decoding

```bash
python3 run_real_sprint1.py \
  --prompt "Adaptive inference control" \
  --workload steady_low_load \
  --checkpoint artifacts/q_table_real.json \
  --branching-factor 2 \
  --max-new-tokens 12 \
  --reward-mode latency_aware \
  --seed 7
```

## 5. Run Benchmarks

```bash
python3 experiments/run_benchmarks.py \
  --output-dir artifacts/benchmarks \
  --train-episodes 120 \
  --sim-eval-episodes 40 \
  --episode-length 16 \
  --branching-factor 2 \
  --max-new-tokens 3 \
  --reward-mode latency_aware \
  --seed 7
```

## 6. Run Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Notes

- Real-model defaults come from `PARETOSERVE_*` environment variables.
- Real-model test defaults come from `SPECINFER_REAL_TEST_*` environment variables.
- The tree verifier only supports GPT-2-family models in this codebase.
