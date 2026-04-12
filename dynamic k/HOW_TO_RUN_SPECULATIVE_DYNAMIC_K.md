# How To Run `speculative devoding dynamic k.py`
ontains spaces, always wrap it in quotes in shell commands.

## 1. Prerequisites

- `python3.12` available on your machine
- internet access for the first Hugging Face model download
- enough disk space for the virtual environment and model cache

## 2. Create The Virtual Environment

From the project directory:

```bash
cd /Users/youcefs/Desktop/school/RL/PROJECT
python3.12 "speculative devoding dynamic k.py" bootstrap-venv
```

This creates `.venv` and installs:

- `torch`
- `transformers`
- `accelerate`
- `sentencepiece`
- `safetensors`
- `matplotlib`
- `pandas`
- `nbformat`

## 3. Activate The Environment

```bash
cd /Users/youcefs/Desktop/school/RL/PROJECT
. .venv/bin/activate
```

## 4. Train The RL Controller

Steady low load:

```bash
python "speculative devoding dynamic k.py" train \
  --episodes 250 \
  --eval-episodes 20 \
  --episode-length 32 \
  --workload steady_low_load \
  --checkpoint artifacts/q_table_steady.json \
  --seed 7
```

Bursty high load:

```bash
python "speculative devoding dynamic k.py" train \
  --episodes 250 \
  --eval-episodes 20 \
  --episode-length 32 \
  --workload bursty_high_load \
  --checkpoint artifacts/q_table_bursty.json \
  --seed 7
```

What this does:

- trains tabular Q-learning on the simulator
- prints final evaluation metrics
- saves the learned Q-table checkpoint into `artifacts/`

## 5. Run Real Decoding

Example:

```bash
python "speculative devoding dynamic k.py" decode \
  --prompt "Adaptive inference control" \
  --draft-model distilgpt2 \
  --target-model gpt2 \
  --max-new-tokens 12 \
  --workload steady_low_load \
  --checkpoint artifacts/q_table_real.json \
  --train-episodes 150 \
  --episode-length 32 \
  --branching-factor 2 \
  --seed 7
```

What this does:

- loads the RL controller from the checkpoint if it exists
- otherwise trains one first
- loads the Hugging Face draft and target models
- runs tree-based speculative decoding with dynamic `k`
- prints rollout statistics and generated text

## 6. Run The Full Benchmark Suite

```bash
python "speculative devoding dynamic k.py" benchmark \
  --output-dir artifacts/benchmarks \
  --train-episodes 100 \
  --sim-eval-episodes 30 \
  --episode-length 16 \
  --draft-model distilgpt2 \
  --target-model gpt2 \
  --branching-factor 2 \
  --max-new-tokens 2 \
  --seed 7
```

This generates:

- `artifacts/benchmarks/simulator_results.csv`
- `artifacts/benchmarks/real_results.csv`
- `artifacts/benchmarks/simulator_results.json`
- `artifacts/benchmarks/real_results.json`
- `artifacts/benchmarks/run_summary.json`
- plot images under `artifacts/benchmarks/plots/`

## 7. Get Help For Commands

Top-level help:

```bash
python "speculative devoding dynamic k.py" --help
```

Subcommand help:

```bash
python "speculative devoding dynamic k.py" bootstrap-venv --help
python "speculative devoding dynamic k.py" train --help
python "speculative devoding dynamic k.py" decode --help
python "speculative devoding dynamic k.py" benchmark --help
```

## 8. Common Notes

- The first real-model run may take longer because model weights are downloaded.
- If you have a Hugging Face token, export `HF_TOKEN` before running to improve download reliability.
- The script saves outputs under `artifacts/` by default.
- If you are using `zsh`, keep the filename in quotes exactly as shown above.
