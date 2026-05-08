# LLM-DynaQ

This folder contains an LLM-backed Dyna-Q loop that uses real speculative
 decoding interactions from the Hugging Face decoder.

## What is real vs simulated

Real (from decoder outputs):
- accepted_tokens
- emitted_tokens
- acceptance_rate
- rejection_position
- tree_nodes
- normalized_tree_cost
- target_logprob_mean
- draft/verify timing metrics

Simulated (environment dynamics approximation):
- queue_length
- system_load
- recent_arrival_rate
- estimated_batch_size
- sla_slack_ms
- tree_branching_factor

These simulated fields are updated with the same lightweight dynamics used by
 the original simulator, while acceptance and cost are driven by real decoding.

## Deterministic prompt sampling

Prompts are loaded with pandas from the HF dataset. Sampling is deterministic
 via a fixed seed and a sequential cursor. Use the dataset loader in
 prompt_dataset.py to switch to shuffled order.

## Run

From the repository root:

```bash
python "LLM-DynaQ/runner.py"
```

Edit constants in LLM-DynaQ/runner.py to change models, episode length, or
 sampling configuration.
