# dynamic k

This is the runnable prototype for Sprint 1.

## Main Entry Points

- `train_sprint1.py`: train the simulator-side RL controller.
- `run_real_sprint1.py`: run real speculative decoding with online RL updates.
- `experiments/run_benchmarks.py`: generate simulator and real-model benchmark artifacts.

## Main Packages

- `decoding/`: speculative tree decoder.
- `rl/`: tabular Q-learning controller.
- `simulation/`: lightweight serving simulator.
- `state/`: raw state initialization and tabular featurization.
- `utils/`: reward helpers and environment-driven model config.
- `tests/`: unit and smoke tests.
- `artifacts/`: saved checkpoints and benchmark outputs.

Use [HOW_TO_RUN.md](/Users/youcefs/Desktop/school/RL/PROJECT/HOW_TO_RUN.md) for
commands.
