# decoding

This folder contains the real speculative decoder implementation.

- `speculative.py`: builds the speculative tree, packs it for SpecInfer-style
  verification, runs the single target-model verification pass, and returns the
  accepted prefix plus verifier token metadata.
