#!/bin/bash
# Standalone smoke test for DenoiserSelfREPA (no DDP, no data): checks teacher-EMA sync,
# alignment hooks, loss balance, and that the frozen teacher is excluded from trainable params.
# Activate the `jit` conda env first (built from environment.yaml), then run from the repo root.
set -e
python smoke_selfrepa.py
