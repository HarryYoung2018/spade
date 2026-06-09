# Reproducibility Status

This repository contains the compact SPADE implementation, package metadata, smoke tests, and a generic NPZ runner. It is suitable for verifying the public API and running SPADE on a user-provided dataset with arrays `x` and `y`.

## Available

Install the package:

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

Run smoke tests:

```bash
python -m pytest
```

Run a small NPZ experiment:

```bash
python scripts/run_quickstart_npz.py --data dataset.npz --epochs 1 --gens 1
```

## Not Included

The current public release does not include the full benchmark reproduction stack used for the ICML 2026 paper:

- Design-Bench preprocessing and task-specific evaluation wrappers.
- TFBind8/TFBind10 preprocessing and benchmark configs.
- LLM-DM preprocessing and evaluation instructions.
- Final per-task hyperparameter files for the paper tables.

The released code should therefore be treated as the method implementation and NPZ quickstart, not as a complete benchmark automation suite.
