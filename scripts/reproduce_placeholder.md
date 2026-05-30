# Reproducing ICML 2026 Experiments

This repository currently contains the compact SPADE implementation and a generic NPZ quickstart runner. Full benchmark preprocessing scripts and final ICML hyperparameter files are not included yet.

Available:

```bash
python scripts/run_quickstart_npz.py --data dataset.npz --epochs 1 --gens 1
```

TODO:

- Add Design-Bench preprocessing scripts.
- Add TFBind8/TFBind10 configs.
- Add LLM-DM preprocessing instructions.
- Add final ICML hyperparameter config files.

Do not treat this scaffold as a complete benchmark reproduction suite.
