# SPADE: Support-Proximity Augmented Diffusion Estimation

This repository contains a minimal, core implementation of SPADE for offline black-box optimization. The code is intentionally compact and focused on a conditional diffusion surrogate trained with calibrated diffusion estimation and support-proximity regularization, then optimized via LCB acquisition and evolutionary search.

## Usage

### 1) Prepare data
Provide a NumPy `.npz` file with:
- `x`: shape `(N, D)` design vectors
- `y`: shape `(N,)` or `(N, 1)` property scores

### 2) Train the surrogate and run optimization
```python
import torch
from spade import Dataset, SpadeConfig, train_spade, optimize_spade

# Load dataset
data = Dataset.from_npz("dataset.npz")

# Configure SPADE
cfg = SpadeConfig()

# Train diffusion surrogate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = train_spade(data, cfg, device=device)

# Optimize designs with LCB + evolutionary search
result = optimize_spade(model, data, cfg, device=device)
print("best_x_norm:", result.x_best_norm)
print("best_acq:", result.best_acq)
```

### 3) Optional: discrete or structured design spaces
If your design space is discrete (e.g., categorical sequences), you can:
- provide a `transform` for kNN distance (e.g., logits -> probabilities), and
- provide a `project_fn` to map continuous candidates back to valid discrete designs.

Both hooks are supported in `optimize_spade` and `KnnStats`/`KnnDensityHelper`.

## Dependencies
- Python 3.9+
- `numpy`
- `torch`
- `scikit-learn`

## Repo layout (core only)
- `spade/diffusion.py`: conditional diffusion surrogate + DDIM sampling
- `spade/regularizers.py`: calibration + support-proximity losses
- `spade/knn.py`: kNN density helper and cached training stats
- `spade/train.py`: training loop for surrogate fitting
- `spade/optimize.py`: LCB acquisition + evolutionary search
- `spade/data.py`: minimal dataset loading and normalization
