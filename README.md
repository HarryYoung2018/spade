# SPADE: Support-Proximity Augmented Diffusion Estimation

Official implementation of:

**Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization**

Accepted by **ICML 2026**.

[![arXiv](https://img.shields.io/badge/arXiv-2605.11246-b31b1b.svg)](https://arxiv.org/abs/2605.11246)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-243b6b.svg)](https://icml.cc/)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776ab.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Project page](https://img.shields.io/badge/project-page-1f6feb.svg)](https://harryyoung2018.github.io/spade/)

| Resource | Link |
|---|---|
| Paper | [arXiv:2605.11246](https://arxiv.org/abs/2605.11246) |
| PDF | [arXiv PDF](https://arxiv.org/pdf/2605.11246) |
| Project page | [harryyoung2018.github.io/spade](https://harryyoung2018.github.io/spade/) |
| Poster | [ICML 2026 poster](docs/assets/files/spade_icml2026_poster.pdf) |
| Slides | [PDF](docs/assets/files/spade_icml2026_slides.pdf) · [PPTX](docs/assets/files/spade_icml2026_slides.pptx) |

## TL;DR

SPADE turns forward surrogate modeling into a calibrated conditional diffusion problem and injects a kNN support prior to prevent offline optimizers from exploiting unsupported regions.

## Overview

Offline black-box optimization searches for high-scoring designs from a fixed dataset without online oracle access. A central difficulty is OOD exploitation: optimizers can chase surrogate errors in unsupported regions.

SPADE models the forward likelihood \(p(y|x)\) with a conditional diffusion surrogate, then adds calibration and support-proximity regularization so acquisition optimization remains both expressive and conservative. The method first learns an uncertainty-aware forward surrogate, then searches candidates with a lower-confidence-bound acquisition and evolutionary updates.

## Method summary

- **Conditional Diffusion Surrogate:** models \(p_\theta(y|x)\), giving a predictive distribution rather than only a point estimate.
- **Calibrated Diffusion Estimation:** adds moment matching and pairwise rank consistency so the surrogate is useful for optimization.
- **Support-Proximity Regularization:** uses kNN distance to shrink predicted means and inflate uncertainty in low-density regions.

```math
p(x | y) \propto p(y | x)p(x)
```

```math
\mathrm{LCB}(x) = \hat{\mu}_\theta(x) - \beta \hat{\sigma}_\theta(x)
```

## Results snapshot

| Metric | SPADE |
|---|---:|
| Mean rank, normalized max score | 2.8 / 24 |
| Median rank, normalized max score | 1.5 / 24 |
| Top-2 tasks, normalized max score | 5 / 6 |
| Mean rank, normalized median score | 1.7 / 24 |
| Median rank, normalized median score | 1.0 / 24 |

| Task | SPADE normalized max score |
|---|---:|
| SuperC | 0.546 ± 0.013 |
| Ant | 0.978 ± 0.006 |
| D’Kitty | 0.981 ± 0.003 |
| LLM-DM | 1.019 ± 0.064 |
| TF8 | 0.923 ± 0.015 |
| TF10 | 0.915 ± 0.010 |

## Installation

```bash
git clone https://github.com/HarryYoung2018/spade.git
cd spade

conda create -n spade python=3.10 -y
conda activate spade
pip install -r requirements.txt
pip install -e .
```

For development and smoke tests:

```bash
pip install -e ".[dev]"
python -m pytest
```

## Quick start

```python
import torch
from spade import Dataset, SpadeConfig, train_spade, optimize_spade

data = Dataset.from_npz("dataset.npz")
cfg = SpadeConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = train_spade(data, cfg, device=device)
result = optimize_spade(model, data, cfg, device=device)

print("best_x_norm:", result.x_best_norm)
print("best_acq:", result.best_acq)
```

For a command-line NPZ smoke run:

```bash
python scripts/run_quickstart_npz.py --data dataset.npz --epochs 1 --gens 1
```

## Data format

SPADE expects a NumPy `.npz` file with:

- `x`: shape `(N, D)`
- `y`: shape `(N,)` or `(N, 1)`

Datasets are not bundled in this repository. Benchmark-specific preprocessing for Design-Bench, TFBind, and LLM-DM is not included in the current public release.

## Reproducibility status

This repository provides the compact SPADE implementation, editable package installation, smoke tests, and a generic NPZ quickstart runner. It does not currently include the full benchmark preprocessing and evaluation scripts used for the ICML 2026 paper.

Available:

```text
scripts/
  run_quickstart_npz.py
  reproducibility_status.md
tests/
  test_smoke.py
```

Not included in this release:

- Design-Bench preprocessing and task-specific evaluation wrappers.
- TFBind8/TFBind10 preprocessing and benchmark configs.
- LLM-DM preprocessing and evaluation instructions.
- Final per-task experiment configuration files used for the paper tables.

See [scripts/reproducibility_status.md](scripts/reproducibility_status.md) for the current reproducibility surface.

## Repository layout

```text
spade/
  diffusion.py
  regularizers.py
  knn.py
  train.py
  optimize.py
  data.py
  config.py
docs/
  index.html
  assets/
scripts/
  run_quickstart_npz.py
  reproducibility_status.md
tests/
  test_smoke.py
```

## Project page

The project page is served from `docs/` via GitHub Pages:

https://harryyoung2018.github.io/spade/

Local preview:

```bash
python3 -m http.server 8000 -d docs
```

Then open:
http://localhost:8000

## Citation

If you find SPADE useful in your research, please cite the ICML paper:

```bibtex
@inproceedings{yang2026spade,
  title     = {Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization},
  author    = {Yang, Yonghan and Yuan, Ye and Sun, Zipeng and Du, Linfeng and He, Bowei and Wu, Haolun and Chen, Can and Liu, Xue},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  address   = {Seoul, South Korea},
  publisher = {PMLR},
  year      = {2026}
}
```

arXiv citation:

```bibtex
@article{yang2026support,
  title   = {Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization},
  author  = {Yang, Yonghan and Yuan, Ye and Sun, Zipeng and Du, Linfeng and He, Bowei and Wu, Haolun and Chen, Can and Liu, Xue},
  journal = {arXiv preprint arXiv:2605.11246},
  year    = {2026}
}
```

## License

This repository is released under the [MIT License](LICENSE).

## Acknowledgments

SPADE builds on the offline black-box optimization benchmark ecosystem, including Design-Bench-style tasks and the LLM-DM benchmark setting used in the paper.
