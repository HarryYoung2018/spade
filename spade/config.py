from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpadeConfig:
    # Diffusion surrogate
    diff_steps: int = 100
    diff_hidden: int = 2048
    diff_t_dim: int = 128
    diff_beta_start: float = 1e-4
    diff_beta_end: float = 2e-2
    diff_lr: float = 1e-3
    diff_grad_clip: float = 1.0
    diff_epochs: int = 100
    diff_batch: int = 64

    # Calibrated diffusion estimation (moment + rank)
    calib_weight: float = 1.0
    calib_pairs: int = 32
    calib_temp: float = 1.0
    calib_mc_samples: int = 4
    calib_mc_steps: int = 10
    calib_mc_eta: float = 0.0
    calib_mc_batch: int = 4096

    # Support-proximity regularization (kNN)
    support_weight: float = 1.0
    support_k: int = 10
    support_tau_a: float = 0.02
    support_sigma_a0: float = 0.02
    support_sigma_a1: float = 0.005
    support_cache: bool = True

    # Acquisition and optimization
    acq_kind: str = "lcb"
    acq_beta: float = 0.1
    acq_ei_xi: float = 0.0
    acq_mvr_lambda: float = 0.1
    acq_mc_samples: int = 256
    acq_mc_steps: int = 50
    acq_mc_eta: float = 0.0
    acq_mc_batch: int = 4096

    # Evolutionary search
    ea_pop: int = 128
    ea_elite: int = 64
    ea_gens: int = 100
    ea_mut_sigma_init: float = 0.12
    ea_mut_sigma_min: float = 0.02
    ea_crossover: float = 0.3

    # Misc
    seed: int = 0
    deterministic: bool = False
