from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import torch

from spade import Dataset, SpadeConfig, optimize_spade, train_spade


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small SPADE training/optimization pass on an NPZ dataset.")
    parser.add_argument("--data", required=True, help="Path to a .npz file containing x and y arrays.")
    parser.add_argument("--epochs", type=int, default=1, help="Diffusion training epochs.")
    parser.add_argument("--gens", type=int, default=1, help="Evolutionary optimization generations.")
    parser.add_argument("--batch", type=int, default=64, help="Training batch size.")
    parser.add_argument("--hidden", type=int, default=128, help="Diffusion hidden width.")
    parser.add_argument("--diff-steps", type=int, default=20, help="Diffusion steps.")
    parser.add_argument("--mc-samples", type=int, default=8, help="Acquisition Monte Carlo samples.")
    parser.add_argument("--mc-steps", type=int, default=10, help="Acquisition DDIM steps.")
    parser.add_argument("--support-k", type=int, default=5, help="kNN support-proximity neighbors.")
    parser.add_argument("--no-calibration", action="store_true", help="Disable calibration loss for a faster smoke run.")
    parser.add_argument("--no-support", action="store_true", help="Disable support-proximity loss for a faster smoke run.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = Dataset.from_npz(args.data)
    support_k = max(1, min(int(args.support_k), int(data.x_norm.shape[0])))

    cfg = replace(
        SpadeConfig(),
        diff_epochs=args.epochs,
        diff_batch=args.batch,
        diff_hidden=args.hidden,
        diff_steps=args.diff_steps,
        calib_weight=0.0 if args.no_calibration else 1.0,
        calib_mc_samples=max(1, min(args.mc_samples, 4)),
        calib_mc_steps=max(1, min(args.mc_steps, args.diff_steps)),
        support_weight=0.0 if args.no_support else 1.0,
        support_k=support_k,
        acq_mc_samples=args.mc_samples,
        acq_mc_steps=max(1, min(args.mc_steps, args.diff_steps)),
        ea_pop=max(2, min(32, int(data.x_norm.shape[0]))),
        ea_elite=max(1, min(16, int(data.x_norm.shape[0]))),
        ea_gens=args.gens,
    )
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    model = train_spade(data, cfg, device=device)
    result = optimize_spade(model, data, cfg, device=device)

    print("device:", device)
    print("x_shape:", tuple(data.x.shape))
    print("y_range:", (float(np.min(data.y)), float(np.max(data.y))))
    print("best_x_norm:", result.x_best_norm)
    print("best_acq:", result.best_acq)


if __name__ == "__main__":
    main()
