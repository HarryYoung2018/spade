from __future__ import annotations

import torch
import torch.nn.functional as F


def calibration_loss(
    mu_hat: torch.Tensor,
    y_true: torch.Tensor,
    *,
    num_pairs: int,
    temperature: float,
) -> torch.Tensor:
    mu_hat = mu_hat.view(-1)
    y_true = y_true.view(-1)
    moment = F.mse_loss(mu_hat, y_true)

    rank_loss = torch.tensor(0.0, device=mu_hat.device)
    if num_pairs > 0 and y_true.numel() > 1:
        idx1 = torch.randint(0, y_true.numel(), (num_pairs,), device=mu_hat.device)
        idx2 = torch.randint(0, y_true.numel(), (num_pairs,), device=mu_hat.device)
        mask = y_true[idx1] > y_true[idx2]
        if mask.any():
            diff = mu_hat[idx1[mask]] - mu_hat[idx2[mask]]
            rank_loss = F.softplus(-float(temperature) * diff).mean()

    return moment + rank_loss


def support_proximity_loss(
    mu_hat: torch.Tensor,
    sigma_hat: torch.Tensor,
    neighbor_mean: torch.Tensor,
    dist_k: torch.Tensor,
    *,
    tau_a: float,
    sigma_a0: float,
    sigma_a1: float,
) -> torch.Tensor:
    log_rk = torch.log(dist_k + 1e-8)
    tau = float(tau_a) * log_rk
    sigma_min = float(sigma_a0) + float(sigma_a1) * log_rk

    mean_shrink = F.relu(mu_hat - neighbor_mean - tau)
    var_floor = F.relu(sigma_min - sigma_hat)
    return (mean_shrink + var_floor).mean()


def apply_support_transform(
    samples: torch.Tensor,
    log_rk: torch.Tensor,
    *,
    tau_a: float,
    sigma_a0: float,
    sigma_a1: float,
) -> torch.Tensor:
    mu = samples.mean(dim=0)
    sigma = samples.std(dim=0, unbiased=False)
    log_rk = log_rk.view(-1)

    mu_adj = mu - float(tau_a) * log_rk
    sigma_min = float(sigma_a0) + float(sigma_a1) * log_rk
    sigma_adj = torch.maximum(sigma, sigma_min)
    scale = sigma_adj / (sigma + 1e-8)
    return (samples - mu) * scale + mu_adj
