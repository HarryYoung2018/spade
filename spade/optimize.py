from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from spade.config import SpadeConfig
from spade.data import Dataset, clip_and_renorm, z_score_denorm
from spade.diffusion import ScalarDiffusionSurrogate, mc_samples
from spade.knn import build_knn_helper
from spade.regularizers import apply_support_transform


@dataclass(frozen=True)
class OptimizationResult:
    x_best_norm: np.ndarray
    best_acq: float
    population_norm: np.ndarray | None


def acquisition_from_samples(
    samples: torch.Tensor | np.ndarray,
    y_best: float,
    *,
    kind: str,
    ucb_beta: float,
    ei_xi: float = 0.0,
    mvr_lambda: float = 0.0,
) -> torch.Tensor:
    if not torch.is_tensor(samples):
        samples_t = torch.as_tensor(samples, dtype=torch.float32)
    else:
        samples_t = samples
    if samples_t.ndim == 1:
        samples_t = samples_t.unsqueeze(0)

    kind = str(kind).lower()
    mu = samples_t.mean(dim=0)
    sigma = samples_t.std(dim=0, unbiased=False).clamp_min(1e-8)
    if kind == "lcb":
        return mu - float(ucb_beta) * sigma
    if kind == "ei":
        xi = float(ei_xi)
        z = (mu - float(y_best) - xi) / sigma
        normal = torch.distributions.Normal(0.0, 1.0)
        cdf = normal.cdf(z)
        pdf = torch.exp(normal.log_prob(z))
        return (mu - float(y_best) - xi) * cdf + sigma * pdf
    if kind == "mvr":
        lam = float(mvr_lambda)
        return mu - lam * (sigma ** 2)
    raise ValueError(f"Unknown acquisition kind: {kind}")


def evolutionary_optimize(
    score_fn: Callable[[np.ndarray], np.ndarray],
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    bounds: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    rng: np.random.Generator,
    pop_size: int,
    elite_size: int,
    gens: int,
    mut_sigma_init: float,
    mut_sigma_min: float,
    crossover_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    pop_size = min(int(pop_size), int(x_train.shape[0]))
    elite_size = min(int(elite_size), pop_size)

    y_flat = np.asarray(y_train).reshape(-1)
    idx = np.argsort(-y_flat)[:pop_size]
    pop = x_train[idx].copy()
    pop = pop + rng.normal(0, 0.05, size=pop.shape).astype(np.float32)
    pop = clip_and_renorm(pop, bounds, x_mean, x_std)

    mut_sigma = float(mut_sigma_init)
    best_x = pop[0].copy()
    best_score = -float("inf")

    for _ in range(int(gens)):
        scores = score_fn(pop)
        elite_idx = np.argsort(-scores)[:elite_size]
        elites = pop[elite_idx].copy()
        elite_scores = scores[elite_idx]

        if float(elite_scores[0]) > best_score:
            best_score = float(elite_scores[0])
            best_x = elites[0].copy()

        new_pop = list(elites.tolist())

        num_crossover = int((pop_size - elite_size) * float(crossover_rate))
        for _ in range(num_crossover):
            parent1 = elites[rng.integers(0, elite_size)]
            parent2 = elites[rng.integers(0, elite_size)]
            alpha = float(rng.uniform(0.0, 1.0))
            child = alpha * parent1 + (1.0 - alpha) * parent2
            new_pop.append(child.tolist())

        while len(new_pop) < pop_size:
            parent = elites[rng.integers(0, elite_size)]
            child = parent + rng.normal(0, mut_sigma, size=(x_train.shape[1],))
            new_pop.append(child.tolist())

        pop = np.asarray(new_pop, dtype=np.float32)
        pop = clip_and_renorm(pop, bounds, x_mean, x_std)
        mut_sigma = max(mut_sigma * 0.98, float(mut_sigma_min))

    return best_x, pop


def optimize_spade(
    model: ScalarDiffusionSurrogate,
    dataset: Dataset,
    cfg: SpadeConfig,
    *,
    device: torch.device,
    knn_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    support_transform: bool = False,
    project_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> OptimizationResult:
    x_train = dataset.x_norm
    y_train = dataset.y_norm
    y_best = float(np.max(y_train))
    rng = np.random.default_rng(cfg.seed)

    knn_helper = None
    if support_transform and cfg.support_k > 0:
        knn_helper = build_knn_helper(x_train, [cfg.support_k], transform=knn_transform)

    def score_population(pop: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(pop).float().to(device)
        samples = mc_samples(
            model,
            xt,
            samples=cfg.acq_mc_samples,
            steps=cfg.acq_mc_steps,
            eta=cfg.acq_mc_eta,
            batch_size=cfg.acq_mc_batch,
            requires_grad=False,
        )
        if support_transform and knn_helper is not None:
            log_rk = knn_helper.log_rk(pop, cfg.support_k)
            log_rk_t = torch.from_numpy(log_rk).to(samples.device, dtype=samples.dtype)
            samples = apply_support_transform(
                samples,
                log_rk_t,
                tau_a=cfg.support_tau_a,
                sigma_a0=cfg.support_sigma_a0,
                sigma_a1=cfg.support_sigma_a1,
            )
        acq = acquisition_from_samples(
            samples,
            y_best,
            kind=cfg.acq_kind,
            ucb_beta=cfg.acq_beta,
            ei_xi=cfg.acq_ei_xi,
            mvr_lambda=cfg.acq_mvr_lambda,
        )
        return acq.detach().cpu().numpy()

    best_x_norm, pop = evolutionary_optimize(
        score_population,
        x_train=x_train,
        y_train=y_train,
        bounds=dataset.bounds,
        x_mean=dataset.x_mean,
        x_std=dataset.x_std,
        rng=rng,
        pop_size=cfg.ea_pop,
        elite_size=cfg.ea_elite,
        gens=cfg.ea_gens,
        mut_sigma_init=cfg.ea_mut_sigma_init,
        mut_sigma_min=cfg.ea_mut_sigma_min,
        crossover_rate=cfg.ea_crossover,
    )

    if project_fn is not None:
        x_raw = z_score_denorm(best_x_norm, dataset.x_mean, dataset.x_std)
        x_raw = project_fn(x_raw)
        best_x_norm = (x_raw - dataset.x_mean) / dataset.x_std

    best_acq = float(score_population(best_x_norm[None, :])[0])
    return OptimizationResult(
        x_best_norm=best_x_norm.astype(np.float32),
        best_acq=best_acq,
        population_norm=pop.astype(np.float32),
    )
