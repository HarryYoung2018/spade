from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

from spade.config import SpadeConfig
from spade.data import Dataset
from spade.diffusion import ScalarDiffusionSurrogate, mc_samples
from spade.knn import KnnStats
from spade.regularizers import calibration_loss, support_proximity_loss


class IndexedTensorDataset(TorchDataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, *, with_index: bool):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.with_index = bool(with_index)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        if self.with_index:
            return self.x[idx], self.y[idx], idx
        return self.x[idx], self.y[idx]


def set_seed(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_spade(
    dataset: Dataset,
    cfg: SpadeConfig,
    *,
    device: torch.device,
    knn_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> ScalarDiffusionSurrogate:
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    x_train = dataset.x_norm
    y_train = dataset.y_norm

    knn_stats = None
    if cfg.support_weight > 0.0:
        knn_stats = KnnStats(
            x_train,
            y_train,
            k=cfg.support_k,
            cache=cfg.support_cache,
            transform=knn_transform,
        )

    model = ScalarDiffusionSurrogate(
        x_dim=x_train.shape[1],
        steps=cfg.diff_steps,
        hidden=cfg.diff_hidden,
        t_dim=cfg.diff_t_dim,
        beta_start=cfg.diff_beta_start,
        beta_end=cfg.diff_beta_end,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.diff_lr)
    with_index = bool(knn_stats and cfg.support_cache)
    loader = DataLoader(
        IndexedTensorDataset(x_train, y_train, with_index=with_index),
        batch_size=cfg.diff_batch,
        shuffle=True,
    )

    for _ in range(cfg.diff_epochs):
        model.train()
        for batch in loader:
            if len(batch) == 3:
                xb, yb, idx = batch
            else:
                xb, yb = batch
                idx = None
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            loss = model.loss(xb, yb)

            if cfg.calib_weight > 0.0 or cfg.support_weight > 0.0:
                samples = mc_samples(
                    model,
                    xb,
                    samples=cfg.calib_mc_samples,
                    steps=cfg.calib_mc_steps,
                    eta=cfg.calib_mc_eta,
                    batch_size=cfg.calib_mc_batch,
                    requires_grad=True,
                )
                mu_hat = samples.mean(dim=0).view(-1, 1)
                sigma_hat = samples.std(dim=0, unbiased=False).view(-1, 1)

                if cfg.calib_weight > 0.0:
                    calib = calibration_loss(
                        mu_hat,
                        yb,
                        num_pairs=cfg.calib_pairs,
                        temperature=cfg.calib_temp,
                    )
                    loss = loss + cfg.calib_weight * calib

                if cfg.support_weight > 0.0 and knn_stats is not None:
                    mean_y, dist_k = knn_stats.query(xb, idx)
                    supp = support_proximity_loss(
                        mu_hat,
                        sigma_hat,
                        mean_y,
                        dist_k,
                        tau_a=cfg.support_tau_a,
                        sigma_a0=cfg.support_sigma_a0,
                        sigma_a1=cfg.support_sigma_a1,
                    )
                    loss = loss + cfg.support_weight * supp

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.diff_grad_clip)
            optimizer.step()

    return model
