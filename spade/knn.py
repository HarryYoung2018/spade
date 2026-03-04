from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, :]
    return arr


class KnnDensityHelper:
    def __init__(
        self,
        x_train: np.ndarray,
        max_k: int,
        *,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.transform = transform
        x_fit = transform(x_train) if transform is not None else x_train
        x_fit = _as_2d(x_fit)
        self.x = x_fit
        self.dim = x_fit.shape[1]
        self.n = x_fit.shape[0]
        self.nn = NearestNeighbors(n_neighbors=max_k).fit(x_fit)
        self.max_k = int(max_k)

    def _apply(self, x_query: np.ndarray) -> np.ndarray:
        return self.transform(x_query) if self.transform is not None else x_query

    def log_rk(self, x_query: np.ndarray, k: int) -> np.ndarray:
        x_query = _as_2d(x_query)
        x_query = self._apply(x_query)
        k = int(k)
        if k <= 0 or k > self.max_k:
            raise ValueError(f"Invalid k={k} for kNN helper (max_k={self.max_k}).")
        dists = self.nn.kneighbors(x_query, n_neighbors=k, return_distance=True)[0]
        r_k = dists[:, -1] + 1e-8
        return np.log(r_k)


class KnnStats:
    """Cached kNN statistics for support-proximity regularization."""

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        k: int,
        *,
        cache: bool = True,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.transform = transform
        x_fit = transform(x_train) if transform is not None else x_train
        x_fit = _as_2d(x_fit)
        self.y = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
        self.k = int(k)
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(x_fit)
        self.cache = bool(cache)
        self._mean_cache: np.ndarray | None = None
        self._dist_cache: np.ndarray | None = None
        if self.cache:
            dists, idxs = self.nn.kneighbors(x_fit, n_neighbors=self.k, return_distance=True)
            self._mean_cache = self.y[idxs].mean(axis=1)
            self._dist_cache = dists[:, -1:]

    def query(
        self, x_query: torch.Tensor, idx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if idx is not None and self._mean_cache is not None:
            idx_np = idx.detach().cpu().numpy()
            mean_y = self._mean_cache[idx_np]
            dist_k = self._dist_cache[idx_np]
        else:
            x_np = x_query.detach().cpu().numpy()
            if self.transform is not None:
                x_np = self.transform(x_np)
            dists, idxs = self.nn.kneighbors(x_np, n_neighbors=self.k, return_distance=True)
            mean_y = self.y[idxs].mean(axis=1)
            dist_k = dists[:, -1:]
        device = x_query.device
        mean_y = mean_y.reshape(-1, 1)
        return (
            torch.from_numpy(mean_y).to(device=device, dtype=torch.float32),
            torch.from_numpy(dist_k).to(device=device, dtype=torch.float32),
        )


def build_knn_helper(
    x_train: np.ndarray,
    k_list: Sequence[int],
    *,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> KnnDensityHelper:
    max_k = max([int(k) for k in k_list]) if k_list else 1
    return KnnDensityHelper(x_train, max_k=max_k, transform=transform)
