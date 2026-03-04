from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def z_score(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (arr - mean) / std, mean, std


def z_score_denorm(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return arr * std + mean


def clip_to_bounds(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return np.clip(x, lower, upper)


def clip_and_renorm(
    x_norm: np.ndarray, bounds: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray
) -> np.ndarray:
    x_raw = z_score_denorm(x_norm, x_mean, x_std)
    x_raw = clip_to_bounds(x_raw, bounds)
    return (x_raw - x_mean) / x_std


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray
    y: np.ndarray
    x_norm: np.ndarray
    y_norm: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    bounds: np.ndarray
    meta: dict[str, Any] | None = None

    @classmethod
    def from_npz(cls, path: str, *, meta: dict[str, Any] | None = None) -> "Dataset":
        payload = np.load(path)
        x = np.asarray(payload["x"], dtype=np.float32)
        y = np.asarray(payload["y"], dtype=np.float32)
        y = _ensure_2d(y)
        x_norm, x_mean, x_std = z_score(x)
        y_norm, y_mean, y_std = z_score(y)
        bounds = np.stack([x.min(axis=0), x.max(axis=0)], axis=1)
        return cls(
            x=x,
            y=y,
            x_norm=x_norm,
            y_norm=y_norm,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            bounds=bounds,
            meta=meta,
        )
