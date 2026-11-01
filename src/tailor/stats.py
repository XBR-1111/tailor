"""Numerically stable streaming statistics over episode/file chunks.

Mean/std use the Chan et al. parallel-merge form of Welford's algorithm:
each chunk reduces to (count, mean, M2) with numpy, chunks merge pairwise.
This is one pass, O(chunk) memory, and does not suffer the catastrophic
cancellation of the naive sum-of-squares formula — datasets too big to
concatenate in memory (the reason this module exists) reduce exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunningStats:
    """Stream min/max/mean/std over (n_frames, dims) chunks.

    update() with each chunk; read .min/.max/.mean/.std/.count at the end.
    Untouched (no update called) reads are all zeros with count 0.
    """

    _count: int = 0
    _min: np.ndarray | None = None
    _max: np.ndarray | None = None
    _mean: np.ndarray | None = None
    _m2: np.ndarray | None = None
    _chunks: list = field(default_factory=list, repr=False)
    _cache: tuple | None = None
    _dims: int = 0

    def update(self, arr: np.ndarray) -> None:
        """Fold one (n, dims) float array into the running state."""
        if arr.size == 0:
            return
        arr = arr.astype(np.float64, copy=False)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n, dims = arr.shape
        chunk_mean = arr.mean(axis=0)
        chunk_m2 = ((arr - chunk_mean) ** 2).sum(axis=0)
        self._chunks.append((n, chunk_mean, chunk_m2, arr.min(axis=0), arr.max(axis=0)))
        self._count += n
        self._dims = dims
        self._cache = None  # invalidate merged view

    def _merged(self) -> tuple:
        if self._cache is not None:
            return self._cache
        # Chan et al. merge; O(chunks), numerically stable
        it = iter(self._chunks)
        n, mu, m2, lo, hi = next(it)
        for n_b, mu_b, m2_b, lo_b, hi_b in it:
            n_a = n
            delta = mu_b - mu
            n = n_a + n_b
            mu = mu + delta * n_b / n
            m2 = m2 + m2_b + delta * delta * n_a * n_b / n
            lo = np.minimum(lo, lo_b)
            hi = np.maximum(hi, hi_b)
        result = (n, mu, m2, lo, hi)
        self._cache = result
        return result

    @property
    def count(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(getattr(self, "_dims", 0), dtype=np.int64)
        n, _, _, _, _ = self._merged()
        return np.full(self._dims, n, dtype=np.int64)

    @property
    def min(self) -> np.ndarray:
        if not self._chunks:
            raise ValueError("no data")
        _, _, _, lo, _ = self._merged()
        return lo

    @property
    def max(self) -> np.ndarray:
        if not self._chunks:
            raise ValueError("no data")
        _, _, _, _, hi = self._merged()
        return hi

    @property
    def mean(self) -> np.ndarray:
        if not self._chunks:
            raise ValueError("no data")
        _, mu, _, _, _ = self._merged()
        return mu

    @property
    def std(self) -> np.ndarray:
        if not self._chunks:
            raise ValueError("no data")
        n, _, m2, _, _ = self._merged()
        return np.sqrt(m2 / n)
