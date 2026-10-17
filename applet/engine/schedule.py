"""Schedule policies: pick which cells update this step (SPEC §2.2).

Interface: `select(field, step) -> list[(row, col)]`. The policy is the
pluggable module; these are the simple defaults to build against. K is the
main speed dial (step time ~ K x (rule + generation)).
"""
from __future__ import annotations

import numpy as np

from .grid import Field


class Schedule:
    name = "base"

    def __init__(self, n: int, K: int):
        self.n = n
        self.K = K

    def select(self, field: Field, step: int) -> list[tuple[int, int]]:
        raise NotImplementedError


class RandomUniform(Schedule):
    """K distinct cells, uniformly at random (fixed rng => reproducible)."""

    name = "random_uniform"

    def __init__(self, n: int, K: int, rng: np.random.Generator):
        super().__init__(n, K)
        self.rng = rng

    def select(self, field: Field, step: int) -> list[tuple[int, int]]:
        n2 = self.n * self.n
        k = min(self.K, n2)
        flat = self.rng.choice(n2, size=k, replace=False)
        return [(int(i // self.n), int(i % self.n)) for i in flat]


class RingScan(Schedule):
    """Deterministic moving band: diagonal rings ((r+c) mod (2n-1)), one ring
    offset per step, K cells drawn from the band and the following rings.
    Covers the whole field evenly over 2n-1 steps; no rng involved."""

    name = "ring_scan"

    def _rings(self) -> list[np.ndarray]:
        # ring i = cells with (r+c) == i, for i in 0..2n-2
        rings: list[np.ndarray] = []
        for i in range(2 * self.n - 1):
            cells = [(r, c) for r in range(self.n) for c in range(self.n) if (r + c) == i]
            rings.append(np.asarray(cells, dtype=np.int64))
        return rings

    def select(self, field: Field, step: int) -> list[tuple[int, int]]:
        n2 = self.n * self.n
        k = min(self.K, n2)
        rings = self._rings()
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        m = len(rings)
        i = 0
        while len(out) < k and i < m * 2:  # wrap at most two full passes
            ring = rings[(step + i) % m]
            for r, c in ring:
                cell = (int(r), int(c))
                if cell not in seen:
                    seen.add(cell)
                    out.append(cell)
                if len(out) == k:
                    break
            i += 1
        return out[:k]


class All(Schedule):
    """Every cell each step (K = n^2; slow)."""

    name = "all"

    def select(self, field: Field, step: int) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.n) for c in range(self.n)]


def make_schedule(name: str, n: int, K: int, seed: int | None = None) -> Schedule:
    if name == "random_uniform":
        return RandomUniform(n, K, np.random.default_rng(seed))
    if name == "ring_scan":
        return RingScan(n, K)
    if name == "all":
        return All(n, K)
    raise ValueError(f"unknown schedule {name!r}")
