"""Per-step metrics and the collapse watch (SPEC §2.8).

Attractor collapse is the main qualitative risk: the diffusion model has
known attractors (see base_workflows/OBSERVATIONS.md) and a CA can
homogenize onto one. Logged per step to CSV + (later) GUI panel:

- changed_cells   count of cells that changed this step
- neighbor_dist   mean RGB color distance over all adjacent cell pairs
- palette_entropy Shannon entropy (bits) of the global palette-index histogram
- uniformity      fraction of cells identical to the most common cell
                 (warning above `uniformity_warn`)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .grid import Field

CSV_COLUMNS = ["step", "changed_cells", "neighbor_dist", "palette_entropy", "uniformity", "warn"]


@dataclass
class StepMetrics:
    step: int
    changed_cells: int
    neighbor_dist: float
    palette_entropy: float
    uniformity: float
    warn: bool = False


def neighbor_distance(field: Field) -> float:
    """Mean Euclidean RGB distance between adjacent cell tiles (wrapping).

    Pairs: (r,c)-(r,c+1) and (r,c)-(r+1,c), each pair counted once.
    Distance is over the full tile (art*art*3 components), range 0..255*sqrt(3*art^2).
    """
    P = field.palette.colors.astype(np.float64)
    A = field.indices
    n = field.n
    total, cnt = 0.0, 0
    for r in range(n):
        for c in range(n):
            C = P[A[r, c]]
            total += float(np.sqrt(((C - P[A[r, (c + 1) % n]]) ** 2).sum()))
            total += float(np.sqrt(((C - P[A[(r + 1) % n, c]]) ** 2).sum()))
            cnt += 2
    return total / cnt if cnt else 0.0


def palette_entropy(field: Field) -> float:
    """Shannon entropy (bits) of the histogram of palette indices over the field."""
    idx = field.indices.reshape(-1)
    counts = np.bincount(idx, minlength=field.palette.n).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def uniformity(field: Field) -> float:
    """Fraction of cells identical to the most common cell (exact index match)."""
    cells = field.indices.reshape(field.n * field.n, -1)
    _, counts = np.unique(cells, axis=0, return_counts=True)
    return float(counts.max() / counts.sum())


def compute(field: Field, changed_cells: int, step: int,
            warn_threshold: float) -> StepMetrics:
    u = uniformity(field)
    return StepMetrics(
        step=step,
        changed_cells=changed_cells,
        neighbor_dist=neighbor_distance(field),
        palette_entropy=palette_entropy(field),
        uniformity=u,
        warn=bool(u > warn_threshold),
    )


class MetricsLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with self.path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(CSV_COLUMNS)

    def append(self, m: StepMetrics) -> None:
        d = asdict(m)
        d["warn"] = int(m.warn)
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([d[k] for k in CSV_COLUMNS])

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return list(csv.DictReader(f))
