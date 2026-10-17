"""State log: the cell-delta store and authoritative record (SPEC §2.5).

One entry `(step, cell, png)` **only when a cell changes** — lossless
(palette-indexed P-mode PNGs), disk-efficient, and the source for video
reconstruction. Plus the current-field snapshot for resumable restarts:
current field snapshot + log => the applet restarts cleanly.

Layout under `state/<run_name>/`:
  log/step_%05d/r%02dc%02d.png   changed cells per step
  snapshot/field_%05d.npy        field index array after the step
  metrics.csv                    one row per step (see metrics.py)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .grid import Field
from .palette import Palette


class StateLog:
    def __init__(self, root: str | Path, field: Field):
        self.root = Path(root)
        self.field = field
        self.log_dir = self.root / "log"
        self.snap_dir = self.root / "snapshot"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.snap_dir.mkdir(parents=True, exist_ok=True)

    # -- writes ----------------------------------------------------------- #
    def write_step(self, step: int, changed: list[tuple[int, int]]) -> list[Path]:
        """Store one PNG per *changed* cell; unchanged cells are not logged."""
        if not changed:
            return []
        d = self.log_dir / f"step_{step:05d}"
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for r, c in changed:
            p = d / f"r{r:02d}c{c:02d}.png"
            p.write_bytes(_png_bytes(self.field.palette, self.field.cell_index(r, c)))
            paths.append(p)
        return paths

    def save_snapshot(self, step: int) -> Path:
        p = self.snap_dir / f"field_{step:05d}.npy"
        np.save(p, self.field.indices)
        return p

    # -- reads -------------------------------------------------------------- #
    def steps(self) -> list[int]:
        out = []
        if self.log_dir.exists():
            for d in self.log_dir.iterdir():
                if d.is_dir() and d.name.startswith("step_"):
                    out.append(int(d.name.split("_")[1]))
        return sorted(out)

    def load_cell(self, step: int, r: int, c: int) -> np.ndarray:
        p = self.log_dir / f"step_{step:05d}" / f"r{r:02d}c{c:02d}.png"
        idx = _png_index(p, self.field.palette.n)
        return idx

    def changed_cells(self, step: int) -> list[tuple[int, int]]:
        d = self.log_dir / f"step_{step:05d}"
        if not d.exists():
            return []
        out = []
        for p in sorted(d.glob("r*.png")):
            m = p.stem  # r03c04
            out.append((int(m[1:3]), int(m[4:6])))
        return out

    @staticmethod
    def latest_snapshot(root: str | Path) -> tuple[int, np.ndarray] | None:
        snap_dir = Path(root) / "snapshot"
        snaps = sorted(snap_dir.glob("field_*.npy")) if snap_dir.exists() else []
        if not snaps:
            return None
        step = int(snaps[-1].name.split("_")[1].split(".")[0])
        return step, np.load(snaps[-1])


def _png_bytes(palette: Palette, indices: np.ndarray) -> bytes:
    import io
    buf = io.BytesIO()
    palette.pil_image(indices).save(buf, format="PNG")
    return buf.getvalue()


def _png_index(path: str | Path, palette_n: int) -> np.ndarray:
    from PIL import Image
    img = Image.open(path)
    if img.mode != "P":
        raise ValueError(f"{path}: expected palette-mode PNG")
    # trust the stored indices; check they are in range for the current palette
    idx = np.asarray(img, dtype=np.uint8)
    if idx.max() >= palette_n:
        raise ValueError(f"{path}: index {idx.max()} out of range for palette of {palette_n}")
    return idx
