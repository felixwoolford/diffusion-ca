"""Step orchestration (SPEC §1 dynamics) and generators.

One lockstep step:
  1. schedule policy selects K cells
  2. a snapshot of the field is frozen
  3. per selected cell: rule (LLM/template) reads the 5 neighborhood tiles
     from the snapshot and emits one prompt
  4. generator renders the new tile (ComfyUI arm or fake)
  5. output is block-quantized down to the art grid and written back
  6. state log + snapshot + metrics
Rule/generation dispatch runs with `concurrency` workers (the server
serializes anyway); write-back happens in schedule order for determinism.
"""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config
from .comfy import ComfyClient
from .grid import Field
from .imageops import block_mode_downscale, nearest_resize
from .metrics import MetricsLog, StepMetrics, compute
from .render import render_cross
from .schedule import Schedule, make_schedule
from .statelog import StateLog
from ..rule import RuleBackend, RuleRequest
from ..rule.prompts import frame_text


@dataclass
class GenRequest:
    arm: str
    prompt: str
    images: dict[str, np.ndarray]   # "image_1".. -> rgb at write_size (nearest-upscaled)
    seed: int
    step: int
    cell: tuple[int, int]


class Generator:
    name = "base"

    def generate(self, req: GenRequest) -> np.ndarray:
        """Return the new tile as (write_size, write_size, 3) uint8 rgb."""
        raise NotImplementedError


class FakeGenerator(Generator):
    """Deterministic tile generator for pipeline tests without a server.

    Generates in art space (flat palette blocks — exactly on-distribution
    for the pixel-art frame) then nearest-upscales to write_size, so the
    pipeline's block downscale recovers the pattern exactly. Output is a
    pure function of (seed, prompt, write_size).
    """

    name = "fake"

    def __init__(self, write_size: int, art_size: int, palette):
        self.write_size = write_size
        self.art = art_size
        self.palette = palette

    def generate(self, req: GenRequest) -> np.ndarray:
        h = hashlib.sha256(f"{req.seed}|{req.prompt}".encode()).digest()
        n = self.palette.n
        bg = h[0] % n
        fg = h[1] % n
        while fg == bg and n > 1:
            fg = (fg + 1) % n
        kind = h[2] % 4
        A = self.art
        idx = np.full((A, A), bg, dtype=np.uint8)
        q = max(1, A // 8)
        if kind == 0:      # vertical stripe
            idx[:, A // 3:2 * A // 3] = fg
        elif kind == 1:    # diamond
            d = (np.abs(np.arange(A)[:, None] - A // 2) + np.abs(np.arange(A)[None, :] - A // 2))
            idx[d <= A // 3] = fg
        elif kind == 2:    # quadrants
            idx[:A // 2, :A // 2] = fg
            idx[A // 2:, A // 2:] = fg
        else:              # border frame
            idx[:q, :] = fg
            idx[-q:, :] = fg
            idx[:, :q] = fg
            idx[:, -q:] = fg
        return nearest_resize(self.palette.decode(idx), self.write_size)


class ComfyGenerator(Generator):
    """Renders tiles through a ComfyUI workflow copy per arm (SPEC §2.4)."""

    name = "comfy"

    def __init__(self, client: ComfyClient, cfg: Config, out_root: str | Path):
        self.client = client
        self.cfg = cfg
        self.out_root = Path(out_root)

    def generate(self, req: GenRequest) -> np.ndarray:
        cfg, spec = self.cfg, self.cfg.arm_spec(req.arm)
        ws = cfg.write_size
        overrides: dict[str, dict] = {}
        if spec.prompt_node:
            overrides[spec.prompt_node] = {spec.prompt_field: req.prompt}
        if spec.size_node:
            if spec.size_via == "empty_latent":
                overrides[spec.size_node] = {"width": ws, "height": ws}
            elif spec.size_via == "resolution_selector":
                mp = (ws / 1024) ** 2
                overrides[spec.size_node] = {"megapixels": mp, **spec.size_extra}
            else:
                raise ValueError(f"unknown size_via {spec.size_via!r} for arm {req.arm!r}")
        if spec.seed_node:
            overrides[spec.seed_node] = {"seed": req.seed}
        for slot, arr in req.images.items():
            node = spec.image_nodes.get(slot)
            if node is None:
                raise ValueError(
                    f"arm {req.arm!r} needs {slot} but {spec.workflow} has no such slot wired "
                    f"(ArmSpec.image_nodes: {spec.image_nodes})")
            r, c = req.cell
            name = f"sdca_{req.step:05d}_{r}_{c}_{slot}.png"
            Path(self.out_root / "uploads").mkdir(parents=True, exist_ok=True)
            local = self.out_root / "uploads" / name
            Image.fromarray(arr, mode="RGB").save(local)
            server_name = self.client.upload(local, as_name=name)
            overrides[node] = {"image": server_name}
        sub = self.out_root / f"step_{req.step:05d}" / f"r{req.cell[0]:02d}c{req.cell[1]:02d}"
        paths = self.client.run(spec.workflow, overrides, sub)
        if not paths:
            raise RuntimeError(f"no output image from {spec.workflow}")
        return np.asarray(Image.open(paths[0]).convert("RGB"))


def make_generator(cfg: Config, palette) -> Generator:
    if cfg.generator == "fake":
        return FakeGenerator(cfg.write_size, cfg.art_size, palette)
    if not cfg.comfy_url:
        raise ValueError("generator=comfy requires comfy_url in config")
    return ComfyGenerator(ComfyClient(cfg.comfy_url), cfg, cfg.images_out_dir)


@dataclass
class StepReport:
    step: int
    cells: list[tuple[int, int]]
    changed: list[tuple[int, int]]
    metrics: StepMetrics | None
    errors: list[tuple[tuple[int, int], str]] = field(default_factory=list)
    elapsed_s: float = 0.0


class Engine:
    def __init__(self, cfg: Config, field: Field, rule: RuleBackend,
                 generator: Generator, schedule: Schedule | None = None,
                 state_log: StateLog | None = None,
                 metrics_log: MetricsLog | None = None,
                 start_step: int = 0):
        cfg.validate()
        self.cfg = cfg
        self.field = field
        self.rule = rule
        self.generator = generator
        self.schedule = schedule or make_schedule(cfg.schedule, cfg.grid_n, cfg.K,
                                                 seed=cfg.seed)
        self.state_log = state_log
        self.metrics_log = metrics_log
        self.step_index = start_step
        self.frame = frame_text(cfg.art_size, cfg.palette_n,
                                palette_hex=field.palette.to_hex())
        base_seed = cfg.seed if cfg.seed is not None else int.from_bytes(
            np.random.bytes(8), "big")
        self.base_seed = base_seed

    # -- seeds ----------------------------------------------------------- #
    def seed_for(self, step: int, r: int, c: int) -> int:
        if self.cfg.seed_mode == "deterministic":
            h = hashlib.sha256(f"{self.base_seed}|{step}|{r}|{c}".encode()).digest()
            return int.from_bytes(h[:8], "big") % (2**32)
        return int.from_bytes(np.random.bytes(8), "big") % (2**32)

    # -- one step --------------------------------------------------------- #
    def step(self) -> StepReport:
        cfg = self.cfg
        t0 = time.time()
        step = self.step_index
        cells = self.schedule.select(self.field, step)
        snap = self.field.snapshot()

        results: dict[tuple[int, int], np.ndarray] = {}
        errors: list[tuple[tuple[int, int], str]] = []
        if cfg.concurrency > 1 and len(cells) > 1:
            with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
                futs = {ex.submit(self._generate_cell, snap, r, c, step): (r, c)
                        for r, c in cells}
                for fut in as_completed(futs):
                    rc = futs[fut]
                    try:
                        results[rc] = fut.result()
                    except Exception as e:  # per-cell failure: log, continue step
                        errors.append((rc, f"{type(e).__name__}: {e}"))
        else:
            for r, c in cells:
                try:
                    results[(r, c)] = self._generate_cell(snap, r, c, step)
                except Exception as e:
                    errors.append(((r, c), f"{type(e).__name__}: {e}"))

        # write back in schedule order (deterministic), only changed cells logged
        changed: list[tuple[int, int]] = []
        for r, c in cells:
            if (r, c) in results and self.field.set_cell(r, c, results[(r, c)]):
                changed.append((r, c))

        if self.state_log and changed:
            self.state_log.write_step(step, changed)
        m = compute(self.field, len(changed), step, cfg.uniformity_warn)
        if self.metrics_log:
            self.metrics_log.append(m)
        if self.state_log and cfg.history != "none":
            self.state_log.save_snapshot(step)
        self.step_index += 1
        return StepReport(step=step, cells=cells, changed=changed, metrics=m,
                          errors=errors, elapsed_s=time.time() - t0)

    # -- per-cell pipeline ------------------------------------------------- #
    def _generate_cell(self, snap: Field, r: int, c: int, step: int) -> np.ndarray:
        cfg = self.cfg
        tiles = snap.neighborhood(r, c)
        composite = render_cross(tiles, cfg.read_size)
        req = RuleRequest(composite=composite, step=step, cell=(r, c), arm=cfg.arm,
                          frame=self.frame, instruction=cfg.instruction)
        res = self.rule.rule(req)

        images: dict[str, np.ndarray] = {}
        if cfg.arm == "edit_self":
            images["image_1"] = nearest_resize(tiles["center"], cfg.write_size)
        elif cfg.arm == "edit_neighborhood":
            order = {"image_1": "center", "image_2": "north", "image_3": "east",
                     "image_4": "south", "image_5": "west"}
            for slot, name in order.items():
                images[slot] = nearest_resize(tiles[name], cfg.write_size)

        gen_req = GenRequest(arm=cfg.arm, prompt=res.prompt, images=images,
                             seed=self.seed_for(step, r, c), step=step, cell=(r, c))
        out = self.generator.generate(gen_req)
        return block_mode_downscale(out, cfg.art_size, self.field.palette)
