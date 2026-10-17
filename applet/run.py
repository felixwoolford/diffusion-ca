"""Headless run entrypoint: `python -m applet.run --config <yaml> [--steps N] [--headless]`.

The engine is GUI-agnostic and importable with no display (SPEC §2.1); the
GUI (applet/gui, built later on the user's machine) drives the same Engine
object. `--headless` is accepted now as a no-op so the documented CLI
already works in this container.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine.config import load_config
from .engine.grid import Field
from .engine.metrics import MetricsLog
from .engine.palette import Palette
from .engine.pipeline import Engine, make_generator
from .engine.statelog import StateLog
from .rule import make_backend


def init_field(cfg) -> tuple[Field, int]:
    """Field from (in priority order): resume snapshot, initial_field PNG,
    random. Returns (field, start_step)."""
    pal = _load_palette(cfg)
    state_root = Path(cfg.state_dir) / cfg.run_name
    latest = StateLog.latest_snapshot(state_root)
    if latest is not None:
        step, idx = latest
        field = Field(idx.shape[0], idx.shape[2], pal, idx)
        print(f"[init] resumed from snapshot at step {step} ({state_root / 'snapshot'})")
        return field, step + 1
    if cfg.initial_field:
        field = Field.from_png(cfg.initial_field, cfg.grid_n, cfg.art_size, pal)
        print(f"[init] field from {cfg.initial_field}")
    else:
        field = Field.random(cfg.grid_n, cfg.art_size, pal,
                             seed=cfg.seed if cfg.seed_mode == "deterministic" else None)
        print(f"[init] random field (n={cfg.grid_n}, art={cfg.art_size}, "
              f"palette={pal.n}, seed_mode={cfg.seed_mode})")
    return field, 0


def _load_palette(cfg) -> Palette:
    if not cfg.palette_path:
        raise SystemExit("config: palette_path is required to run "
                         "(pick the E5 palette file, e.g. config/palette_64.json)")
    return Palette.load(cfg.palette_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="applet.run",
                                 description="run SD-CA steps headless")
    ap.add_argument("--config", required=True, help="run config YAML")
    ap.add_argument("--steps", type=int, default=1, help="number of steps to run")
    ap.add_argument("--headless", action="store_true",
                    help="no display (default behavior; flag kept for the documented CLI)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg.check_runtime(needs_comfy=cfg.generator == "comfy",
                      needs_llm=cfg.rule_backend == "llamacpp")

    field, start_step = init_field(cfg)
    state_root = Path(cfg.state_dir) / cfg.run_name
    state_log = StateLog(state_root, field) if cfg.history != "none" else None
    metrics_log = MetricsLog(state_root / "metrics.csv") if cfg.history != "none" else None
    rule = make_backend(cfg)
    generator = make_generator(cfg, field.palette)
    engine = Engine(cfg, field, rule, generator,
                    state_log=state_log, metrics_log=metrics_log,
                    start_step=start_step)

    print(f"[run] arm={cfg.arm} schedule={cfg.schedule} K={cfg.K} "
          f"write={cfg.write_size} read={cfg.read_size} rule={cfg.rule_backend} "
          f"generator={cfg.generator} steps={args.steps} "
          f"comfy={cfg.comfy_url or '-'} llm={cfg.llm_url or '-'}")
    for _ in range(args.steps):
        rep = engine.step()
        m = rep.metrics
        warn = " [COLLAPSE-WARN]" if m.warn else ""
        err = f" errors={len(rep.errors)}" if rep.errors else ""
        print(f"[step {rep.step}] changed={len(rep.changed)} "
              f"nbd={m.neighbor_dist:.1f} H={m.palette_entropy:.3f} "
              f"uni={m.uniformity:.3f}{warn}{err} ({rep.elapsed_s:.1f}s)")
        for rc, e in rep.errors:
            print(f"    error at {rc}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
