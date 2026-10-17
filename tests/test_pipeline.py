"""End-to-end pipeline tests with the template rule + fake generator:
full lockstep steps, state log, snapshot resume, determinism — all headless."""
import numpy as np
import pytest

from applet.engine.config import Config
from applet.engine.grid import Field
from applet.engine.metrics import MetricsLog
from applet.engine.palette import Palette
from applet.engine.pipeline import Engine, FakeGenerator
from applet.engine.statelog import StateLog
from applet.rule import make_backend
from conftest import STANDARD


@pytest.fixture
def pal():
    return Palette(STANDARD.copy())


def make_cfg(tmp_path, arm="t2i", **kw):
    cfg = Config(
        grid_n=2, art_size=8, write_size=32, read_size=32,
        render_scale=1, palette_n=8,
        arm=arm, schedule="random_uniform", K=2,
        seed_mode="deterministic", seed=7,
        rule_backend="template", generator="fake", concurrency=4,
        state_dir=str(tmp_path), run_name="t", history="log+snapshot",
        uniformity_warn=0.9,
    )
    cfg.validate()
    return cfg


def run_steps(cfg, n=3, field=None):
    field = field or Field.random(cfg.grid_n, cfg.art_size,
                                  Palette(STANDARD.copy()), seed=cfg.seed)
    log = StateLog(cfg.state_dir + "/t", field)
    mlog = MetricsLog(cfg.state_dir + "/t/metrics.csv")
    eng = Engine(cfg, field, make_backend(cfg), FakeGenerator(cfg.write_size,
                cfg.art_size, field.palette), state_log=log, metrics_log=mlog)
    reports = [eng.step() for _ in range(n)]
    return eng, field, reports


def test_full_steps_headless(pal, tmp_path):
    cfg = make_cfg(tmp_path)
    eng, field, reports = run_steps(cfg, 3)
    assert all(len(r.errors) == 0 for r in reports)
    assert all(r.metrics is not None for r in reports)
    # every step updated K cells and (with a fresh random field) at least
    # most of them changed
    assert all(len(r.cells) == 2 for r in reports)
    total_changed = sum(len(r.changed) for r in reports)
    assert total_changed >= 2
    # state log: only changed cells stored, snapshots per step
    log = StateLog(cfg.state_dir + "/t", field)
    assert StateLog.latest_snapshot(cfg.state_dir + "/t")[0] == 2
    stored = sum(len(log.changed_cells(s)) for s in (0, 1, 2))
    assert stored == total_changed
    # metrics csv has 3 rows
    assert len(MetricsLog(cfg.state_dir + "/t/metrics.csv").read()) == 3


def test_deterministic_run_reproducible(pal, tmp_path):
    cfg1, cfg2 = make_cfg(tmp_path / "a"), make_cfg(tmp_path / "b")
    f1 = Field.random(2, 8, Palette(STANDARD.copy()), seed=7)
    f2 = Field.random(2, 8, Palette(STANDARD.copy()), seed=7)

    def run(cfg, f):
        log = StateLog(cfg.state_dir + "/t", f)
        eng = Engine(cfg, f, make_backend(cfg), FakeGenerator(32, 8, f.palette),
                     state_log=log)
        for _ in range(3):
            eng.step()
        return f.indices.copy()

    assert (run(cfg1, f1) == run(cfg2, f2)).all()


def test_resume_from_snapshot(pal, tmp_path):
    cfg = make_cfg(tmp_path)
    eng, field, _ = run_steps(cfg, 2)
    # restart: build a fresh engine from the stored snapshot
    step, idx = StateLog.latest_snapshot(cfg.state_dir + "/t")
    assert step == 1  # steps 0 and 1 were run
    resumed = Field(2, 8, Palette(STANDARD.copy()), idx.copy())
    assert (resumed.indices == field.indices).all()
    # and it can keep stepping
    log = StateLog(cfg.state_dir + "/t", resumed)
    eng2 = Engine(cfg, resumed, make_backend(cfg), FakeGenerator(32, 8, resumed.palette),
                  state_log=log, start_step=step + 1)
    rep = eng2.step()
    assert rep.step == 2  # continues after the resumed step
    assert len(rep.errors) == 0


def test_all_arms_run(pal, tmp_path):
    for i, arm in enumerate(("t2i", "edit_self", "edit_neighborhood")):
        cfg = make_cfg(tmp_path / f"arm{i}", arm=arm)
        eng, field, reports = run_steps(cfg, 1)
        assert len(reports[0].errors) == 0, (arm, reports[0].errors)
        assert len(reports[0].changed) >= 1


def test_seed_modes(pal, tmp_path):
    cfg = make_cfg(tmp_path)  # seed_mode=deterministic
    f = Field.random(2, 8, pal, seed=1)
    eng = Engine(cfg, f, make_backend(cfg), FakeGenerator(32, 8, pal))
    s1, s2 = eng.seed_for(0, 0, 0), eng.seed_for(0, 0, 0)
    assert s1 == s2  # deterministic mode: stable per (cell, step)
    assert s1 != eng.seed_for(1, 0, 0)
    # random mode: fresh seeds per call
    cfg.seed_mode = "random"
    eng2 = Engine(cfg, f, make_backend(cfg), FakeGenerator(32, 8, pal))
    assert eng2.seed_for(0, 0, 0) != eng2.seed_for(0, 0, 0)


def test_fake_generator_on_distribution(pal):
    """FakeGenerator tiles survive the block downscale exactly."""
    from applet.engine.imageops import block_mode_downscale
    from applet.engine.pipeline import GenRequest
    g = FakeGenerator(32, 8, pal)
    req = GenRequest(arm="t2i", prompt="x", images={}, seed=1, step=0, cell=(0, 0))
    out = g.generate(req)
    assert out.shape == (32, 32, 3)
    down = block_mode_downscale(out, 8, pal)
    assert (block_mode_downscale(g.generate(req), 8, pal) == down).all()
