import numpy as np

from applet.engine.grid import Field
from applet.engine.statelog import StateLog


def make_field(pal):
    f = Field(2, 4, pal)
    f.set_cell(0, 0, np.full((4, 4), 2, np.uint8))
    return f


def test_write_step_only_changed(pal, tmp_path):
    f = make_field(pal)
    log = StateLog(tmp_path, f)
    paths = log.write_step(0, [(0, 0), (1, 1)])
    assert len(paths) == 2
    assert log.write_step(1, []) == []
    # no-change step writes nothing
    assert not (tmp_path / "log" / "step_00001").exists()
    assert log.steps() == [0]


def test_cell_roundtrip(pal, tmp_path):
    f = make_field(pal)
    log = StateLog(tmp_path, f)
    log.write_step(7, [(0, 0)])
    idx = log.load_cell(7, 0, 0)
    assert (idx == 2).all()
    assert log.changed_cells(7) == [(0, 0)]
    assert log.changed_cells(8) == []


def test_snapshot_roundtrip_and_latest(pal, tmp_path):
    f = make_field(pal)
    log = StateLog(tmp_path, f)
    log.save_snapshot(3)
    f.set_cell(1, 1, np.full((4, 4), 5, np.uint8))
    log.save_snapshot(9)
    step, idx = StateLog.latest_snapshot(tmp_path)
    assert step == 9
    assert (idx[1, 1] == 5).all()
    assert (idx[0, 0] == 2).all()


def test_latest_snapshot_missing(pal, tmp_path):
    f = make_field(pal)
    StateLog(tmp_path, f)  # creates empty dirs
    assert StateLog.latest_snapshot(tmp_path) is None
