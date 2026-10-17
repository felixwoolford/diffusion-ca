import numpy as np

from applet.engine.grid import Field
from applet.engine.metrics import (
    MetricsLog, compute, neighbor_distance, palette_entropy, uniformity)


def checkerboard(pal):
    """2x2 field, art=4: (0,0) red, (0,1) blue, (1,0) blue, (1,1) red.
    Every adjacent pair (wrapping) is red-blue."""
    f = Field(2, 4, pal)
    f.set_cell(0, 0, np.full((4, 4), 2, np.uint8))
    f.set_cell(0, 1, np.full((4, 4), 4, np.uint8))
    f.set_cell(1, 0, np.full((4, 4), 4, np.uint8))
    f.set_cell(1, 1, np.full((4, 4), 2, np.uint8))
    return f


def test_uniform_field_metrics(pal):
    f = Field(2, 4, pal)
    f.set_cell(0, 0, np.full((4, 4), 2, np.uint8))
    f.set_cell(0, 1, np.full((4, 4), 2, np.uint8))
    f.set_cell(1, 0, np.full((4, 4), 2, np.uint8))
    f.set_cell(1, 1, np.full((4, 4), 2, np.uint8))
    assert neighbor_distance(f) == 0.0
    assert palette_entropy(f) == 0.0
    assert uniformity(f) == 1.0


def test_checkerboard_neighbor_distance(pal):
    f = checkerboard(pal)
    # red-blue tile distance = art * sqrt(255^2 + 255^2)
    expected = 4 * np.sqrt(2 * 255 ** 2)
    assert neighbor_distance(f) == expected


def test_entropy_half_half(pal):
    f = Field(2, 4, pal)
    f.set_cell(0, 0, np.full((4, 4), 0, np.uint8))
    f.set_cell(0, 1, np.full((4, 4), 0, np.uint8))
    f.set_cell(1, 0, np.full((4, 4), 1, np.uint8))
    f.set_cell(1, 1, np.full((4, 4), 1, np.uint8))
    assert palette_entropy(f) == 1.0  # bits


def test_uniformity_fraction(pal):
    f = Field(2, 4, pal)
    f.set_cell(0, 0, np.full((4, 4), 2, np.uint8))
    f.set_cell(0, 1, np.full((4, 4), 2, np.uint8))
    f.set_cell(1, 0, np.full((4, 4), 3, np.uint8))
    f.set_cell(1, 1, np.full((4, 4), 4, np.uint8))
    assert uniformity(f) == 0.5


def test_compute_and_warn(pal):
    f = Field(2, 4, pal)  # all black cells -> uniformity 1
    m = compute(f, changed_cells=0, step=0, warn_threshold=0.5)
    assert m.warn == True  # noqa: E712
    m2 = compute(checkerboard(pal), changed_cells=4, step=1, warn_threshold=0.99)
    assert m2.warn == False  # noqa: E712
    assert m2.changed_cells == 4


def test_metrics_log_csv(pal, tmp_path):
    path = tmp_path / "m.csv"
    log = MetricsLog(path)
    f = checkerboard(pal)
    log.append(compute(f, 2, 0, 0.5))
    log.append(compute(f, 3, 1, 0.5))
    rows = log.read()
    assert len(rows) == 2
    assert rows[0]["step"] == "0"
    assert float(rows[1]["changed_cells"]) == 3.0
    assert rows[0]["warn"] in ("0", "1")
