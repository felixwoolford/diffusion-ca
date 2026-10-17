import numpy as np
import pytest

from applet.engine.grid import Field
from applet.engine.schedule import make_schedule


def test_random_uniform_unique_and_deterministic(pal):
    f = Field(4, 4, pal)
    s1 = make_schedule("random_uniform", 4, 5, seed=42)
    s2 = make_schedule("random_uniform", 4, 5, seed=42)
    a1 = s1.select(f, 0)
    a2 = s2.select(f, 0)
    assert a1 == a2
    assert len(a1) == 5
    assert len(set(a1)) == 5
    assert all(0 <= r < 4 and 0 <= c < 4 for r, c in a1)
    # consecutive steps keep drawing deterministically
    assert s1.select(f, 1) == s2.select(f, 1)


def test_random_uniform_k_clamped(pal):
    f = Field(2, 4, pal)
    s = make_schedule("random_uniform", 2, 99, seed=0)
    assert len(s.select(f, 0)) == 4


def test_ring_scan_covers_field(pal):
    n = 4
    f = Field(n, 4, pal)
    s = make_schedule("ring_scan", n, 8)
    seen = set()
    steps = 2 * n - 1
    for step in range(steps):
        cells = s.select(f, step)
        assert len(cells) == 8
        assert len(set(cells)) == 8
        seen |= set(cells)
    assert seen == {(r, c) for r in range(n) for c in range(n)}
    # deterministic
    assert s.select(f, 3) == make_schedule("ring_scan", n, 8).select(f, 3)


def test_ring_scan_starts_on_ring0(pal):
    f = Field(4, 4, pal)
    cells = make_schedule("ring_scan", 4, 10).select(f, 0)
    assert all(r + c <= 3 for r, c in cells)  # ring 0..(until filled)


def test_all(pal):
    f = Field(3, 4, pal)
    cells = make_schedule("all", 3, 1).select(f, 0)
    assert cells == [(r, c) for r in range(3) for c in range(3)]


def test_unknown_schedule(pal):
    with pytest.raises(ValueError):
        make_schedule("nope", 4, 4)
