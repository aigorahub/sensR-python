import numpy as np
import pytest
from senspy import (
    duotrio_pc,
    triangle_pc,
    pc2pd,
    pd2pc,
    get_pguess,
)


def test_duotrio_half_at_zero():
    assert duotrio_pc(0.0) == 0.5


def test_duotrio_monotonic():
    low = duotrio_pc(0.1)
    high = duotrio_pc(1.0)
    assert low < high < 1.0


def test_triangle_baseline():
    assert triangle_pc(0.0) == pytest.approx(1.0 / 3.0)


def test_pc_pd_roundtrip():
    pg = get_pguess("triangle")
    pc = 0.75
    pd = pc2pd(pc, pg)
    assert pd2pc(pd, pg) == pytest.approx(pc)
