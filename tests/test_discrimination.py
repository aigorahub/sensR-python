import numpy as np
from senspy import (
    duotrio_pc,
    get_pguess,
    pc2pd,
    pd2pc,
    discrim_2afc,
)


def test_duotrio_half_at_zero():
    assert duotrio_pc(0.0) == 0.5


def test_duotrio_monotonic():
    low = duotrio_pc(0.1)
    high = duotrio_pc(1.0)
    assert low < high < 1.0


def test_pc_pd_roundtrip():
    pg = get_pguess("twoafc")
    pc = 0.75
    pd = pc2pd(pc, pg)
    pc_back = pd2pc(pd, pg)
    assert np.isclose(pc, pc_back)


def test_get_pguess_double():
    assert np.isclose(get_pguess("twoafc", double=True), 0.25)


def test_discrim_2afc_positive():
    res = discrim_2afc(correct=30, total=40)
    assert res["d_prime"] > 0 and res["se"] > 0
