import pytest

from train_trigger import decode_fire_t, analytic_saving, fixed_interval_eval, spearman


def test_decode_fire_t():
    assert decode_fire_t([1, 2, 3], [0.1, 0.6, 0.9], tau=0.5) == 2
    assert decode_fire_t([1, 2, 3], [0.1, 0.2, 0.3], tau=0.5) is None


def test_analytic_saving_correct_fire():
    # correct fire at t_suf: H = min(L, (n - fire_t)/delta * 1000)
    # n=10, fire_t=4, delta=3 -> residual = 6/3*1000 = 2000ms, capped at L=600
    assert analytic_saving(4, t_suf=4, n=10, L=600, delta=3, c_waste=600) == 600.0


def test_analytic_saving_premature_is_penalized():
    # fire_t=2 < t_suf=5: saving = H(t_suf) - c_waste
    # H(t_suf=5): residual = (10-5)/3*1000 = 1666 -> cap 600; minus c_waste 600 = 0
    assert analytic_saving(2, t_suf=5, n=10, L=600, delta=3, c_waste=600) == 0.0
    # smaller penalty -> positive
    assert analytic_saving(2, t_suf=5, n=10, L=600, delta=3, c_waste=300) == 300.0
    # c_waste exceeds H(t_suf): saving goes NET-NEGATIVE
    # H(t_suf=9): residual = (10-9)/3*1000 = 333.3 -> minus c_waste 600 = -266.7
    assert analytic_saving(2, t_suf=9, n=10, L=600, delta=3, c_waste=600) == pytest.approx(-266.7, rel=1e-3)


def test_analytic_saving_never_fires():
    assert analytic_saving(None, t_suf=5, n=10, L=600, delta=3, c_waste=600) == 0.0


def test_fixed_interval_eval():
    # interval=2: fires at 2,4,6...; first >= t_suf=5 is 6 -> 3 calls
    assert fixed_interval_eval(t_suf=5, n=10, interval=2) == (6, 3)
    # never reaches gold within n -> None, floor(n/interval) calls
    assert fixed_interval_eval(t_suf=9, n=7, interval=2) == (None, 3)


def test_spearman_monotonic():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
