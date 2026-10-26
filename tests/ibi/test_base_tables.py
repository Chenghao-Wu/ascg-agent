import numpy as np
import pytest

from ascg_core.ibi.base_tables import KB_KJ_PER_MOL_K, bonded_table, hermite, pair_table, with_pressure_ramp


def test_hermite_reproduces_cubic_and_slope():
    x = np.linspace(0, 1, 11)
    f, df = x**3 - x, 3*x**2 - 1
    xs = np.linspace(0, 1, 101)
    assert np.allclose(hermite(xs, 0.0, 0.1, f, df), xs**3 - xs, atol=1e-8)


def test_bond_table_recovers_harmonic_at_both_temperatures():
    # P_T(r) ~ r^2 exp(-k (r-b)^2 / kT): every T inverts to the same U = k (r-b)^2.
    r = np.linspace(0.20, 0.36, 161)
    k, b = 20000.0, 0.28
    p = {T: r**2*np.exp(-k*(r - b)**2/(KB_KJ_PER_MOL_K*T)) for T in (400, 500)}
    t = bonded_table(r, p, kind='bond', smooth_bins=0)
    near = np.abs(r - b) < 0.02
    assert np.allclose(t(r[near]) - t(b), k*(r[near] - b)**2, atol=0.05)
    # walls keep rising outside the sampled range
    assert t(0.10) > t(0.20) and t(0.45) > t(0.36)


def test_angle_table_uses_sin_jacobian():
    th = np.linspace(0.05, np.pi - 0.05, 180)
    p = {400: np.sin(th), 500: np.sin(th)}          # uniform on the sphere -> flat PMF
    t = bonded_table(th, p, kind='angle', smooth_bins=0)
    assert np.ptp(t(th[10:-10])) < 1e-6


def test_pair_table_core_tail_and_temperature_average():
    r = np.linspace(0.005, 1.495, 150)
    g = np.where(r < 0.35, 0.0, 1 + 0.3*np.exp(-((r - 0.5)/0.08)**2))
    t = pair_table(r, {400: g, 500: g})
    # in the sampled region V = -kT_avg ln g with T_avg = 450 K (same g at both T)
    x = 0.5
    assert t(x) == pytest.approx(-KB_KJ_PER_MOL_K*450*np.log(1.3), rel=0.05)
    assert t(1.5) == pytest.approx(0.0, abs=1e-9) and t(1.2) == pytest.approx(0.0, abs=1e-3)
    assert t(0.2) > t(0.3) > t(0.36)                  # repulsive wall below the first sampled r
    grid = np.arange(0.2, 1.5, 0.0005)
    assert np.all(np.isfinite(t(grid)))


def test_pressure_ramp_adds_switched_linear_term_with_consistent_slope():
    r = np.linspace(0.005, 1.495, 150)
    g = np.where(r < 0.3, 0.0, 1 + 0.3 * np.exp(-((r - 0.5) / 0.08)**2))
    t = pair_table(r, {400: g, 500: g})
    A = -0.7
    c = with_pressure_ramp(t, A)
    x = t.x0 + t.dx * np.arange(len(t.value))
    d = c.value - t.value
    ok = x > 0.3                                  # the repulsive wall below is ~1e15 kJ/mol (float round-off)
    assert abs(d[-1]) < 1e-12                                             # zero at rc
    inner = ok & (x < 1.3)
    assert np.allclose(d[inner], A * (1 - x[inner] / 1.5))                # unswitched below r_switch
    num = np.gradient(d, t.dx)                                            # ramp slope consistent with values
    assert np.allclose((c.slope - t.slope - num)[ok][1:-1], 0, atol=2e-3)
    assert abs(c.slope[-1] - t.slope[-1]) < 1e-9                          # force continuous at rc
