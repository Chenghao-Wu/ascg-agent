"""fastMD layer of the multistate IBI pipeline: unit conventions, table / data files, and the
energy of a CG configuration in fastMD against an independent numpy evaluation (1-2 and 1-3 pairs excluded
from the pair table through the zero-energy exclusion bonds). The fastMD comparison needs FASTMD_BINARY
and a GPU."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ascg_core.ibi import fastmd_io as fio
from ascg_core.ibi.selfcheck import fastmd_exclusion_energy_check, toy_potentials, toy_system
from ascg_core.ibi.tables import write_tables


def test_units_and_constants():
    """Units: k_B T(400 K) = 3.325785, k_B T(500 K) = 4.157231 kJ/mol, 1 atm = 0.061019 kJ/mol/nm^3."""
    assert fio.kT(400) == pytest.approx(3.325785, abs=1e-6)
    assert fio.kT(500) == pytest.approx(4.157231, abs=1e-6)
    assert fio.ATM_KJ_MOL_NM3 == pytest.approx(0.061019, abs=1e-6)
    assert fio.ATM_KJ_MOL_NM3 * fio.BAR_PER_KJ_MOL_NM3 == pytest.approx(1.01325, rel=1e-6)


def _read_table(path, keyword):
    lines = Path(path).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.split()[:1] == [keyword])
    n = int(lines[i + 1].split()[1])
    rows = [l.split() for l in lines[i + 2:] if l.strip()][:n]
    return np.array([[float(v) for v in r[1:4]] for r in rows])


def test_tables_and_data_files(tmp_path):
    sysm, pos, box = toy_system()
    pots = toy_potentials()
    pair_e, bond_e, *_ = write_tables(tmp_path/'tables', sysm, pots)
    assert [e[3] for e in pair_e] == ['PAIR_A-A'] and [e[2] for e in bond_e] == ['BOND_A-A', fio.EXCLUSION_BOND_KEYWORD]
    t = _read_table(tmp_path/'tables'/'pair.table', 'PAIR_A-A')
    np.testing.assert_allclose(t[:, 1], pots['pair']['A-A'].u, atol=1e-8)
    inner = slice(5, -5)
    # force column = -dE/dx (1 % catches F vs F/r or per-degree errors; central differences are themselves
    # ~0.2 % off where the LJ curvature is large)
    soft = (t[:, 0] > 0.3)[inner]
    fd = -np.gradient(t[:, 1], t[:, 0])[inner][soft]
    np.testing.assert_allclose(t[inner, 2][soft], fd, rtol=1e-2, atol=1e-3 * np.abs(fd).max())
    a = _read_table(tmp_path/'tables'/'angle.table', 'ANGLE_A-A-A')
    assert a[0, 0] == 0.0 and a[-1, 0] == pytest.approx(180.0)
    np.testing.assert_allclose(a[inner, 2], -np.gradient(a[:, 1], a[:, 0])[inner], rtol=1e-2, atol=1e-4)  # per degree
    ex = _read_table(tmp_path/'tables'/'bond.table', fio.EXCLUSION_BOND_KEYWORD)
    assert np.all(ex[:, 1:] == 0)
    sysm.write_data(tmp_path/'s.data', pos, box, velocities=fio.maxwell_velocities(sysm.masses, 500, 1))
    txt = (tmp_path/'s.data').read_text()
    assert f'{len(sysm.bonds) + len(sysm.excl13)} bonds' in txt and '2 bond types' in txt
    assert 'Atoms # molecular' in txt and 'Velocities' in txt


@pytest.mark.skipif(not os.environ.get('FASTMD_BINARY'), reason='needs FASTMD_BINARY (fastMD build) and a GPU')
def test_fastmd_energy_matches_numpy_with_13_exclusions(tmp_path):
    """The deterministic engine gate (ascg_core.ibi.selfcheck) that every IBI job runs first."""
    r = fastmd_exclusion_energy_check(tmp_path)
    assert abs(r['e_with13'] - r['e_ref']) > 1.0                 # the 1-3 pairs matter in this configuration
    assert r['e_fastmd'] == pytest.approx(r['e_ref'], rel=1e-4, abs=1e-3)
    assert r['net_force'] < 1e-2 and r['ok']


@pytest.mark.skipif(not os.environ.get('FASTMD_BINARY'), reason='needs FASTMD_BINARY (fastMD build) and a GPU')
def test_fastmd_cbt_dihedral_gate(tmp_path):
    """The selfcheck's CBT dihedral gate (IBI default): needs dihedral_table (fastMD PR #3) in FASTMD_BINARY."""
    from ascg_core.ibi.selfcheck import fastmd_cbt_dihedral_check
    r = fastmd_cbt_dihedral_check(tmp_path)
    assert 'error' not in r, r.get('error')
    assert abs(r['e_plain'] - r['e_ref']) > 1.0
    assert r['e_fastmd'] == pytest.approx(r['e_ref'], rel=1e-4, abs=1e-3) and r['ok']


@pytest.mark.skipif(not os.environ.get('FASTMD_BINARY'), reason='needs FASTMD_BINARY (fastMD build) and a GPU')
def test_fastmd_npt_stability_gate(tmp_path):
    """The selfcheck's NPT gate: a CG liquid of small stiff molecules keeps its volume (fastMD PR #4)."""
    from ascg_core.ibi.selfcheck import fastmd_npt_stability_check
    r = fastmd_npt_stability_check(tmp_path)
    assert 'error' not in r, r.get('error')
    assert r['ok'] and r['volume_nm3'] > 172 and r['pressure_sd_bar'] < 1000


def test_schedule_ends_on_a_complete_cycle():
    """Stage 2 (pair:3,bond:1,angle:1,dihedral:1) is rounded up so its last six iterations are pair, pair, pair,
    bond, angle, dihedral; without dihedrals the dihedral family drops out."""
    from ascg_core.ibi.runner import _schedule, whole_cycles
    spec = 'pair:3,bond:1,angle:1,dihedral:1'
    n = whole_cycles(spec, 40)
    assert n == 42 and _schedule(spec, n)[-6:] == ['pair', 'pair', 'pair', 'bond', 'angle', 'dihedral']
    assert _schedule(spec, n)[15:18] == ['bond', 'angle', 'dihedral']          # iterations 16-18 (anchored at 1)
    no_dih = ('pair', 'bond', 'angle')
    assert whole_cycles(spec, 40, no_dih) == 40 and _schedule(spec, 40, no_dih)[-5:] == [
        'pair', 'pair', 'pair', 'bond', 'angle']
    assert whole_cycles('pair', 15) == 15 and set(_schedule('pair', 15)) == {'pair'}


def test_multistate_log_ratio_update():
    """IBI update: lambda sum_s w_s kT_s clip(ln(g_CG/g_AA), -c, c) on the supported bins, held at the edges."""
    from ascg_core.ibi.update import log_ratio_correction
    r = np.linspace(0.005, 1.495, 150)
    g_aa = np.where(r < 0.3, 0.0, 1.0 + 0.3 * np.exp(-((r - 0.5) / 0.05)**2))
    kts = {400: fio.kT(400), 500: fio.kT(500)}
    w = {400: 0.5, 500: 0.5}
    corr, sup = log_ratio_correction(r, {400: g_aa, 500: g_aa}, {400: g_aa, 500: g_aa}, kts, w, lam=0.2, cap=2.0,
                                     floor_rel=1e-3, smooth_bins=0)
    assert np.allclose(corr, 0) and not sup[r < 0.3].any() and sup[r > 0.31].all()
    g400, g500 = g_aa * np.e**0.5, g_aa * np.e**-1.0                # ln ratios +0.5 and -1
    corr, _ = log_ratio_correction(r, {400: g400, 500: g500}, {400: g_aa, 500: g_aa}, kts, w, lam=0.2, cap=2.0,
                                   floor_rel=1e-3, smooth_bins=0)
    expect = 0.2 * (0.5 * kts[400] * 0.5 + 0.5 * kts[500] * -1.0)
    assert np.allclose(corr, expect)                              # incl. the held core region
    corr, _ = log_ratio_correction(r, {400: g_aa * np.e**5, 500: g_aa * np.e**5}, {400: g_aa, 500: g_aa}, kts, w,
                                   lam=1.0, cap=2.0, floor_rel=1e-3, smooth_bins=0)
    assert np.allclose(corr, 2.0 * (0.5 * kts[400] + 0.5 * kts[500]))   # capped at c = 2
    with pytest.raises(ValueError):
        log_ratio_correction(r, {400: g_aa}, {400: g_aa}, kts, {400: 0.7}, lam=0.2, cap=2, floor_rel=1e-3, smooth_bins=0)


def test_pair_correction_keeps_cutoff_and_reith_sign():
    from ascg_core.ibi.update import apply_correction, pressure_ramp, reith_amplitude
    pot = toy_potentials(1.5)['pair']['A-A']
    r_obs = np.linspace(0.205, 1.495, 130)
    new = apply_correction(pot, r_obs, np.full(130, 0.7), r_switch=1.3, rc=1.5)
    assert new.u[-1] == pytest.approx(pot.u[-1], abs=1e-9)        # zero change at rc
    assert np.allclose((new.u - pot.u)[pot.x < 1.3], 0.7)
    kts = {400: fio.kT(400), 500: fio.kT(500)}
    a_hi = reith_amplitude({400: 3000.0, 500: 2000.0}, kts, {400: 0.5, 500: 0.5})
    assert a_hi == pytest.approx(-0.1 * (0.5 * kts[400] * min(1, 0.9) + 0.5 * kts[500] * min(1, 0.6)), rel=1e-3)
    assert reith_amplitude({400: -500.0, 500: -500.0}, kts, {400: 0.5, 500: 0.5}) > 0
    ramp = pressure_ramp(pot, a_hi, r_switch=1.3, rc=1.5)
    assert np.all((ramp.u - pot.u)[pot.x < 1.49] < 0) and ramp.u[-1] == pytest.approx(pot.u[-1], abs=1e-9)


def test_restart_round_trip(tmp_path):
    sysm, pos, box = toy_system()
    pos = pos + np.array([2.5, 0, 0])                             # some beads beyond the box: image flags
    v = fio.maxwell_velocities(sysm.masses, 400, 3)
    sysm.write_data(tmp_path/'s.data', pos, box, velocities=v)
    txt = (tmp_path/'s.data').read_text().replace('Atoms # molecular', 'Atoms # full')
    lines, out, sect = txt.splitlines(), [], None
    for l in lines:                                              # molecular -> full (q = 0), like fastMD restarts
        if l.startswith('Atoms'):
            sect = 'atoms'
        elif l and l[0].isalpha():
            sect = None
        w = l.split()
        out.append(' '.join(w[:3] + ['0'] + w[3:]) if sect == 'atoms' and len(w) == 9 else l)
    (tmp_path/'r.data').write_text('\n'.join(out) + '\n')
    x, b, vel = fio.read_restart(tmp_path/'r.data')
    np.testing.assert_allclose(x, pos, atol=1e-6)
    np.testing.assert_allclose(b, box)
    np.testing.assert_allclose(vel, v, atol=1e-6)


def test_tdep_ab_algebra():
    """U(x, T) = A + B T: per-state corrections map onto dA, dB that reproduce each state's change exactly."""
    from ascg_core.ibi.tables import TPotential, solve_ab
    from ascg_core.ibi.update import apply_tcorrection, reith_amplitudes, tpressure_ramp
    x = np.linspace(0.2, 0.7, 101)
    u400, u500 = 3000 * (x - 0.38)**2, 2500 * (x - 0.39)**2
    a, b = solve_ab({400: u400, 500: u500})
    tp = TPotential('bond', x, a, b)
    np.testing.assert_allclose(tp.at(400, cap=1e9).u, u400, atol=1e-9)
    np.testing.assert_allclose(tp.at(500, cap=1e9).u, u500, atol=1e-9)
    np.testing.assert_allclose(tp.at(450, cap=1e9).u, 0.5 * (u400 + u500), atol=1e-9)    # linear in T
    xo = np.linspace(0.25, 0.65, 41)
    c400, c500 = 0.3 * np.sin(10 * xo), -0.2 * np.cos(7 * xo)
    new = apply_tcorrection(tp, xo, {400: c400, 500: c500})
    from scipy.interpolate import CubicSpline
    xs = np.clip(x, xo[0], xo[-1])
    np.testing.assert_allclose(new.at(400, cap=1e9).u - u400, CubicSpline(xo, c400)(xs), atol=1e-9)
    np.testing.assert_allclose(new.at(500, cap=1e9).u - u500, CubicSpline(xo, c500)(xs), atol=1e-9)
    # per-state pressure ramps: each state gets its own amplitude, zero at rc
    r = np.round(np.arange(0.2, 1.5 + 1e-9, 0.002), 10)
    pp = TPotential('pair', r, np.zeros_like(r), np.zeros_like(r))
    amp = reith_amplitudes({400: 2000.0, 500: -300.0}, {400: fio.kT(400), 500: fio.kT(500)})
    assert amp[400] < 0 < amp[500]
    ramped = tpressure_ramp(pp, amp, r_switch=1.3, rc=1.5)
    k = np.searchsorted(r, 0.5)
    assert ramped.at(400).u[k] == pytest.approx(amp[400] * (1 - r[k] / 1.5), rel=1e-9)
    assert ramped.at(500).u[k] == pytest.approx(amp[500] * (1 - r[k] / 1.5), rel=1e-9)
    assert ramped.at(400).u[-1] == pytest.approx(0, abs=1e-12) and ramped.at(500).u[-1] == pytest.approx(0, abs=1e-12)


def test_tdep_bi_start_matches_per_temperature_inversion(tmp_path):
    """bi_tpotentials: U(x, T_s) equals the Boltzmann inversion of state s alone (cg_tables, single T)."""
    from ascg_core.ibi.base_tables import bonded_table, pair_table
    from ascg_core.ibi.tables import bi_tpotentials
    sysm, _, _ = toy_system()
    r = np.linspace(0.005, 1.495, 150)
    edges = np.arange(0.2, 0.6001, 0.002)
    rb = 0.5 * (edges[1:] + edges[:-1])
    th = np.deg2rad(np.arange(1.0, 180.0, 2.0))
    for T, (r0, k) in {400: (0.48, 0.07), 500: (0.50, 0.08)}.items():
        kt = fio.kT(T)
        pb = rb**2 * np.exp(-3000 * (rb - 0.38)**2 / kt)
        pa = np.sin(th) * np.exp(-8 * (th - 2.0)**2 / kt)
        np.savez(tmp_path/f'targets_T{T}.npz', r_nm=r, **{
            'rdf/A-A': np.where(r < 0.3, 0.0, 1 + 0.4 * np.exp(-((r - r0) / k)**2)),
            'bond_edges/A-A': edges, 'bond/A-A': pb / (pb.sum() * 0.002), 'angle_rad': th,
            'angle/A-A-A': pa / (pa.sum() * np.deg2rad(2.0))})
    tp = bi_tpotentials(tmp_path, sysm, (400, 500))
    for T in (400, 500):
        t = np.load(tmp_path/f'targets_T{T}.npz')
        ref = pair_table(r, {T: t['rdf/A-A']})
        x = ref.x0 + ref.dx * np.arange(len(ref.value))
        sel = (x > 0.35) & (x < 1.5)
        np.testing.assert_allclose(np.interp(x[sel], tp['pair']['A-A'].x, tp['pair']['A-A'].a + T * tp['pair']['A-A'].b),
                                   ref.value[sel], atol=1e-6)
        rb_ref = bonded_table(rb, {T: t['bond/A-A']}, kind='bond', dx=0.0005, x_range=(0.0, edges[-1] + 0.3))
        xb = rb_ref.x0 + rb_ref.dx * np.arange(len(rb_ref.value))
        sel = (xb > 0.3) & (xb < 0.46)
        np.testing.assert_allclose(np.interp(xb[sel], tp['bond']['A-A'].x, tp['bond']['A-A'].a + T * tp['bond']['A-A'].b),
                                   rb_ref.value[sel], atol=1e-6)
    assert np.abs(tp['pair']['A-A'].b[(tp['pair']['A-A'].x > 0.4) & (tp['pair']['A-A'].x < 0.6)]).max() > 1e-4


def test_harmonic_bond_walls_outside_the_sampled_range():
    """Outside the AA-sampled bond range the table follows k/2 (b - b0)^2 (k = kT / var), continuous at the edge, and
    extends until it exceeds the energy cap; inside it is unchanged."""
    from ascg_core.ibi.tables import E_CAP_KJ, harmonic_bond_walls
    c = np.arange(0.30, 0.70, 0.002) + 0.001
    p = np.exp(-((c - 0.45) / 0.03)**2) + 0.4 * np.exp(-((c - 0.56) / 0.03)**2)     # broad, bimodal (like SBS B-S)
    p[(c < 0.36) | (c > 0.64)] = 0
    p /= p.sum() * 0.002
    x = np.arange(0.0, 0.96, 0.0005)
    u = np.where(x < 0.64, 3.0 * (x - 0.45)**2 * 100, 30.0 + 2.0 * (x - 0.64))     # weak original outer wall
    kt = fio.kT(500)
    xw, uw, info = harmonic_bond_walls(x, u, c, p, kt)
    w = p / p.sum()
    b0 = (w * c).sum()
    assert info['b0_nm'] == pytest.approx(b0) and info['k_kj_nm2'] == pytest.approx(kt / (w * (c - b0)**2).sum())
    inside = (xw > info['sampled_nm'][0] + 0.006) & (xw < info['sampled_nm'][1] - 0.006)
    np.testing.assert_allclose(uw[inside], np.interp(xw[inside], x, u))
    out = xw > info['sampled_nm'][1] + 0.01
    curv = np.diff(uw[out], 2) / (xw[1] - xw[0])**2
    np.testing.assert_allclose(curv, info['k_kj_nm2'], rtol=1e-3)                  # harmonic: U'' = k
    assert uw[-1] >= E_CAP_KJ and xw[-1] > x[-1]
    assert np.abs(np.diff(uw)).max() < 5.0                                           # no jump at the edges


def test_broken_bond_gate():
    from ascg_core.ibi.analysis import BrokenBondError, break_lengths, check_bonds
    from ascg_core.ibi.tables import Potential
    x = np.arange(0.2, 1.0, 0.0005)
    pots = {'A-B': Potential('bond', x, 3000 * (x - 0.45)**2)}
    lim = break_lengths(pots, 500)                                  # 50 k_B T at 500 K = 207.9 kJ/mol
    assert lim['A-B'] == pytest.approx(0.45 + np.sqrt(50 * 0.00831446261815324 * 500 / 3000), abs=1e-3)
    assert break_lengths(pots, 300)['A-B'] < lim['A-B']              # the same table breaks earlier at 300 K
    check_bonds({'bond_max_nm': {'A-B': 0.5}}, lim, 'ok')
    with pytest.raises(BrokenBondError, match='A-B longest 1.20 nm'):
        check_bonds({'bond_max_nm': {'A-B': 1.2}}, lim, 'run_T500')


def test_dihedral_module():
    """Dihedral enumeration, the T-dependent BI start (U(T_s) = -kT_s ln P_s up to a constant) and the periodic
    per-state corrections mapped exactly onto A and B."""
    from ascg_core.ibi import dihedrals as dm
    sysm, pos, box = toy_system(n_chains=4, length=6)
    sysm.add_dihedrals()
    assert len(sysm.dihedrals) == 4 * 3 and sysm.dihedral_types == ['A-A-A-A']
    assert set(map(tuple, sysm.dihedrals)) == {(c * 6 + k, c * 6 + k + 1, c * 6 + k + 2, c * 6 + k + 3)
                                               for c in range(4) for k in range(3)}
    e = dm.edges()
    centres = 0.5 * (e[1:] + e[:-1])
    kts = {400: fio.kT(400), 500: fio.kT(500)}
    tg = {}
    for T, (mu, s) in {400: (np.pi, 0.6), 500: (np.pi, 0.9)}.items():
        d = np.angle(np.exp(1j * (centres - mu)))
        p = np.exp(-0.5 * (d / s)**2) + 0.3 * np.exp(-0.5 * ((d + 2.0) / 0.4)**2) + 0.3 * np.exp(-0.5 * ((d - 2.0) / 0.4)**2)
        tg[T] = {'phi': centres, 'dihedral/A-A-A-A': p / (p.sum() * (e[1] - e[0]))}
    tp = dm.bi_tdihedrals(tg, ['A-A-A-A'], kts, smooth_bins=0)['A-A-A-A']
    for T in (400, 500):
        u = tp.at(T).u
        assert u[0] == pytest.approx(u[-1], abs=1e-9)                                    # periodic
        w = -kts[T] * np.log(tg[T]['dihedral/A-A-A-A'])
        np.testing.assert_allclose(np.interp(centres, tp.x, u) - (w - w.min()), 0, atol=0.05)
    corr = {T: dm.state_correction(centres, tg[T]['dihedral/A-A-A-A'] * np.exp(0.3 * np.cos(centres)),
                                   tg[T]['dihedral/A-A-A-A'], kts[T], lam=1.0, cap=2.0, smooth_bins=0) for T in (400, 500)}
    for T in (400, 500):
        np.testing.assert_allclose(corr[T], kts[T] * 0.3 * np.cos(centres), atol=1e-9)
    new = dm.apply_tdihedral(tp, centres, corr)
    for T in (400, 500):
        d = new.at(T).u - tp.at(T).u
        np.testing.assert_allclose(np.interp(centres, tp.x, d), corr[T], atol=0.01 * kts[T])
    pot = new.at(450)
    pot.write  # periodic table writes: first and last rows coincide
    import tempfile
    with tempfile.TemporaryDirectory() as dd:
        pot.write(Path(dd)/'d.table', 'D')
        t = _read_table(Path(dd)/'d.table', 'D')
    assert t[0, 0] == -180 and t[-1, 0] == 180 and t[0, 1] == t[-1, 1] and t[0, 2] == t[-1, 2]


def test_dihedral_cbt_style():
    """CBT dihedrals: the prefactor, the gain-scaled and P_AA-centred BI start and corrections, and the 'cbt' option
    in the table entries / fastMD config."""
    from ascg_core.ibi import dihedrals as dm
    from ascg_core.ibi.tables import write_tables
    sysm, pos, box = toy_system(n_chains=2, length=5)
    sysm.add_dihedrals()
    assert sysm.dihedral_style == 'cbt' and dm.style_of(True) == 'table' and dm.style_of(None) is None
    q = np.array([[0, 1, 2, 3]])
    bent = np.array([[1.0, 0, 0], [0, 0, 0], [0, 1.0, 0], [0, 1.0, 1.0]])           # both angles 90 deg
    assert dm.cbt_prefactor(bent, q)[0] == pytest.approx(1.0)
    straight = np.array([[0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [2.0, 1.0, 0]])
    assert dm.cbt_prefactor(straight, q)[0] == pytest.approx(0.0, abs=1e-12)
    e = dm.edges()
    centres = 0.5 * (e[1:] + e[:-1])
    kts = {400: fio.kT(400), 500: fio.kT(500)}
    tg = {}
    for T, (s, w) in {400: (0.6, 0.05), 500: (0.9, 0.25)}.items():
        d = np.angle(np.exp(1j * (centres - np.pi)))
        p = np.exp(-0.5 * (d / s)**2) + 0.2
        tg[T] = {'phi': centres, 'dihedral/X': p / (p.sum() * (e[1] - e[0])), 'dihedral_w/X': w}
    assert dm.cbt_gain(tg[400], 'X') == pytest.approx(1 / dm.W_FLOOR) and dm.cbt_gain(tg[500], 'X') == pytest.approx(4.0)
    plain = dm.bi_tdihedrals(tg, ['X'], kts, smooth_bins=0)['X']
    cbt = dm.bi_tdihedrals(tg, ['X'], kts, smooth_bins=0, style='cbt')['X']
    for T in (400, 500):
        uc, up = np.interp(centres, cbt.x, cbt.at(T).u), np.interp(centres, plain.x, plain.at(T).u)
        p = tg[T]['dihedral/X']
        assert np.sum(p * uc) / np.sum(p) == pytest.approx(0.0, abs=0.02 * kts[T])          # centred on P_AA
        np.testing.assert_allclose(np.ptp(uc), dm.cbt_gain(tg[T], 'X') * np.ptp(up), rtol=1e-3)
    cg = tg[500]['dihedral/X'] * np.exp(0.3 * np.cos(centres))
    c0 = dm.state_correction(centres, cg, tg[500]['dihedral/X'], kts[500], lam=1.0, cap=2.0, smooth_bins=0)
    c1 = dm.state_correction(centres, cg, tg[500]['dihedral/X'], kts[500], lam=1.0, cap=2.0, smooth_bins=0, gain=4.0)
    p = tg[500]['dihedral/X']
    np.testing.assert_allclose(c1, 4.0 * c0 - np.sum(p * 4.0 * c0) / np.sum(p), atol=1e-12)
    sysm.dihedral_types = ['X']
    sysm.dihedral_type = np.zeros(len(sysm.dihedrals), int)
    pots = toy_potentials()
    pots['dihedral'] = {'X': cbt.at(450)}
    import tempfile
    with tempfile.TemporaryDirectory() as dd:
        entries = write_tables(dd, sysm, pots)
        assert entries[3][0][3:] == ('cbt',)
        conf = fio.conf_text(data_file='s.data', ntypes=1, rc=1.2, skin=0.1, dt=0.002, nsteps=1, ensemble=('nve',),
                             pair_tables=entries[0], dihedral_tables=entries[3])
        assert [l for l in conf.splitlines() if l.startswith('dihedral_table')] == [
            f'dihedral_table 0 {entries[3][0][1]} DIHEDRAL_X degrees cbt']
        sysm.dihedral_style = 'table'
        assert write_tables(dd, sysm, pots)[3][0][3:] == ()


def test_virial_pressure_correction():
    """First-order virial response: dP = -1 / (3 V) sum h r dU'; the ramp amplitude cancels scale x the deviation
    (including the predicted change of the same iteration's pair step) and is capped."""
    from ascg_core.ibi.update import ramp_shape, virial_amplitudes, virial_pressure_change
    edges = np.linspace(0, 1.5, 751)
    r = 0.5 * (edges[1:] + edges[:-1])
    hist = {'A-A': np.where(np.abs(r - 1.0) < 1e-3, 100.0, 0.0)}           # 100 pairs at r = 1 nm
    dp = virial_pressure_change(hist, edges, 8.0, {'A-A': lambda x: np.full_like(x, 2.0)})
    assert dp == pytest.approx(-100 * 1.0 * 2.0 / (3 * 8.0) * fio.BAR_PER_KJ_MOL_NM3, rel=1e-3)
    s, ds = ramp_shape(r, r_switch=1.3, rc=1.5)
    np.testing.assert_allclose(ds[5:-5], np.gradient(s, r)[5:-5], atol=2e-3)
    kts = {400: fio.kT(400), 500: fio.kT(500)}
    hs = {T: {'A-A': np.where((r > 0.4) & (r < 1.5), 50.0, 0.0)} for T in kts}
    amp, resp = virial_amplitudes({400: 300.0, 500: -100.0}, {400: 50.0}, hs, edges, {400: 8.0, 500: 8.0}, kts,
                                  r_switch=1.3, rc=1.5, scale=0.8, cap_kt=10.0)
    assert resp[400] > 0 and amp[400] < 0 < amp[500]
    assert amp[400] * resp[400] == pytest.approx(0.8 * (1.01325 - 300.0 - 50.0), rel=1e-6)
    capped, _ = virial_amplitudes({400: 1e6, 500: 1.0}, {}, hs, edges, {400: 8.0, 500: 8.0}, kts, r_switch=1.3,
                                  rc=1.5, cap_kt=0.3)
    assert capped[400] == pytest.approx(-0.3 * kts[400])


def test_protocol_outcome_classes(tmp_path):
    """Outcome classes from validation.json and the in-loop convergence check (protocol v1)."""
    from ascg_core.ibi.pipeline import classify, converged
    def val(d4, d5, rdf, rg):
        st = lambda d: {'density_error_mean': d, 'rdf_nrmse_worst': rdf, 'pass_all_seeds': abs(d) <= 0.05 and rdf <= 0.15,
                        'aa': {'rg_nm': 1.0}, 'seeds': [{'rg_nm': rg}]}
        return {'states': {'400': st(d4), '500': st(d5)}}
    assert classify(val(-0.01, 0.02, 0.03, 1.05))[0] == 'validated'
    assert classify(val(-0.01, 0.02, 0.03, 1.25))[0] == 'needs expert input'
    assert classify(val(-0.07, 0.02, 0.03, 1.0))[0] == 'failed under budget'
    assert classify({'failed': 'broken bonds'})[0] == 'failed under budget'
    wf = tmp_path/'workflow'
    wf.mkdir()
    h = {'iteration': 42, 'density_error_400': -0.03, 'density_error_500': 0.045, 'rdf_400': {'A-A': 0.02},
         'rdf_500': {'A-A': 0.03}}
    (wf/'state.json').write_text(json.dumps({'completed': 42, 'history': [h]}))
    ok, why = converged(tmp_path)
    assert not ok and '500 K density' in why
    h['density_error_500'] = 0.02
    (wf/'state.json').write_text(json.dumps({'completed': 42, 'history': [h]}))
    assert converged(tmp_path)[0]


def test_broken_bond_gate_needs_persistence():
    """Protocol v1.1: one frame just beyond the break length is an excursion, not a broken bond; two consecutive
    frames, or twice the break length in one frame, are."""
    from ascg_core.ibi.analysis import BrokenBondError, check_bonds
    lim = {'A-A': 0.338}
    check_bonds({'bond_max_nm': {'A-A': 0.341}, 'bond_persistent_over': {'A-A': 0}}, lim, 'excursion')
    with pytest.raises(BrokenBondError):
        check_bonds({'bond_max_nm': {'A-A': 0.36}, 'bond_persistent_over': {'A-A': 1}}, lim, 'persistent')
    with pytest.raises(BrokenBondError):
        check_bonds({'bond_max_nm': {'A-A': 0.70}, 'bond_persistent_over': {'A-A': 0}}, lim, 'catastrophic')
    with pytest.raises(BrokenBondError):                                    # v1 rule without per-frame data
        check_bonds({'bond_max_nm': {'A-A': 0.341}}, lim, 'v1')
