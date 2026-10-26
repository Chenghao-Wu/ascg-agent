"""fastMD dihedral_table (fastMD PR #3, in main since b35e30a): needs FASTMD_BINARY (or FASTMD_ASCG_BINARY) and a
GPU.

* a table of the OPLS function reproduces fastMD's own dihedral_opls energies and forces;
* an asymmetric table gives the energy of numpy's phi (IUPAC: cis 0, trans 180, sign of Blondel-Karplus), so the
  pipeline's dihedral analysis and the engine agree on the angle;
* with tabulated dihedrals 1-4 pairs still interact through the pair table (only 1-2 and the 1-3 exclusion bonds are
  excluded);
* the dihedral virial matches the numerical shear derivative of the energy;
* combined bending-torsion (`cbt`, E = sin^3 theta1 sin^3 theta2 U(phi)): energies and forces match numpy in both
  bonded modes, and stay finite (and vanish) when a bond angle is straight.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from ascg_core.ibi import fastmd_io as fio

BIN = os.environ.get('FASTMD_ASCG_BINARY') or os.environ.get('FASTMD_BINARY')
pytestmark = pytest.mark.skipif(not BIN, reason='needs FASTMD_BINARY (fastMD main b35e30a or later) and a GPU')


def coil_system(n_chains=6, length=8, box=7.0, seed=3):
    """Random-coil chains (bond 0.40 nm, angle ~110 deg, random torsions), each centred on its own grid point,
    inside the box (no wrapping)."""
    rng = np.random.default_rng(seed)
    pos, mol, bonds, angles, dihs = [], [], [], [], []
    for c in range(n_chains):
        x = [np.zeros(3)]
        d = np.array([1.0, 0.0, 0.0])
        for _ in range(1, length):
            perp = np.cross(d, rng.normal(size=3))           # random torsion
            perp /= np.linalg.norm(perp)
            th = np.deg2rad(180 - 110 + rng.normal(0, 8))
            d = np.cos(th) * d + np.sin(th) * perp
            d /= np.linalg.norm(d)
            x.append(x[-1] + 0.40 * d)
        x = np.array(x)
        centre = np.array([1.2 + 2.3 * (c % 3), 1.75 + 3.5 * (c // 3 % 2), 1.75 + 3.5 * (c // 6)])
        pos += list(x - x.mean(0) + centre)
        mol += [c] * length
        b = c * length
        bonds += [(b + k, b + k + 1) for k in range(length - 1)]
        angles += [(b + k, b + k + 1, b + k + 2) for k in range(length - 2)]
        dihs += [(b + k, b + k + 1, b + k + 2, b + k + 3) for k in range(length - 3)]
    pos = np.array(pos)
    assert pos.min() > 0 and pos.max() < box
    return pos, np.array(mol), np.array(bonds), np.array(angles), np.array(dihs), np.full(3, box)


def phi_numpy(pos, dihs, box):
    """Blondel-Karplus phi (rad), as in the pipeline's analysis."""
    mi = lambda d: d - box * np.round(d / box)
    F = mi(pos[dihs[:, 0]] - pos[dihs[:, 1]])
    G = mi(pos[dihs[:, 1]] - pos[dihs[:, 2]])
    H = mi(pos[dihs[:, 3]] - pos[dihs[:, 2]])
    A, B = np.cross(F, G), np.cross(H, G)
    gn = np.linalg.norm(G, axis=1)
    return np.arctan2((np.cross(B, A) * G).sum(1) / gn, (A * B).sum(1))


def opls(phi, k):
    return (0.5 * k[0] * (1 + np.cos(phi)) + 0.5 * k[1] * (1 - np.cos(2 * phi)) + 0.5 * k[2] * (1 + np.cos(3 * phi))
            + 0.5 * k[3] * (1 - np.cos(4 * phi)))


def dopls(phi, k):
    return (-0.5 * k[0] * np.sin(phi) + k[1] * np.sin(2 * phi) - 1.5 * k[2] * np.sin(3 * phi)
            + 2 * k[3] * np.sin(4 * phi))


def run(tmp, name, pos, mol, bonds, angles, dihs, box, *, pair=None, dihedral_line, rc=1.2, extra=(), dump=True):
    """Single-point energies / forces (forces_dump_file; fastMD exits right after it) of one configuration, or with
    dump=False a one-step run whose thermo.dat holds the step-0 pressure tensor."""
    wd = Path(tmp)/name
    wd.mkdir()
    r = np.round(np.arange(0.05, rc + 1e-9, 0.001), 10)
    up = np.zeros_like(r) if pair is None else pair(r)
    fio.write_table(wd/'t.table', 'PAIR', r, up, -np.gradient(up, r))
    b = np.arange(0.1, 1.0, 0.001)
    fio.write_table(wd/'t.table', 'ZB', b, np.zeros_like(b), np.zeros_like(b))
    th = np.arange(0.0, 180.0001, 0.5)
    fio.write_table(wd/'t.table', 'ZA', th, np.zeros_like(th), np.zeros_like(th))
    ex13 = np.array([(a, c) for a, _, c in angles])                      # 1-3 exclusion bonds (zero energy)
    fio.write_data(wd/'s.data', pos=pos, box=box, types=np.zeros(len(pos), int), type_masses=[28.0], mol=mol,
                   bonds=np.concatenate([bonds, ex13]),
                   bond_types=np.concatenate([np.zeros(len(bonds), int), np.ones(len(ex13), int)]), n_bond_types=2,
                   angles=angles, angle_types=np.zeros(len(angles), int), n_angle_types=1,
                   dihedrals=dihs, dihedral_types=np.zeros(len(dihs), int), n_dihedral_types=1)
    tf = str(wd/'t.table')
    conf = fio.conf_text(data_file='s.data', ntypes=1, rc=rc, skin=0.1, dt=1e-7, nsteps=1, ensemble=('nve',),
                         pair_tables=[(0, 0, tf, 'PAIR')], bond_tables=[(0, tf, 'ZB'), (1, tf, 'ZB')],
                         angle_tables=[(0, tf, 'ZA')], thermo=(1, 'thermo.dat'),
                         extra=(dihedral_line(wd),) + (('forces_dump_file forces.dat',) if dump else ()) + tuple(extra))
    (wd/'run.conf').write_text(conf)
    fio.run('run.conf', wd, binary=BIN)
    if not dump:
        return None, None, wd
    f = np.loadtxt(wd/'forces.dat', comments='#')
    return f[:, 1:4], float(f[:, 4].sum()), wd


def table_line(phi_deg, e, f_per_deg, cbt=False):
    def make(wd):
        fio.write_table(wd/'d.table', 'DIH', phi_deg, e, f_per_deg)
        return f'dihedral_table 0 {wd/"d.table"} DIH degrees' + (' cbt' if cbt else '')
    return make


def cbt_energy(pos, dihs, box, u):
    """Sum of sin^3 theta1 sin^3 theta2 u(phi) (theta1 = angle 1-2-3, theta2 = angle 2-3-4)."""
    mi = lambda d: d - box * np.round(d / box)
    F = mi(pos[dihs[:, 0]] - pos[dihs[:, 1]])
    G = mi(pos[dihs[:, 1]] - pos[dihs[:, 2]])
    H = mi(pos[dihs[:, 3]] - pos[dihs[:, 2]])
    n = lambda v: np.linalg.norm(v, axis=1)
    s1 = n(np.cross(F, G)) / (n(F) * n(G))
    s2 = n(np.cross(H, G)) / (n(H) * n(G))
    return (s1**3 * s2**3 * u(phi_numpy(pos, dihs, box))).sum()


def numeric_forces(energy, pos, h=1e-6):
    f = np.zeros_like(pos)
    for a in range(len(pos)):
        for c in range(3):
            p, m = pos.copy(), pos.copy()
            p[a, c] += h
            m[a, c] -= h
            f[a, c] = -(energy(p) - energy(m)) / (2 * h)
    return f


K = (1.3, -0.6, 2.1, 0.4)


def test_table_reproduces_dihedral_opls(tmp_path):
    pos, mol, bonds, angles, dihs, box = coil_system()
    phi = np.deg2rad(np.linspace(-180, 180, 3601))
    f_tab, e_tab, _ = run(tmp_path, 'table', pos, mol, bonds, angles, dihs, box,
                          dihedral_line=table_line(np.degrees(phi), opls(phi, K), -dopls(phi, K) * np.pi / 180))
    f_opl, e_opl, _ = run(tmp_path, 'opls', pos, mol, bonds, angles, dihs, box,
                          dihedral_line=lambda wd: 'dihedral_opls 0 ' + ' '.join(map(str, K)))
    e_ref = opls(phi_numpy(pos, dihs, box), K).sum()
    assert e_opl == pytest.approx(e_ref, rel=1e-4)
    assert e_tab == pytest.approx(e_ref, rel=1e-4)
    np.testing.assert_allclose(f_tab, f_opl, atol=2e-3 * np.abs(f_opl).max())


def test_asymmetric_table_matches_numpy_phi_convention(tmp_path):
    pos, mol, bonds, angles, dihs, box = coil_system(seed=5)
    phi = np.deg2rad(np.linspace(-180, 180, 3601))
    u = 2.0 * np.sin(phi) + 0.7 * np.cos(2 * phi) + 0.3 * np.sin(3 * phi)      # odd terms: sign of phi matters
    du = 2.0 * np.cos(phi) - 1.4 * np.sin(2 * phi) + 0.9 * np.cos(3 * phi)
    f, e, _ = run(tmp_path, 'asym', pos, mol, bonds, angles, dihs, box,
                  dihedral_line=table_line(np.degrees(phi), u, -du * np.pi / 180))
    p = phi_numpy(pos, dihs, box)
    assert e == pytest.approx(np.interp(p, phi, u).sum(), rel=1e-4, abs=1e-4)
    assert abs(e - np.interp(-p, phi, u).sum()) > 1.0                      # the mirrored convention would differ
    assert np.abs(f.sum(0)).max() < 1e-3


def test_one_four_pairs_keep_the_pair_table(tmp_path):
    pos, mol, bonds, angles, dihs, box = coil_system(seed=7)
    pair = lambda r: 3.0 * np.exp(-((r - 0.75) / 0.25)**2) * np.clip((1.2 - r) / 0.3, 0, 1)**2
    phi = np.deg2rad(np.linspace(-180, 180, 3601))
    _, e, _ = run(tmp_path, 'p14', pos, mol, bonds, angles, dihs, box, pair=pair,
                  dihedral_line=table_line(np.degrees(phi), np.zeros_like(phi), np.zeros_like(phi)))
    excl = {tuple(sorted(x)) for x in bonds.tolist()} | {(a, c) for a, _, c in angles.tolist()}
    i, j = np.triu_indices(len(pos), 1)
    keep = np.array([(a, b) not in excl for a, b in zip(i, j)])
    d = pos[i[keep]] - pos[j[keep]]
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=1)
    rr = np.round(np.arange(0.05, 1.2 + 1e-9, 0.001), 10)
    e_ref = np.interp(r[r < 1.2], rr, pair(rr)).sum()
    one4 = {(a, dd) for a, _, _, dd in dihs.tolist()}
    e_no14 = e_ref - np.interp(np.linalg.norm(pos[[a for a, _ in one4]] - pos[[dd for _, dd in one4]], axis=1),
                               rr, pair(rr)).sum()
    assert e == pytest.approx(e_ref, rel=1e-4)
    assert abs(e_ref - e_no14) > 1.0                                        # the 1-4 pairs matter here


@pytest.mark.parametrize('cbt', [False, True])
def test_dihedral_virial_matches_shear_derivative(tmp_path, cbt):
    pos, mol, bonds, angles, dihs, box = coil_system(seed=11)
    phi = np.deg2rad(np.linspace(-180, 180, 3601))
    line = table_line(np.degrees(phi), opls(phi, K), -dopls(phi, K) * np.pi / 180, cbt=cbt)
    _, _, wd = run(tmp_path, 'v0', pos, mol, bonds, angles, dihs, box, dihedral_line=line, dump=False)
    th = fio.read_thermo(wd/'thermo.dat')
    k0 = int(np.argmin(th['step']))
    V = float(np.prod(box))
    eps = 1e-4
    es = []
    for s in (+1, -1):
        p = pos.copy()
        p[:, 1] += s * eps * (pos[:, 0] - pos[:, 0].mean())                 # y' = y + eps x
        es.append(run(tmp_path, f'v{s}', p, mol, bonds, angles, dihs, box, dihedral_line=line)[1])
    w_xy = -(es[0] - es[1]) / (2 * eps)                                     # W_xy = sum x F_y = -dE/d(eps)
    assert th['Pxy'][k0] * V == pytest.approx(w_xy, rel=2e-2, abs=2e-3)


@pytest.mark.parametrize('mode', ['gather', 'scatter'])
def test_cbt_energy_and_forces_match_numpy(tmp_path, mode):
    pos, mol, bonds, angles, dihs, box = coil_system(seed=13)
    phi = np.deg2rad(np.linspace(-180, 180, 3601))
    u = lambda p: opls(p, K) + 2.0 * np.sin(p)                             # asymmetric, nonzero at the minimum
    du = lambda p: dopls(p, K) + 2.0 * np.cos(p)
    f, e, _ = run(tmp_path, mode, pos, mol, bonds, angles, dihs, box, extra=(f'bonded_mode {mode}',),
                  dihedral_line=table_line(np.degrees(phi), u(phi), -du(phi) * np.pi / 180, cbt=True))
    energy = lambda p: cbt_energy(p, dihs, box, u)
    assert e == pytest.approx(energy(pos), rel=1e-4)
    assert abs(e - opls(phi_numpy(pos, dihs, box), K).sum()) > 1.0        # the bending prefactor matters
    np.testing.assert_allclose(f, numeric_forces(energy, pos), atol=2e-3 * np.abs(f).max())


def test_cbt_forces_stay_finite_for_straight_angles(tmp_path):
    """Chain 0 has two exactly straight bond angles (the configuration that crashed the plain table on PCL)."""
    pos, mol, bonds, angles, dihs, box = coil_system(seed=17)
    x = pos[:8].copy()
    for k in (2, 5):                                                        # angles k-1, k, k+1 -> 180 degrees
        d = x[k] - x[k - 1]
        x[k + 1:] += x[k] + d - x[k + 1]
    pos[:8] = x - x.mean(0) + pos[:8].mean(0)
    assert pos.min() > 0 and pos.max() < box[0]
    phi = np.deg2rad(np.linspace(-180, 180, 3601))
    u = lambda p: opls(p, K) + 2.0
    f, e, _ = run(tmp_path, 'straight', pos, mol, bonds, angles, dihs, box,
                  dihedral_line=table_line(np.degrees(phi), u(phi), -dopls(phi, K) * np.pi / 180, cbt=True))
    energy = lambda p: cbt_energy(p, dihs, box, u)
    ref = numeric_forces(energy, pos)
    assert np.all(np.isfinite(f)) and np.abs(f).max() < 100.0
    assert e == pytest.approx(energy(pos), rel=1e-4)
    np.testing.assert_allclose(f, ref, atol=2e-3 * np.abs(ref).max())
