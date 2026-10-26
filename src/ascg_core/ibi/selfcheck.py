"""Deterministic gate for the fastMD engine on the current host (run by every IBI job before sampling):

    python -m ascg_core.ibi.selfcheck [--no-dihedrals]     # exit status 0 = pass

1. Exclusions: a toy melt of straight chains with analytic pair / bond / angle tables (a bump in the pair table at
   the 1-3 distance) is written with the pipeline's own writers; fastMD's total energy (forces_dump_file) must equal
   an independent numpy evaluation with fastMD's linear table interpolation and the 1-2 and 1-3 pairs excluded from
   the pair potential (the zero-energy exclusion bonds), and must differ from the evaluation that includes 1-3
   pairs. Net force must vanish.
2. CBT dihedrals (the IBI default; skipped with --no-dihedrals): random-coil chains with an asymmetric dihedral table
   and zero pair / bond / angle tables; fastMD's energy must equal numpy's sum of sin^3 theta1 sin^3 theta2 U(phi)
   (and differ from the plain sum of U(phi)), forces finite, net force zero. Fails on a fastMD without
   dihedral_table (before fastMD PR #3).
3. NPT stability: a short npt_nh run of a CG liquid of small stiff molecules (data/npt_check.tgz) must keep its
   volume and a moderate pressure spread; the barostat before fastMD PR #4 collapses it.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from . import fastmd_io as fio
from .system import CGSystem
from .tables import Potential, write_tables


def toy_system(n_chains=8, length=6, box=3.0, seed=0):
    """Straight chains of one bead type (bonds 0.38 nm) with small noise; 1-3 pairs as exclusions."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_chains)))
    pos, mol, bonds, angles = [], [], [], []
    for c in range(n_chains):
        start = np.array([0.2, (c % side + 0.5) * box / side, (c // side + 0.5) * box / side])
        for k in range(length):
            pos.append(start + np.array([0.38 * k, 0, 0]) + rng.normal(0, 0.03, 3))
            mol.append(c)
        b = c * length
        bonds += [(b + k, b + k + 1) for k in range(length - 1)]
        angles += [(b + k, b + k + 1, b + k + 2) for k in range(length - 2)]
    bonds, angles = np.array(bonds), np.array(angles)
    sysm = CGSystem(task='toy', type_names=['A'], types=np.zeros(len(pos), int), masses=np.full(len(pos), 28.0),
                    mol=np.array(mol), bonds=bonds, bond_type=np.zeros(len(bonds), int), bond_types=['A-A'],
                    angles=angles, angle_type=np.zeros(len(angles), int), angle_types=['A-A-A'],
                    excl13=np.array([(a, c) for a, _, c in angles]), pair_keys=['A-A'], pair_index=[(0, 0)])
    return sysm, np.array(pos), np.full(3, box)


def toy_potentials(rc=1.2):
    r = np.round(np.arange(0.2, rc + 1e-9, 0.002), 10)
    s = np.clip((rc - r) / (rc - 0.9), 0, 1)**2                      # smooth to zero at rc
    pair = (4 * ((0.35 / r)**12 - (0.35 / r)**6) + 2.0 * np.exp(-((r - 0.75) / 0.05)**2)) * s   # bump at the 1-3 distance
    b = np.arange(0.2, 0.7, 0.0005)
    th = np.deg2rad(np.arange(0.0, 180.0001, 0.25))
    return {'pair': {'A-A': Potential('pair', r, pair)},
            'bond': {'A-A': Potential('bond', b, 3000.0 * (b - 0.38)**2)},
            'angle': {'A-A-A': Potential('angle', th, 15.0 * (th - 2.6)**2)}}


def numpy_energy(sysm, pots, pos, box, rc, exclude13=True):
    """Total energy with fastMD's linear table interpolation; pairs: all i < j within rc except 1-2 (and 1-3)."""
    excl = {tuple(sorted(b)) for b in sysm.bonds.tolist()}
    if exclude13:
        excl |= {tuple(sorted(e)) for e in sysm.excl13.tolist()}
    p, bt, at = pots['pair']['A-A'], pots['bond']['A-A'], pots['angle']['A-A-A']
    i, j = np.triu_indices(len(pos), 1)
    keep = np.array([(a, b) not in excl for a, b in zip(i, j)])
    d = pos[i[keep]] - pos[j[keep]]
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=1)
    e = np.interp(r[r < rc], p.x, p.u).sum()
    db = pos[sysm.bonds[:, 0]] - pos[sysm.bonds[:, 1]]
    e += np.interp(np.linalg.norm(db, axis=1), bt.x, bt.u).sum()
    u = pos[sysm.angles[:, 0]] - pos[sysm.angles[:, 1]]
    v = pos[sysm.angles[:, 2]] - pos[sysm.angles[:, 1]]
    th = np.degrees(np.arccos(np.clip((u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1), -1, 1)))
    e += np.interp(th, np.degrees(at.x), at.u).sum()
    return e


def fastmd_exclusion_energy_check(workdir, rc=1.2, env=None):
    """Run the check in `workdir`; returns a dict with the energies and 'ok'."""
    wd = Path(workdir)
    sysm, pos, box = toy_system()
    pots = toy_potentials(rc)
    entries = write_tables(wd/'tables', sysm, pots)
    sysm.write_data(wd/'s.data', pos, box)
    conf = fio.conf_text(data_file='s.data', ntypes=1, rc=rc, skin=0.1, dt=0.002, nsteps=1,
                         ensemble=('nve',), pair_tables=entries[0], bond_tables=entries[1], angle_tables=entries[2],
                         extra=('forces_dump_file forces.dat',))
    (wd/'run.conf').write_text(conf)
    fio.run('run.conf', wd, env=env)
    f = np.loadtxt(wd/'forces.dat', comments='#')
    out = {'e_fastmd': float(f[:, 4].sum()), 'e_ref': float(numpy_energy(sysm, pots, pos, box, rc)),
           'e_with13': float(numpy_energy(sysm, pots, pos, box, rc, exclude13=False)),
           'net_force': float(np.abs(f[:, 1:4].sum(0)).max())}
    out['ok'] = bool(abs(out['e_with13'] - out['e_ref']) > 1.0
                     and abs(out['e_fastmd'] - out['e_ref']) <= max(1e-3, 1e-4 * abs(out['e_ref']))
                     and np.all(np.isfinite(f[:, 1:4])) and out['net_force'] < 1e-2)
    return out


def coil_system(n_chains=6, length=8, box=4.0, seed=1):
    """Chains of one bead type with bonds 0.40 nm, angles ~110 deg and random torsions; 1-3 pairs as exclusions."""
    rng = np.random.default_rng(seed)
    pos, mol, bonds, angles = [], [], [], []
    for c in range(n_chains):
        x, d = [np.zeros(3)], np.array([1.0, 0.0, 0.0])
        for _ in range(1, length):
            perp = np.cross(d, rng.normal(size=3))
            perp /= np.linalg.norm(perp)
            th = np.deg2rad(70 + rng.normal(0, 8))
            d = np.cos(th) * d + np.sin(th) * perp
            x.append(x[-1] + 0.40 * d / np.linalg.norm(d))
        centre = np.array([(c % 3 + 0.5) * box / 3, (c // 3 % 2 + 0.5) * box / 2, box / 2])
        pos += list(np.array(x) - np.mean(x, 0) + centre)
        mol += [c] * length
        b = c * length
        bonds += [(b + k, b + k + 1) for k in range(length - 1)]
        angles += [(b + k, b + k + 1, b + k + 2) for k in range(length - 2)]
    bonds, angles = np.array(bonds), np.array(angles)
    sysm = CGSystem(task='toy', type_names=['A'], types=np.zeros(len(pos), int), masses=np.full(len(pos), 28.0),
                    mol=np.array(mol), bonds=bonds, bond_type=np.zeros(len(bonds), int), bond_types=['A-A'],
                    angles=angles, angle_type=np.zeros(len(angles), int), angle_types=['A-A-A'],
                    excl13=np.array([(a, c) for a, _, c in angles]), pair_keys=['A-A'], pair_index=[(0, 0)])
    return sysm.add_dihedrals('cbt'), np.array(pos), np.full(3, box)


def fastmd_cbt_dihedral_check(workdir, rc=1.2, env=None):
    """CBT dihedral energy of a toy coil melt vs numpy; returns a dict with the energies and 'ok'."""
    from .dihedrals import cbt_prefactor, phi, table_grid
    wd = Path(workdir)
    sysm, pos, box = coil_system()
    r = np.round(np.arange(0.2, rc + 1e-9, 0.002), 10)
    b = np.arange(0.2, 0.7, 0.0005)
    th = np.deg2rad(np.arange(0.0, 180.0001, 0.25))
    g = table_grid()
    u = 1.5 * np.cos(g) + 0.8 * np.sin(g) + 0.5 * np.cos(3 * g) + 1.0
    pots = {'pair': {'A-A': Potential('pair', r, 0 * r)}, 'bond': {'A-A': Potential('bond', b, 0 * b)},
            'angle': {'A-A-A': Potential('angle', th, 0 * th)},
            'dihedral': {sysm.dihedral_types[0]: Potential('dihedral', g, u)}}
    entries = write_tables(wd/'tables', sysm, pots)
    sysm.write_data(wd/'s.data', pos, box)
    conf = fio.conf_text(data_file='s.data', ntypes=1, rc=rc, skin=0.1, dt=0.002, nsteps=1, ensemble=('nve',),
                         pair_tables=entries[0], bond_tables=entries[1], angle_tables=entries[2],
                         dihedral_tables=entries[3], extra=('forces_dump_file forces.dat',))
    (wd/'run.conf').write_text(conf)
    out = {'ok': False}
    try:
        fio.run('run.conf', wd, env=env)
    except Exception as e:                      # e.g. a fastMD without dihedral_table
        out['error'] = f'{type(e).__name__}: {e}'.splitlines()[0][:200]
        return out
    f = np.loadtxt(wd/'forces.dat', comments='#')
    ph = phi(pos, sysm.dihedrals)
    u_phi = np.interp(ph, g, u)
    out.update(e_fastmd=float(f[:, 4].sum()), e_ref=float((cbt_prefactor(pos, sysm.dihedrals) * u_phi).sum()),
               e_plain=float(u_phi.sum()), net_force=float(np.abs(f[:, 1:4].sum(0)).max()))
    out['ok'] = bool(abs(out['e_plain'] - out['e_ref']) > 1.0
                     and abs(out['e_fastmd'] - out['e_ref']) <= max(1e-3, 1e-4 * abs(out['e_ref']))
                     and np.all(np.isfinite(f[:, 1:4])) and out['net_force'] < 1e-2)
    return out


def fastmd_npt_stability_check(workdir, env=None):
    """fastMD npt_nh on a liquid of small stiff molecules (coarse-grained toluene, 2000 beads, tabulated pairs and a
    stiff bond; 40 ps at 300 K, 1 atm, Pdamp 1 ps, from its equilibrium volume 180.1 nm3). The npt_nh before fastMD
    PR #4 collapses it (volume about 150 nm3, pressure sd about 13,500 bar); a correct barostat keeps 177-182 nm3 with a
    pressure sd of 90-160 bar (repeated runs on a 3090; GPU runs are not bitwise repeatable)."""
    import tarfile
    from importlib import resources
    from . import fastmd_io as fio
    wd = Path(workdir)
    with resources.as_file(resources.files('ascg_core.ibi').joinpath('data/npt_check.tgz')) as f:
        with tarfile.open(f) as t:
            try:
                t.extractall(wd, filter='data')
            except TypeError:                                  # Python without extraction filters
                t.extractall(wd)
    run = wd/'npt_check'
    try:
        fio.run('run.conf', run, env=env, timeout=600)
    except Exception as e:                                        # engine failure is a failed check, not a crash
        return {'ok': False, 'error': f'{type(e).__name__}: {e}'}
    a = np.loadtxt(run/'thermo.dat', comments='#', ndmin=2)
    t = a[:, 0] * 0.001
    p_bar = a[:, 4:7].mean(1) * fio.BAR_PER_KJ_MOL_NM3
    _, box, _ = fio.read_restart(run/'restart_final.data')
    vol = float(np.prod(box))
    late = t > 20
    sd = float(p_bar[late].std())
    return {'ok': bool(vol > 172.0 and sd < 1000.0), 'volume_nm3': round(vol, 2), 'start_volume_nm3': 180.1,
            'pressure_sd_bar': round(sd, 1), 'pressure_mean_bar': round(float(p_bar[late].mean()), 1)}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-dihedrals', action='store_true', help='skip the CBT dihedral check')
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix='fastmd_selfcheck_') as d:
        r = fastmd_exclusion_energy_check(d)
    print(f"fastMD selfcheck: E_fastMD {r['e_fastmd']:.4f}, E_numpy(1-2,1-3 excluded) {r['e_ref']:.4f}, "
          f"E_numpy(1-3 included) {r['e_with13']:.4f}, |net F| {r['net_force']:.2e} -> {'PASS' if r['ok'] else 'FAIL'}",
          flush=True)
    ok = r['ok']
    if not a.no_dihedrals:
        with tempfile.TemporaryDirectory(prefix='fastmd_selfcheck_') as d:
            q = fastmd_cbt_dihedral_check(d)
        if 'error' in q:
            print(f"fastMD selfcheck, CBT dihedrals: fastMD run failed ({q['error']}); the IBI default needs fastMD "
                  f"with dihedral_table ... cbt (main b35e30a or later) -> FAIL", flush=True)
        else:
            print(f"fastMD selfcheck, CBT dihedrals: E_fastMD {q['e_fastmd']:.4f}, E_numpy(CBT) {q['e_ref']:.4f}, "
                  f"E_numpy(plain) {q['e_plain']:.4f}, |net F| {q['net_force']:.2e} -> "
                  f"{'PASS' if q['ok'] else 'FAIL'}", flush=True)
        ok = ok and q['ok']
    with tempfile.TemporaryDirectory(prefix='fastmd_selfcheck_') as d:
        n = fastmd_npt_stability_check(d)
    if 'error' in n:
        print(f"fastMD selfcheck, NPT stability: fastMD run failed ({n['error']}) -> FAIL", flush=True)
    else:
        print(f"fastMD selfcheck, NPT stability (stiff small molecules, Pdamp 1 ps): volume {n['volume_nm3']} nm3 "
              f"(from 180.1), pressure sd {n['pressure_sd_bar']} bar -> {'PASS' if n['ok'] else 'FAIL'}"
              + ('' if n['ok'] else ' (an npt_nh without the symmetric barostat splitting of fastMD PR #4)'), flush=True)
    ok = ok and n['ok']
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
