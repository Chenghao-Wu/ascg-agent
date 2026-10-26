"""Shared CG potentials as tables: the Boltzmann-inverted start and fastMD table files.

Two models: one shared potential set for all temperatures (Potential), or temperature-dependent potentials
U(x, T) = A(x) + B(x) T (TPotential), A and B tables on one uniform grid; the fastMD tables of a
state are U(x, T_s) (cut at E_CAP like the shared tables). The T-dependent BI start inverts each temperature's own
distributions, W_s = -k_B T_s ln P_s (pair: cg_tables.pair_table of that T alone), and solves A + B T_s = W_s.

A Potential is U(x) on a uniform grid (pair / bond: x = r in nm; angle: x = theta in rad), kJ/mol. The force
column written for fastMD is -dU/dx from a cubic spline through the grid values, so energy and force are
always consistent (angles: degrees and force per degree, fastMD's angle_table convention).

BI start: cg_tables' Boltzmann inversion of the mapped-AA distributions, the documented average of the 400
and 500 K potentials of mean force (one shared table set for both temperatures): pair tables with a
repulsive wall below the sampled range and a C2 switch to zero between r_switch and rc, bond / angle
tables with quadratic walls outside the sampled range. Grid points where the wall exceeds E_CAP are cut
from the start of pair / bond tables (fastMD clamps below the first point; beads never get there).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base_tables import bonded_table, pair_table, with_pressure_ramp
from .states import label as _lab


@dataclass
class Table:
    """Minimal uniform table (x0, dx, value) for the BI conversions here (slopes come from the cubic spline)."""
    x0: float
    dx: float
    value: np.ndarray
from . import fastmd_io as fio

E_CAP_KJ = 250.0          # first table point: wall energy below this (~60 k_B T at 500 K)
BOND_BLEND_NM = 0.005     # width of the smooth hand-over from the BI table to the harmonic bond wall
BOND_X_MAX_NM = 5.0       # bond tables never extend beyond this
RAD2DEG = 180.0 / np.pi


@dataclass
class Potential:
    kind: str                # 'pair' | 'bond' | 'angle'
    x: np.ndarray            # uniform grid (nm or rad)
    u: np.ndarray            # kJ/mol

    def force(self):
        """-dU/dx on the grid (per nm or per rad); dihedrals: periodic spline on [-pi, pi]."""
        from scipy.interpolate import CubicSpline
        if self.kind == 'dihedral':
            u = self.u.copy()
            u[-1] = u[0]
            return -CubicSpline(self.x, u, bc_type='periodic')(self.x, 1)
        return -CubicSpline(self.x, self.u)(self.x, 1)

    def __call__(self, x):
        return np.interp(x, self.x, self.u)

    def copy(self, u=None):
        return Potential(self.kind, self.x.copy(), self.u.copy() if u is None else np.asarray(u, float))

    def write(self, path, keyword, header=''):
        f = self.force()
        if self.kind == 'dihedral':           # one period, identical first / last rows
            u = self.u.copy()
            u[-1] = u[0]
            f[-1] = f[0]
            fio.write_table(path, keyword, self.x * RAD2DEG, u, f / RAD2DEG, header=header)
        elif self.kind == 'angle':
            fio.write_table(path, keyword, self.x * RAD2DEG, self.u, f / RAD2DEG, header=header)
        else:
            fio.write_table(path, keyword, self.x, self.u, f, header=header)

    def to_dict(self):
        return {'kind': self.kind, 'x': self.x.tolist(), 'u': self.u.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d['kind'], np.asarray(d['x'], float), np.asarray(d['u'], float))


def _from_table(t, kind, cap=E_CAP_KJ, x_max=None):
    x = t.x0 + t.dx * np.arange(len(t.value))
    keep = x > 1e-9 if kind != 'angle' else np.ones(len(x), bool)
    if x_max is not None:
        keep &= x <= x_max + 1e-9
    x, u = x[keep], t.value[keep]
    if kind != 'angle':
        first = int(np.argmax(u < cap))          # first point below the cap (walls rise towards small x)
        x, u = x[first:], u[first:]
    return Potential(kind, x, u.copy())


def harmonic_bond_walls(x, u, centres, prob, kt, *, floor_rel=1e-3, cap=E_CAP_KJ, blend=BOND_BLEND_NM):
    """Bond table outside the AA-sampled range replaced by a harmonic bond: U_h = k/2 (b - b0)^2 with
    b0 the mean and k = k_B T / var of the AA bond-length distribution, shifted to the table's value at each edge of
    the sampled range (P > floor_rel max P) and blended in over `blend` nm with a C1 smoothstep; the grid is extended
    (same spacing) until the wall exceeds `cap` on the long-bond side. Returns (x, u)."""
    x, u = np.asarray(x, float), np.asarray(u, float)
    c, p = np.asarray(centres, float), np.clip(np.asarray(prob, float), 0, None)
    on = p > floor_rel * p.max()
    lo, hi = float(c[on].min()), float(c[on].max())
    w = p / p.sum()
    b0 = float((w * c).sum())
    k = kt / float((w * (c - b0)**2).sum())
    dx = x[1] - x[0]
    uh = lambda r: 0.5 * k * (r - b0)**2
    u_hi, u_lo = np.interp(hi, x, u), np.interp(lo, x, u)
    n_more = 0
    while uh(x[-1] + n_more * dx) - uh(hi) + u_hi < cap and x[-1] + n_more * dx < BOND_X_MAX_NM:
        n_more += 1
    if n_more:
        x = np.concatenate([x, x[-1] + dx * np.arange(1, n_more + 1)])
        u = np.concatenate([u, np.full(n_more, u[-1])])
    smooth = lambda t: np.clip(t, 0, 1)**2 * (3 - 2 * np.clip(t, 0, 1))
    s_hi = smooth((x - hi) / blend)                 # 0 inside, 1 beyond hi + blend
    s_lo = smooth((lo - x) / blend)
    u = (1 - s_hi) * u + s_hi * (uh(x) - uh(hi) + u_hi)
    u = (1 - s_lo) * u + s_lo * (uh(x) - uh(lo) + u_lo)
    return x, u, {'b0_nm': b0, 'k_kj_nm2': k, 'sampled_nm': (lo, hi)}


def bi_potentials(ref_dir, system, temperatures=(400, 500), *, rc=1.5, r_switch=1.3, pressure_ramp_kj=0.0):
    """Shared BI start {'pair': {key: Potential}, 'bond': {...}, 'angle': {...}} from the mapped-AA targets."""
    ref = Path(ref_dir)
    t = {T: np.load(ref/f'targets_T{_lab(T)}.npz') for T in temperatures}
    T0 = temperatures[0]
    out = {'pair': {}, 'bond': {}, 'angle': {}}
    for k in system.pair_keys:
        tab = pair_table(t[T0]['r_nm'], {T: t[T][f'rdf/{k}'] for T in temperatures}, r_switch=r_switch, r_cut=rc)
        if pressure_ramp_kj:
            tab = with_pressure_ramp(tab, pressure_ramp_kj, r_switch=r_switch, r_cut=rc)
        out['pair'][k] = _from_table(tab, 'pair', x_max=rc)
    for k in system.bond_types:
        e = t[T0][f'bond_edges/{k}']
        c = 0.5 * (e[1:] + e[:-1])
        tab = bonded_table(c, {T: t[T][f'bond/{k}'] for T in temperatures}, kind='bond', dx=0.0005,
                           x_range=(0.0, float(e[-1]) + 0.3))
        xt = tab.x0 + tab.dx * np.arange(len(tab.value))
        xw, uw, _ = harmonic_bond_walls(xt, tab.value, c, np.mean([t[T][f'bond/{k}'] for T in temperatures], axis=0),
                                        fio.kT(np.mean([float(T) for T in temperatures])))
        out['bond'][k] = _from_table(Table(float(xw[0]), float(xw[1] - xw[0]), uw), 'bond')
    for k in system.angle_types:
        tab = bonded_table(t[T0]['angle_rad'], {T: t[T][f'angle/{k}'] for T in temperatures}, kind='angle',
                           dx=np.deg2rad(0.25))
        out['angle'][k] = _from_table(tab, 'angle')
    return out


@dataclass
class TPotential:
    """U(x, T) = A(x) + B(x) T on a uniform grid (kJ/mol; B in kJ/mol/K)."""
    kind: str
    x: np.ndarray
    a: np.ndarray
    b: np.ndarray

    def at(self, T, cap=E_CAP_KJ):
        u = self.a + self.b * float(T)
        x = self.x
        if self.kind in ('pair', 'bond'):
            first = int(np.argmax(u < cap))
            x, u = x[first:], u[first:]
        return Potential(self.kind, x.copy(), u.copy())

    def to_dict(self):
        return {'kind': self.kind, 'x': self.x.tolist(), 'a': self.a.tolist(), 'b': self.b.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d['kind'], np.asarray(d['x'], float), np.asarray(d['a'], float), np.asarray(d['b'], float))


def solve_ab(u_by_T, weights=None):
    """A, B with A + B T_s = U_s over the states s (temperatures or ibi.states.State; the model depends on T only):
    one distinct temperature -> B = 0 and A = the (weighted) mean, a single table set; two states at two temperatures
    -> exact; otherwise least squares weighted by the state weights (default equal)."""
    temps = sorted(u_by_T)
    U = np.stack([np.asarray(u_by_T[T], float) for T in temps])
    if len({float(T) for T in temps}) == 1:
        w = np.array([float(weights[T]) if weights else 1.0 for T in temps])
        return (w[:, None] * U).sum(0) / w.sum(), np.zeros_like(U[0])
    M = np.array([[1.0, float(T)] for T in temps])
    if weights is not None and len(temps) > 2:
        w = np.sqrt(np.array([float(weights[T]) for T in temps]))
        M, U = M * w[:, None], U * w[:, None]
    coef, *_ = np.linalg.lstsq(M, U, rcond=None)
    return coef[0], coef[1]


def bi_tpotentials(ref_dir, system, temperatures=(400, 500), *, rc=1.5, r_switch=1.3, weights=None):
    """T-dependent BI start {'pair'|'bond'|'angle': {key: TPotential}} from each temperature's own distributions."""
    ref = Path(ref_dir)
    t = {T: np.load(ref/f'targets_T{_lab(T)}.npz') for T in temperatures}
    T0 = temperatures[0]
    out = {'pair': {}, 'bond': {}, 'angle': {}}

    def add(cat, key, kind, tabs):
        x = tabs[T0].x0 + tabs[T0].dx * np.arange(len(tabs[T0].value))
        if any(len(tb.value) != len(x) or abs(tb.x0 - tabs[T0].x0) > 1e-12 for tb in tabs.values()):
            raise ValueError(f'{cat} {key}: per-temperature BI tables are on different grids')
        keep = x > 1e-9 if kind != 'angle' else np.ones(len(x), bool)
        if kind == 'pair':
            keep &= x <= rc + 1e-9
        a, b = solve_ab({T: tabs[T].value[keep] for T in temperatures}, weights)
        out[cat][key] = TPotential(kind, x[keep], a, b)
    for k in system.pair_keys:
        add('pair', k, 'pair', {T: pair_table(t[T]['r_nm'], {T: t[T][f'rdf/{k}']}, r_switch=r_switch, r_cut=rc)
                                for T in temperatures})
    for k in system.bond_types:
        e = t[T0][f'bond_edges/{k}']
        c = 0.5 * (e[1:] + e[:-1])
        tabs = {T: bonded_table(c, {T: t[T][f'bond/{k}']}, kind='bond', dx=0.0005, x_range=(0.0, float(e[-1]) + 0.3))
                for T in temperatures}
        walled = {}
        for T, tb in tabs.items():                       # harmonic walls per state, then one common grid
            xt = tb.x0 + tb.dx * np.arange(len(tb.value))
            walled[T] = harmonic_bond_walls(xt, tb.value, c, t[T][f'bond/{k}'], fio.kT(T))[:2]
        n = max(len(xw) for xw, _ in walled.values())
        for T, (xw, uw) in walled.items():
            if len(xw) < n:                              # extend with the same harmonic tail (quadratic in x)
                ext = xw[-1] + (xw[1] - xw[0]) * np.arange(1, n - len(xw) + 1)
                coef = np.polyfit(xw[-20:], uw[-20:], 2)
                walled[T] = (np.concatenate([xw, ext]), np.concatenate([uw, np.polyval(coef, ext)]))
        add('bond', k, 'bond', {T: Table(float(xw[0]), float(xw[1] - xw[0]), uw) for T, (xw, uw) in walled.items()})
    for k in system.angle_types:
        add('angle', k, 'angle', {T: bonded_table(t[T0]['angle_rad'], {T: t[T][f'angle/{k}']}, kind='angle',
                                                  dx=np.deg2rad(0.25)) for T in temperatures})
    return out


def tpotentials_at(tpots, T):
    """The Potential set of state T."""
    return {c: {k: p.at(T) for k, p in v.items()} for c, v in tpots.items()}


def save_tpotentials(dirpath, tpots):
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    (d/'potentials_tdep.json').write_text(json.dumps({c: {k: p.to_dict() for k, p in v.items()}
                                                      for c, v in tpots.items()}))


def load_tpotentials(dirpath):
    raw = json.loads((Path(dirpath)/'potentials_tdep.json').read_text())
    return {c: {k: TPotential.from_dict(p) for k, p in v.items()} for c, v in raw.items()}


def write_tables(dirpath, system, pots):
    """fastMD table files in `dirpath` (pair.table, bond.table, angle.table, dihedral.table when the system has
    dihedrals, plus the zero 1-3 exclusion bond); returns the config entries (pair_tables, bond_tables, angle_tables,
    dihedral_tables) with absolute paths."""
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    for name in ('pair.table', 'bond.table', 'angle.table', 'dihedral.table'):
        (d/name).unlink(missing_ok=True)
    pair_entries, bond_entries, angle_entries, dihedral_entries = [], [], [], []
    for (ti, tj), k in zip(system.pair_index, system.pair_keys):
        kw = f'PAIR_{k}'
        pots['pair'][k].write(d/'pair.table', kw, header=f'pair {k}')
        pair_entries.append((ti, tj, str((d/'pair.table').resolve()), kw))
    for i, k in enumerate(system.bond_types):
        kw = f'BOND_{k}'
        pots['bond'][k].write(d/'bond.table', kw, header=f'bond {k}')
        bond_entries.append((i, str((d/'bond.table').resolve()), kw))
    if len(system.excl13):
        x = np.linspace(0.01, 5.0, 2)
        fio.write_table(d/'bond.table', fio.EXCLUSION_BOND_KEYWORD, x, np.zeros(2), np.zeros(2),
                        header='zero-energy bond type marking 1-3 pairs excluded from the pair potential')
        bond_entries.append((len(system.bond_types), str((d/'bond.table').resolve()), fio.EXCLUSION_BOND_KEYWORD))
    for i, k in enumerate(system.angle_types):
        kw = f'ANGLE_{k}'
        pots['angle'][k].write(d/'angle.table', kw, header=f'angle {k}')
        angle_entries.append((i, str((d/'angle.table').resolve()), kw))
    for i, k in enumerate(getattr(system, 'dihedral_types', []) or []):
        kw = f'DIHEDRAL_{k}'
        pots['dihedral'][k].write(d/'dihedral.table', kw, header=f'dihedral {k} ({system.dihedral_style})')
        dihedral_entries.append((i, str((d/'dihedral.table').resolve()), kw)
                                + (('cbt',) if system.dihedral_style == 'cbt' else ()))
    (d/'potentials.json').write_text(json.dumps({c: {k: p.to_dict() for k, p in v.items()} for c, v in pots.items()}))
    return pair_entries, bond_entries, angle_entries, dihedral_entries


def load_potentials(dirpath):
    raw = json.loads((Path(dirpath)/'potentials.json').read_text())
    return {c: {k: Potential.from_dict(p) for k, p in v.items()} for c, v in raw.items()}
