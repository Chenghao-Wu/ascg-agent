"""CG structure from fastMD trajectories on the reference's grids, and the agreement metrics.

Same estimators and normalisations as the mapped-AA targets (cg_reference.prepare_reference):
intermolecular partial g(r) per type pair (counts / (n_pairs shell / V) per frame), bond-length and angle
probability densities per type (each frame's histogram over its own count and bin width), averaged over
frames; bonds and angles from unwrapped coordinates.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..mapping.cg_mapping import angle_values, bond_lengths, frame_rdf_counts, intermolecular_pair_counts


def nrmse(x, ref):
    return float(math.sqrt(np.mean((np.asarray(x) - ref)**2) / np.mean(np.asarray(ref)**2)))


class Grids:
    """The reference's observable grids (from targets_T*.npz)."""

    def __init__(self, targets, system):
        r = targets['r_nm']
        dr = float(r[1] - r[0])
        self.r_edges = np.concatenate([[r[0] - dr / 2], r + dr / 2])
        self.r = r
        self.bond_edges = {k: targets[f'bond_edges/{k}'] for k in system.bond_types}
        th = targets['angle_rad']
        dth = float(th[1] - th[0])
        self.angle_edges = np.concatenate([[th[0] - dth / 2], th + dth / 2])
        self.theta = th


def _density_hist(values, edges):
    h, _ = np.histogram(values, bins=edges)
    tot = h.sum()
    return h / (tot * np.diff(edges)) if tot else h.astype(float)


def distributions(frames, system, grids, bond_limits=None):
    """Mean distributions over frames [(unwrapped_pos, box)] -> {'rdf': {key: g}, 'bond': {...}, 'angle': {...},
    'volume_nm3': [...]}; with bond_limits ({type: break length}) also 'bond_persistent_over': the number of bonds of
    each type beyond their break length in two consecutive frames (the broken-bond gate, check_bonds)."""
    tname = [system.type_names[t] for t in system.types]
    n_pairs = intermolecular_pair_counts(tname, system.mol, system.type_names)
    shell = 4 / 3 * np.pi * (grids.r_edges[1:]**3 - grids.r_edges[:-1]**3)
    acc = {'rdf': {k: 0.0 for k in system.pair_keys}, 'bond': {k: 0.0 for k in system.bond_types},
           'angle': {k: 0.0 for k in system.angle_types}}
    with_dih = getattr(system, 'dihedrals', None) is not None
    if with_dih:
        from . import dihedrals as dmod
        acc['dihedral'] = {k: 0.0 for k in system.dihedral_types}
    bond_max = {k: 0.0 for k in system.bond_types}
    persist = {k: 0 for k in system.bond_types}
    lim = (np.array([bond_limits[k] for k in system.bond_types])[system.bond_type]
           if bond_limits is not None else None)
    over_prev = None
    vols = []
    for pos, box in frames:
        box = np.asarray(box, float)[:3]
        v = float(np.prod(box))
        vols.append(v)
        counts = frame_rdf_counts(pos, box, tname, system.mol, system.type_names, grids.r_edges)
        for k in system.pair_keys:
            acc['rdf'][k] = acc['rdf'][k] + (counts[k] / (n_pairs[k] * shell / v) if n_pairs[k] else 0.0)
        bl = bond_lengths(pos, system.bonds)
        for i, k in enumerate(system.bond_types):
            acc['bond'][k] = acc['bond'][k] + _density_hist(bl[system.bond_type == i], grids.bond_edges[k])
            bond_max[k] = max(bond_max[k], float(bl[system.bond_type == i].max()))
        if lim is not None:
            over = bl > lim
            if over_prev is not None:
                both = over & over_prev
                for i, k in enumerate(system.bond_types):
                    persist[k] += int(both[system.bond_type == i].sum())
            over_prev = over
        if len(system.angles):
            av = angle_values(pos, system.angles)
            for i, k in enumerate(system.angle_types):
                acc['angle'][k] = acc['angle'][k] + _density_hist(av[system.angle_type == i], grids.angle_edges)
        if with_dih:
            fd = dmod.frame_density(dmod.phi(pos, system.dihedrals), system.dihedral_type, len(system.dihedral_types))
            for i, k in enumerate(system.dihedral_types):
                acc['dihedral'][k] = acc['dihedral'][k] + fd[i]
    nf = max(len(vols), 1)
    out = {c: {k: np.asarray(v) / nf for k, v in d.items()} for c, d in acc.items()}
    out['volume_nm3'] = np.asarray(vols)
    out['bond_max_nm'] = bond_max                    # longest bond of each type over the frames (unwrapped)
    if lim is not None:
        out['bond_persistent_over'] = persist
    return out


def trajectory_frames(path, *, skip_ps=0.0, dt_ps=None):
    """[(unwrapped positions, box)] from a fastMD trajectory, dropping frames before skip_ps."""
    from .fastmd_io import read_trajectory
    tr = read_trajectory(path)
    steps = tr.steps
    keep = np.ones(len(tr), bool) if not skip_ps else (steps * dt_ps >= skip_ps - 1e-9)
    return [(tr.unwrapped(k), tr.box(k)) for k in np.flatnonzero(keep)]


def metrics(cg, targets, system):
    """Per-type NRMSE of RDFs, bonds and angles against the mapped-AA targets."""
    m = {'rdf': {k: nrmse(cg['rdf'][k], targets[f'rdf/{k}']) for k in system.pair_keys},
         'bond': {k: nrmse(cg['bond'][k], targets[f'bond/{k}']) for k in system.bond_types},
         'angle': {k: nrmse(cg['angle'][k], targets[f'angle/{k}']) for k in system.angle_types}}
    if 'dihedral' in cg:
        m['dihedral'] = {k: nrmse(cg['dihedral'][k], targets[f'dihedral/{k}']) for k in system.dihedral_types}
    return m


PAIRHIST_DR = 0.002     # nm, grid of the nonbonded pair-distance histogram (virial pressure correction)
PAIRHIST_STRIDE = 10    # every 10th frame


def pair_histograms(frames, system, rc, dr=PAIRHIST_DR, stride=PAIRHIST_STRIDE):
    """Mean number of nonbonded pairs (every pair the pair tables act on: intra- and intermolecular, 1-2 and 1-3
    excluded) per type pair in r bins up to rc, over every stride-th frame, and the mean volume:
    ({key: counts}, edges). The first-order pressure change of a pair-table change dU_k(r) is then
    dP = -1 / (3 V) sum_k sum_b h_k(r_b) r_b dU_k'(r_b) (update.virial_pressure_change)."""
    from scipy.spatial import cKDTree
    n = system.n
    edges = np.arange(0.0, rc + dr / 2, dr)
    excl = np.concatenate([np.asarray(system.bonds, int).reshape(-1, 2), np.asarray(system.excl13, int).reshape(-1, 2)])
    excl_key = np.unique(np.minimum(excl[:, 0], excl[:, 1]).astype(np.int64) * n + np.maximum(excl[:, 0], excl[:, 1]))
    t = np.asarray(system.types)
    key_index = {pair: i for i, pair in enumerate(system.pair_index)}
    out = np.zeros((len(system.pair_keys), len(edges) - 1))
    sel_frames = frames[::stride] if len(frames) > stride else frames
    for pos, box in sel_frames:
        box = np.asarray(box, float)[:3]
        x = np.mod(pos, box)
        pairs = cKDTree(x, boxsize=box).query_pairs(rc, output_type='ndarray')
        i, j = np.minimum(pairs[:, 0], pairs[:, 1]), np.maximum(pairs[:, 0], pairs[:, 1])
        keep = ~np.isin(i.astype(np.int64) * n + j, excl_key)
        i, j = i[keep], j[keep]
        d = x[i] - x[j]
        d -= box * np.round(d / box)
        r = np.linalg.norm(d, axis=1)
        ta, tb = np.minimum(t[i], t[j]), np.maximum(t[i], t[j])
        for (a, b), k in key_index.items():
            m = (ta == a) & (tb == b)
            out[k] += np.histogram(r[m], bins=edges)[0]
    out /= max(len(sel_frames), 1)
    return {k: out[i] for i, k in enumerate(system.pair_keys)}, edges


BREAK_KT = 50.0       # a bond is broken beyond the length where its table rises 50 k_B T above its minimum
                      # (protocol v1.3; v1 - v1.2 used a fixed 100 kJ/mol)


class BrokenBondError(RuntimeError):
    """A bond stretched past its break length: the model (or a table) has failed; the task must stop."""


def break_lengths(bond_pots, T):
    """{bond type: length (nm)} where the bond table of state T first exceeds its minimum + 50 k_B T on the long side
    (the table end if it never does)."""
    from .fastmd_io import kT
    rise = BREAK_KT * kT(T)
    out = {}
    for k, p in bond_pots.items():
        j = int(np.argmin(p.u))
        above = np.flatnonzero(p.u[j:] > p.u[j] + rise)
        out[k] = float(p.x[j + above[0]]) if len(above) else float(p.x[-1])
    return out


CATASTROPHIC = 2.0     # x break length: broken at once, whatever the persistence


def check_bonds(cg, limits, where):
    """Raise BrokenBondError for broken bonds. Protocol v1.1 (for stiff bonds the break length lies only a few
    standard deviations from the mean, so one transient excursion can cross it): a bond is broken when it
    is beyond its break length in two consecutive analysed frames or beyond CATASTROPHIC x the break length in any
    frame; a single-frame excursion is not a broken bond. Without per-frame data (bond_persistent_over) the v1 rule
    (longest bond beyond the break length) applies."""
    if 'bond_persistent_over' in cg:
        bad = {k: (cg['bond_max_nm'][k], limits[k], cg['bond_persistent_over'][k]) for k in limits
               if cg['bond_persistent_over'][k] > 0 or cg['bond_max_nm'][k] > CATASTROPHIC * limits[k]}
    else:
        bad = {k: (cg['bond_max_nm'][k], limits[k], None) for k in limits if cg['bond_max_nm'][k] > limits[k]}
    if bad:
        raise BrokenBondError(f'{where}: broken bonds ' + ', '.join(
            f'{k} longest {v:.2f} nm > break length {lim:.2f} nm'
            + (f' ({n} bond-frame pairs beyond it)' if n is not None else '') for k, (v, lim, n) in bad.items()))


def save(path, cg):
    arrays = {f'{c}/{k}': v for c in ('rdf', 'bond', 'angle', 'dihedral', 'pairhist') if c in cg
              for k, v in cg[c].items()}
    if 'pairhist_edges' in cg:
        arrays['pairhist_edges'] = cg['pairhist_edges']
    np.savez_compressed(Path(path), volume_nm3=cg['volume_nm3'], **arrays)
