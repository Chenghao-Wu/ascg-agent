"""Tabulated CG dihedrals for the IBI (needs fastMD dihedral_table, upstream since fastMD
PR #3): every path i-j-k-l of three bonds is a dihedral, typed by its bead types (a-b-c-d and d-c-b-a are
the same type; phi is invariant under reversal). phi follows the engine: Blondel-Karplus, IUPAC (cis 0, trans
180 deg), in (-180, 180].

AA targets: per-frame probability densities P(phi) (1/rad) on NBINS bins over [-pi, pi] from the mapped-AA
trajectories (mapped_T*.npz), averaged, with block standard errors; written to <task>/targets/dihedrals_T*.npz.
BI start: W_s(phi) = -k_B T_s ln P_s(phi) (uniform measure in phi), unsampled bins floored at FLOOR_REL x max P,
smoothed periodically, on a 1-degree table grid (cubic periodic interpolation), A + B T from the fit's states.
IBI: the same per-state log-ratio correction as bonds / angles, periodic (unsupported bins interpolated across
periodically), mapped onto A and B.

Styles: 'table' is the plain U(phi); 'cbt' (the default: a plain table can blow up when a bond
angle nears 180 deg, through the 1/sin(theta) of the torsion gradient) is the combined bending-torsion
E = sin^3(theta1) sin^3(theta2) U(phi) (Bulacu et al., JCTC 9, 3282 (2013); fastMD `dihedral_table ... cbt`).
U is still the torsion table that IBI adjusts, with two CBT-specific choices:
* the prefactor w = sin^3 theta1 sin^3 theta2 averages well below 1 (PCL AA: 0.08-0.22 per type), which scales
  both the response of P(phi) to U and the effect of the BI start; the BI start and each correction are divided
  by the AA mean <w> of the type and state (floored at W_FLOOR, i.e. at most a 10x gain);
* the energy zero of U is no longer arbitrary: w U also acts on the bond angles (U > 0 everywhere would push them
  straight), so each state's start and corrections are shifted to zero mean over the AA P(phi).
"""
from __future__ import annotations

import numpy as np

from ..mapping.cg_mapping import block_se
from .tables import TPotential, solve_ab

NBINS = 72                 # 5-degree bins: the CG dihedral distributions are broad
FLOOR_REL = 1e-3
TABLE_DEG = 1.0
STYLES = ('table', 'cbt')
W_FLOOR = 0.1              # CBT: floor of the mean prefactor <sin^3 theta1 sin^3 theta2> used as the IBI gain


def style_of(setting):
    """Dihedral style of a run's --dihedrals setting (None: no dihedrals; True: the first runs, plain tables)."""
    return 'table' if setting is True else (setting or None)


def enumerate_dihedrals(system):
    """[(i, j, k, l)] for every path of three bonds, the type index of each and the type names."""
    adj = [[] for _ in range(system.n)]
    for a, b in system.bonds:
        adj[a].append(int(b))
        adj[b].append(int(a))
    quads = set()
    for j in range(system.n):
        for k in adj[j]:
            if k <= j:
                continue
            for i in adj[j]:
                if i == k:
                    continue
                for l in adj[k]:
                    if l == j or l == i:
                        continue
                    q = (i, j, k, l)
                    quads.add(q if q <= q[::-1] else q[::-1])
    quads = np.array(sorted(quads), int).reshape(-1, 4)
    tn = [system.type_names[t] for t in system.types]
    label = lambda q: '-'.join(min([tn[x] for x in q], [tn[x] for x in q[::-1]]))
    names = sorted({label(q) for q in quads})
    return quads, np.array([names.index(label(q)) for q in quads], int), names


def phi(pos, quads, box=None):
    """Dihedral angles (rad) of quads; box: minimum image (wrapped positions), None: unwrapped positions."""
    mi = (lambda d: d) if box is None else (lambda d: d - box * np.round(d / box))
    F = mi(pos[quads[:, 0]] - pos[quads[:, 1]])
    G = mi(pos[quads[:, 1]] - pos[quads[:, 2]])
    H = mi(pos[quads[:, 3]] - pos[quads[:, 2]])
    A, B = np.cross(F, G), np.cross(H, G)
    gn = np.linalg.norm(G, axis=1)
    return np.arctan2((np.cross(B, A) * G).sum(1) / gn, (A * B).sum(1))


def cbt_prefactor(pos, quads, box=None):
    """sin^3 theta1 sin^3 theta2 of each dihedral (theta1 = angle i-j-k, theta2 = angle j-k-l)."""
    mi = (lambda d: d) if box is None else (lambda d: d - box * np.round(d / box))
    F = mi(pos[quads[:, 0]] - pos[quads[:, 1]])
    G = mi(pos[quads[:, 1]] - pos[quads[:, 2]])
    H = mi(pos[quads[:, 3]] - pos[quads[:, 2]])
    n = lambda v: np.linalg.norm(v, axis=1)
    s1 = n(np.cross(F, G)) / (n(F) * n(G))
    s2 = n(np.cross(H, G)) / (n(H) * n(G))
    return s1**3 * s2**3


def edges(nbins=NBINS):
    return np.linspace(-np.pi, np.pi, nbins + 1)


def frame_density(ph, dtype, n_types, nbins=NBINS):
    """[n_types, nbins] probability densities (1/rad) of one frame."""
    e = edges(nbins)
    out = np.zeros((n_types, nbins))
    for t in range(n_types):
        h, _ = np.histogram(ph[dtype == t], bins=e)
        out[t] = h / max(h.sum(), 1) / (e[1] - e[0])
    return out


def aa_targets(ref_dir, quads, dtype, names, temperatures=(400, 500), nbins=NBINS, n_blocks=10):
    """{T: {'phi': centres, 'dihedral/<type>': P, 'dihedral_se/<type>': SE, 'dihedral_w/<type>': <w>}} from
    mapped_T*.npz (<w>: mean CBT prefactor)."""
    from pathlib import Path
    out = {}
    for T in temperatures:
        from .states import label
        m = np.load(Path(ref_dir)/f'mapped_T{label(T)}.npz')
        pos = [p.astype(float) for p in m['pos_nm']]
        fr = np.array([frame_density(phi(p, quads), dtype, len(names), nbins) for p in pos])
        w = np.array([cbt_prefactor(p, quads) for p in pos])
        e = edges(nbins)
        d = {'phi': 0.5 * (e[1:] + e[:-1])}
        for t, k in enumerate(names):
            d[f'dihedral/{k}'] = fr[:, t].mean(0)
            d[f'dihedral_se/{k}'] = block_se(fr[:, t], n_blocks)
            d[f'dihedral_w/{k}'] = np.float64(w[:, dtype == t].mean())
        out[T] = d
    return out


def _periodic_smooth(y, sigma_bins):
    if sigma_bins <= 0:
        return y
    k = np.arange(-int(4 * sigma_bins), int(4 * sigma_bins) + 1)
    w = np.exp(-0.5 * (k / sigma_bins)**2)
    w /= w.sum()
    n = len(y)
    return np.array([np.sum(w * y[(i + k) % n]) for i in range(n)])


def _periodic_on_grid(centres, y, grid):
    """Cubic periodic interpolation of bin values onto grid (both in rad, period 2 pi)."""
    from scipy.interpolate import CubicSpline
    xs = np.concatenate([centres, [centres[0] + 2 * np.pi]])
    ys = np.concatenate([y, [y[0]]])
    g = (grid - centres[0]) % (2 * np.pi) + centres[0]
    return CubicSpline(xs, ys, bc_type='periodic')(g)


def table_grid():
    return np.deg2rad(np.arange(-180.0, 180.0 + TABLE_DEG / 2, TABLE_DEG))


def cbt_gain(targets_T, k):
    """CBT: 1 / max(<sin^3 theta1 sin^3 theta2>_AA, W_FLOOR) of type k in one state."""
    return 1.0 / max(float(targets_T[f'dihedral_w/{k}']), W_FLOOR)


def _centre(y, p):
    """y shifted to zero mean over the distribution p (same bins)."""
    return y - np.sum(p * y) / np.sum(p)


def bi_tdihedrals(targets, names, kts, smooth_bins=1.0, style='table', weights=None):
    """T-dependent BI start {type: TPotential('dihedral')} from the AA targets of each state; style 'cbt': scaled
    by cbt_gain and centred on the AA P(phi) (module docstring)."""
    grid = table_grid()
    out = {}
    for k in names:
        u_by_T = {}
        for T, kt in kts.items():
            p0 = np.asarray(targets[T][f'dihedral/{k}'], float)
            p = np.maximum(p0, FLOOR_REL * p0.max())
            w = _periodic_smooth(-kt * np.log(p), smooth_bins)
            w = w - w.min()
            if style == 'cbt':
                w = _centre(w * cbt_gain(targets[T], k), p0)
            u = _periodic_on_grid(targets[T]['phi'], w, grid)
            u[-1] = u[0]
            u_by_T[T] = u
        a, b = solve_ab(u_by_T, weights)
        out[k] = TPotential('dihedral', grid.copy(), a, b)
    return out


def state_correction(centres, cg, aa, kt, *, lam, cap, floor_rel=FLOOR_REL, smooth_bins=1.0, gain=None):
    """lambda k_B T clip(ln(P_CG / P_AA), -c, c) on the periodic bins; unsupported bins interpolated periodically.
    gain (CBT, cbt_gain): the correction is multiplied by it and centred on the AA P(phi)."""
    aa, cg = np.asarray(aa, float), np.asarray(cg, float)
    sup = (aa > floor_rel * aa.max()) & (cg > floor_rel * aa.max())
    corr = np.zeros(len(centres))
    if not sup.any():
        return corr
    with np.errstate(divide='ignore', invalid='ignore'):
        corr[sup] = lam * kt * np.clip(np.log(cg[sup] / aa[sup]), -cap, cap)
    idx = np.flatnonzero(sup)
    ext_x = np.concatenate([centres[idx] - 2 * np.pi, centres[idx], centres[idx] + 2 * np.pi])
    corr = _periodic_smooth(np.interp(centres, ext_x, np.tile(corr[idx], 3)), smooth_bins)
    return corr if gain is None else _centre(gain * corr, aa)


def apply_tdihedral(tpot, centres, corr_by_T, weights=None):
    """Per-state periodic corrections mapped onto A and B."""
    d = {T: _periodic_on_grid(centres, c, tpot.x) for T, c in corr_by_T.items()}
    for T in d:
        d[T][-1] = d[T][0]
    da, db = solve_ab(d, weights)
    return TPotential('dihedral', tpot.x.copy(), tpot.a + da, tpot.b + db)
