"""LAMMPS data file (AutoPoly OPLS-AA, units real) -> GROMACS .gro/.top with the same energy.

Supported styles (those AutoPoly/moltemplate OPLS-AA produce): pair lj/cut/coul/long with
geometric mixing and special_bonds lj/coul 0 0 0.5; bond/angle harmonic; dihedral opls;
improper cvff. The data file must contain the coefficient sections (write_data default).

Conversions (kcal -> kJ x 4.184, A -> nm / 10):
  bond   LAMMPS K(r-r0)^2               -> GROMACS type 1, kb = 2 K           (kJ/mol/nm^2)
  angle  LAMMPS K(t-t0)^2               -> GROMACS type 1, kt = 2 K           (kJ/mol/rad^2)
  dihed  LAMMPS opls 1/2 sum K_n(...)   -> GROMACS type 5 (Fourier), C_n = K_n (same form)
  improp LAMMPS cvff K[1 + d cos(n phi)] -> GROMACS type 4, k = K, phi_s = 0 (d=+1) / 180 (d=-1)
  1-4    [pairs] from bond-graph distance 3 (as LAMMPS special lists), fudgeLJ = fudgeQQ = 0.5,
         comb-rule 3 (geometric sigma and epsilon), nrexcl 3.
Coordinates are unwrapped with the image flags, so every molecule is whole.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

KJ = 4.184
SECTIONS = ('Masses', 'Pair Coeffs', 'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs',
            'Atoms', 'Velocities', 'Bonds', 'Angles', 'Dihedrals', 'Impropers')
ELEMENT_BY_MASS = {1: 'H', 12: 'C', 14: 'N', 16: 'O', 19: 'F', 28: 'Si', 32: 'S', 35: 'Cl'}


def read_lammps_data(path):
    lines = Path(path).read_text().splitlines()
    box, sec, cur = [], {}, None
    for raw in lines[1:]:
        line = raw.split('#')[0].strip()
        head = raw.split('#')[0].strip()
        if head in SECTIONS:
            cur = head
            sec[cur] = []
            if cur.endswith('Coeffs'):
                style = raw.split('#')[1].strip() if '#' in raw else ''
                sec[cur + ' style'] = style
            continue
        if not line:
            continue
        if cur is None:
            p = line.split()
            if len(p) == 4 and p[2].endswith('lo'):
                box.append((float(p[0]), float(p[1])))
            continue
        sec[cur].append(line.split())
    styles = {k: v for k, v in sec.items() if k.endswith(' style')}
    expected = {'Pair Coeffs style': 'lj/cut/coul/long', 'Bond Coeffs style': 'harmonic',
                'Angle Coeffs style': 'harmonic', 'Dihedral Coeffs style': 'opls', 'Improper Coeffs style': 'cvff'}
    for k, v in expected.items():
        if k in styles and styles[k] != v:
            raise ValueError(f'{k} is {styles[k]!r}; this converter handles {v}')
    return np.array(box), sec


def one_four_pairs(n_atoms, bonds):
    """Pairs at bond-graph distance exactly 3 (LAMMPS special 1-4 list semantics)."""
    nbr = defaultdict(list)
    for a, b in bonds:
        nbr[a].append(b)
        nbr[b].append(a)
    pairs = set()
    for s in range(n_atoms):
        if s not in nbr:
            continue
        dist = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] == 3:
                continue
            for v in nbr[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        pairs.update((s, v) for v, d in dist.items() if d == 3 and s < v)
    return sorted(pairs)


def read_settings(path, sec):
    """Coefficients from a moltemplate/AutoPoly system.in.settings (numeric types) into `sec`."""
    keys = {'pair_coeff': 'Pair Coeffs', 'bond_coeff': 'Bond Coeffs', 'angle_coeff': 'Angle Coeffs',
            'dihedral_coeff': 'Dihedral Coeffs', 'improper_coeff': 'Improper Coeffs'}
    for raw in Path(path).read_text().splitlines():
        p = raw.split('#')[0].split()
        if not p or p[0] not in keys:
            continue
        if p[0] == 'pair_coeff':
            if p[1] != p[2]:
                raise ValueError(f'Only i-i pair coefficients are supported (geometric mixing): {raw}')
            sec.setdefault('Pair Coeffs', []).append([p[1]] + p[3:5])
        else:
            sec.setdefault(keys[p[0]], []).append(p[1:])


def declash(xyz, box, bonds, *, min_dist=0.8, passes=50, seed=0):
    """Push apart non-bonded atom pairs (1-2 and 1-3 excluded) closer than min_dist (A).

    AutoPoly's rigid-unit chain growth leaves near-coincident atoms (down to 0.04 A) whose
    forces overflow in single precision, so steepest descent cannot start. Each pass moves
    both atoms of every short pair apart along their separation (random direction if they
    coincide) to min_dist; returns (coordinates, pairs fixed in the first pass, passes used).
    """
    from scipy.spatial import cKDTree
    x = np.array(xyz, float)
    n = len(x)
    nbr = defaultdict(set)
    for a, b in bonds:
        nbr[a].add(b)
        nbr[b].add(a)
    excl = set()
    for a in range(n):
        for b in nbr[a]:
            excl.add((min(a, b), max(a, b)))
            for c in nbr[b]:
                if c != a:
                    excl.add((min(a, c), max(a, c)))
    rng = np.random.default_rng(seed)
    first = None
    for it in range(passes):
        t = cKDTree(np.mod(x, box), boxsize=box)
        pairs = [p for p in t.query_pairs(min_dist) if p not in excl]
        if first is None:
            first = len(pairs)
        if not pairs:
            return x, first, it
        for i, j in pairs:
            d = x[i] - x[j]
            d -= box * np.round(d / box)
            r = np.linalg.norm(d)
            u = d / r if r > 1e-6 else rng.normal(size=3) / np.sqrt(3)
            shift = 0.5 * (min_dist - r) + 1e-3
            x[i] += shift * u
            x[j] -= shift * u
    raise RuntimeError(f'declash: pairs closer than {min_dist} A remain after {passes} passes')


# OPLS-AA gives every hydrogen on O/N/S sigma = epsilon = 0 (the parent
# atom's LJ sphere is assumed to shield it). With geometric mixing such an H has no repulsion with
# any atom, and an attractive 1-4 pair whose donor-acceptor pair is excluded (POM's hemiacetal end
# H-O-C-O, amide H-N-C=O) is unbounded below and collapsed at 600 K. These hydrogens get
# sigma 0.100 nm, epsilon 0.1925 kJ/mol: CHARMM's polar-H epsilon, and a sigma that under OPLS's
# geometric mixing reproduces CHARMM's H...O contact (sigma_HO ~0.17-0.18 nm; CHARMM mixes
# arithmetically). CHARMM's own sigma (0.04 nm) left a 1-4 minimum at 0.084 nm, -134 kJ/mol
# (scan of the 1-4 pair energy); with 0.10 nm it is at 0.138 nm, -82 kJ/mol. At hydrogen-bond
# distances (0.18-0.19 nm) the added H...O LJ is ~ -0.3 kJ/mol.
POLAR_H_LJ = (0.100, 0.1925)


def fourier_to_rb(c1, c2, c3, c4):
    """OPLS Fourier coefficients -> Ryckaert-Bellemans C0..C5 (GROMACS manual; psi = phi - 180).
    GROMACS converts type 5 to RB internally; writing type 3 lets GPU listed-force kernels take them."""
    return (c2 + 0.5 * (c1 + c3), 0.5 * (-c1 + 3 * c3), -c2 + 4 * c4, -2 * c3, -4 * c4, 0.0)


def convert(data_file, gro, top, *, settings=None, title='converted from LAMMPS', declash_min_A=None,
            hydrogen_mass=None, polar_h_lj=None, dihedral_style='fourier'):
    """declash_min_A: separate near-coincident non-bonded atoms first (see `declash`).
    hydrogen_mass: hydrogen mass repartitioning (mass moved from the bonded heavy atom; total mass,
    potential and hence equilibrium structure unchanged; slows only the fastest motions)."""
    box, sec = read_lammps_data(data_file)
    if settings is not None:
        read_settings(settings, sec)
    L = (box[:, 1] - box[:, 0])
    mass = {int(r[0]): float(r[1]) for r in sec['Masses']}
    pair = {int(r[0]): (float(r[1]), float(r[2])) for r in sec['Pair Coeffs']}
    bondc = {int(r[0]): (float(r[1]), float(r[2])) for r in sec['Bond Coeffs']}
    anglec = {int(r[0]): (float(r[1]), float(r[2])) for r in sec['Angle Coeffs']}
    dihc = {int(r[0]): tuple(float(x) for x in r[1:5]) for r in sec.get('Dihedral Coeffs', [])}
    impc = {int(r[0]): (float(r[1]), int(r[2]), int(r[3])) for r in sec.get('Improper Coeffs', [])}
    atoms = sorted(sec['Atoms'], key=lambda r: int(r[0]))
    n = len(atoms)
    ids = np.array([int(r[0]) for r in atoms])
    if not np.array_equal(ids, np.arange(1, n + 1)):
        raise ValueError('Atom ids must be 1..N')
    mol = np.array([int(r[1]) for r in atoms])
    typ = np.array([int(r[2]) for r in atoms])
    q = np.array([float(r[3]) for r in atoms])
    xyz = np.array([[float(x) for x in r[4:7]] for r in atoms])
    img = np.array([[int(x) for x in r[7:10]] for r in atoms]) if len(atoms[0]) >= 10 else np.zeros((n, 3), int)
    xyz = xyz + img * L - box[:, 0]            # unwrapped, box origin at 0

    def idx(r, k):
        return [int(x) - 1 for x in r[2:2 + k]]
    bonds = [(int(r[1]), *idx(r, 2)) for r in sec.get('Bonds', [])]
    angles = [(int(r[1]), *idx(r, 3)) for r in sec.get('Angles', [])]
    dihs = [(int(r[1]), *idx(r, 4)) for r in sec.get('Dihedrals', [])]
    imps = [(int(r[1]), *idx(r, 4)) for r in sec.get('Impropers', [])]
    pairs14 = one_four_pairs(n, [(a, b) for _, a, b in bonds])
    info_extra = {}
    if polar_h_lj:
        sig_nm, eps_kj = polar_h_lj
        zero = sorted(t for t, (e, sg) in pair.items() if e == 0 and sg == 0 and mass[t] < 1.5)
        for t in zero:
            pair[t] = (eps_kj / KJ, sig_nm * 10)
        info_extra['polar_h_lj'] = {'types': zero, 'atoms': int(np.isin(typ, zero).sum()),
                                    'sigma_nm': sig_nm, 'epsilon_kJ_mol': eps_kj}
    if declash_min_A:
        xyz, n_fixed, n_pass = declash(xyz, L, [(a, b) for _, a, b in bonds], min_dist=declash_min_A)
        info_extra['declash'] = {'min_A': declash_min_A, 'pairs_below': n_fixed, 'passes': n_pass}
    atom_mass = np.array([mass[t] for t in typ], float)
    if hydrogen_mass:
        heavy_of = {}
        for _, a, b in bonds:
            for h, o in ((a, b), (b, a)):
                if atom_mass[h] < 1.5 and atom_mass[o] >= 1.5:
                    heavy_of[h] = o
        for h, o in heavy_of.items():
            dm = hydrogen_mass - atom_mass[h]
            atom_mass[h] += dm
            atom_mass[o] -= dm
        info_extra['hydrogen_mass'] = hydrogen_mass

    # per-molecule bookkeeping (GROMACS indices are molecule-local, 1-based)
    mols = sorted(set(mol))
    members = {m: np.where(mol == m)[0] for m in mols}
    local = np.empty(n, int)
    for m, ix in members.items():
        if np.any(np.diff(ix) != 1):
            raise ValueError(f'Molecule {m} atoms are not contiguous')
        local[ix] = np.arange(1, len(ix) + 1)
    by_mol = defaultdict(lambda: defaultdict(list))
    for t, a, b in bonds:
        by_mol[mol[a]]['bonds'].append((t, a, b))
    for a, b in pairs14:
        by_mol[mol[a]]['pairs'].append((a, b))
    for t, a, b, c in angles:
        by_mol[mol[a]]['angles'].append((t, a, b, c))
    for t, a, b, c, d in dihs:
        by_mol[mol[a]]['dihedrals'].append((t, a, b, c, d))
    for t, a, b, c, d in imps:
        by_mol[mol[a]]['impropers'].append((t, a, b, c, d))

    def elem(t):
        return ELEMENT_BY_MASS.get(int(round(mass[t])), 'X')

    out = [f'; {title}', '; generated by ascg_core.aa.lmp2gmx from ' + str(data_file), '',
           '[ defaults ]', '; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ', '  1  3  yes  0.5  0.5', '',
           '[ atomtypes ]', '; name  mass  charge  ptype  sigma(nm)  epsilon(kJ/mol)']
    for t in sorted(mass):
        eps, sig = pair[t]
        out.append(f'  t{t}  {mass[t]:.4f}  0.0  A  {sig / 10:.6f}  {eps * KJ:.6f}')
    for m in mols:
        ix = members[m]
        d = by_mol[m]
        out += ['', '[ moleculetype ]', f'  M{m}  3', '', '[ atoms ]',
                '; nr type resnr res atom cgnr charge mass']
        for i in ix:
            out.append(f'  {local[i]}  t{typ[i]}  1  M{m % 10000}  {elem(typ[i])}{local[i] % 1000}  {local[i]}  '
                       f'{q[i]:.10f}  {atom_mass[i]:.4f}')
        out += ['', '[ bonds ]']
        for t, a, b in d['bonds']:
            K, r0 = bondc[t]
            out.append(f'  {local[a]}  {local[b]}  1  {r0 / 10:.6f}  {2 * K * KJ * 100:.4f}')
        out += ['', '[ pairs ]']
        out += [f'  {local[a]}  {local[b]}  1' for a, b in d['pairs']]
        out += ['', '[ angles ]']
        for t, a, b, c in d['angles']:
            K, th0 = anglec[t]
            out.append(f'  {local[a]}  {local[b]}  {local[c]}  1  {th0:.6f}  {2 * K * KJ:.6f}')
        if dihedral_style == 'rb':
            out += ['', '[ dihedrals ]', '; OPLS proper dihedrals as Ryckaert-Bellemans (type 3, exact)']
            for t, a, b, c, e in d['dihedrals']:
                C = fourier_to_rb(*(x * KJ for x in dihc[t]))
                out.append(f'  {local[a]}  {local[b]}  {local[c]}  {local[e]}  3  ' + '  '.join(f'{x:.6f}' for x in C))
        else:
            out += ['', '[ dihedrals ]', '; Fourier (OPLS) proper dihedrals']
            for t, a, b, c, e in d['dihedrals']:
                C = dihc[t]
                out.append(f'  {local[a]}  {local[b]}  {local[c]}  {local[e]}  5  ' + '  '.join(f'{x * KJ:.6f}' for x in C))
        if d['impropers']:
            out += ['', '[ dihedrals ]', '; cvff impropers as periodic impropers (type 4)']
            for t, a, b, c, e in d['impropers']:
                K, dd, nn = impc[t]
                out.append(f'  {local[a]}  {local[b]}  {local[c]}  {local[e]}  4  {0.0 if dd > 0 else 180.0:.1f}  '
                           f'{K * KJ:.6f}  {nn}')
    out += ['', '[ system ]', f'  {title}', '', '[ molecules ]']
    out += [f'  M{m}  1' for m in mols]
    Path(top).write_text('\n'.join(out) + '\n')

    # 5-decimal positions (GROMACS reads the precision from the decimal-point spacing); 0.001 nm
    # rounding would add ~0.05 kJ/mol per stiff bond. Velocities are not written (generated by mdp).
    g = [title, f'{n:5d}']
    for i in range(n):
        g.append(f'{1:5d}{"M" + str(mol[i] % 10000):<5s}{elem(typ[i]) + str(local[i] % 1000):>5s}{(i + 1) % 100000:5d}'
                 + ''.join(f'{x / 10:10.5f}' for x in xyz[i]))
    g.append(''.join(f'{x / 10:10.5f}' for x in L))
    Path(gro).write_text('\n'.join(g) + '\n')
    return {'atoms': n, 'molecules': len(mols), 'pairs14': len(pairs14), 'box_nm': (L / 10).tolist(), **info_extra}


def add_decoupled_states(top_in, top_out):
    """Copy of a converted topology in which every atom has a B state without non-bonded
    interactions (LJ epsilon 0, charge 0, same mass): with free-energy = yes and soft-core,
    lambda 1 -> 0 switches all non-bonded interactions on smoothly (overlap relaxation)."""
    out, section = [], None
    for line in Path(top_in).read_text().splitlines():
        s = line.strip()
        if s.startswith('['):
            section = s.strip('[] ').lower()
            out.append(line)
            continue
        if not s or s.startswith(';'):
            out.append(line)
            continue
        p = s.split(';')[0].split()
        if section == 'atomtypes':
            # same columns as the A type (lmp2gmx: name mass charge ptype sigma eps; LigParGen adds a
            # bond type after the name), epsilon 0 and charge 0
            out.append(line)
            b = [p[0] + '_B'] + p[1:]
            b[-1] = '0.0'
            b[-4] = '0.0'
            out.append('  ' + '  '.join(b))
        elif section == 'atoms':
            out.append(f'{line.split(";")[0].rstrip()}  {p[1]}_B  0.0  {p[7]}')
        else:
            out.append(line)
    Path(top_out).write_text('\n'.join(out) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('data')
    ap.add_argument('--gro', required=True)
    ap.add_argument('--top', required=True)
    ap.add_argument('--settings', default=None, help='system.in.settings when the data file has no Coeffs')
    a = ap.parse_args(argv)
    print(convert(a.data, a.gro, a.top, settings=a.settings))


if __name__ == '__main__':
    main()
