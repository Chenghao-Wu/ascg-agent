"""AA -> CG mapping of AutoPoly-built systems and mapped-AA reference targets.

The atom layout is read from the moltemplate inputs AutoPoly wrote
(`moltemplate/input/system.lt`, `poly_N.lt`, `monomer_*.lt`): moltemplate numbers
atoms chain by chain, unit by unit, in each unit's `Data Atoms` order. That
layout is checked against the LAMMPS data file (elements, molecule ids,
junction bonds) before any bead is defined. Beads follow
the mapping's repeat-unit groups (the system library's, converted by mapping.formats): heavy-atom index groups per
unit in SMILES order (matched to
the .lt atoms by graph isomorphism, since AutoPoly moves connection atoms first);
hydrogens join the bead of the heavy atom they are bonded to.

Lengths: AA files are in Angstrom; everything returned here is in nm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import json
import re
from pathlib import Path

import numpy as np

ANGSTROM_PER_NM = 10.0
AMU_PER_NM3_TO_G_CM3 = 1.66053906660e-3
KB_KJ_PER_MOL_K = 0.00831446261815324
BAR_PER_KJ_MOL_NM3 = 16.6053906660
ELEMENT_BY_MASS = {1.008: 'H', 12.011: 'C', 14.007: 'N', 15.999: 'O', 18.998: 'F', 28.086: 'Si', 28.085: 'Si',
                   32.06: 'S', 32.065: 'S', 35.453: 'Cl'}
POSITION = {'le': 'first', 'i': 'middle', 're': 'last', 'single': 'single'}


class MappingError(ValueError):
    pass


# ---------------------------------------------------------------- AA layout
def _element(name: str) -> str:
    m = re.match(r'[A-Z][a-z]?', name)
    if not m:
        raise MappingError(f'Cannot read element from atom name {name!r}')
    return m.group(0)


def read_unit_lt(path) -> tuple[list[str], list[tuple[int, int]]]:
    """Atom names (Data Atoms order) and intra-unit bonds of one AutoPoly monomer .lt."""
    text = Path(path).read_text()
    atoms = re.search(r'write\("Data Atoms"\)\s*\{(.*?)\n\s*\}', text, re.S)
    if not atoms:
        raise MappingError(f'No Data Atoms block in {path}')
    names = re.findall(r'\$atom:(\S+)\s+\$mol', atoms.group(1))
    index = {n: i for i, n in enumerate(names)}
    bonds_block = re.search(r"write\('Data Bond List'\)\s*\{(.*?)\n\s*\}", text, re.S)
    bonds = []
    if bonds_block:
        for a, b in re.findall(r'\$atom:(\S+)\s+\$atom:(\S+)', bonds_block.group(1)):
            bonds.append((index[a], index[b]))
    return names, bonds


@dataclass
class Unit:
    variant: str
    position: str            # first | middle | last | single (a whole molecule of a liquid)
    names: list[str]
    bonds: list[tuple[int, int]]
    first_atom: int          # 0-based atom index in the data file
    code: str = ''

    @property
    def elements(self):
        return [_element(n) for n in self.names]

    @property
    def heavy(self):
        return [i for i, e in enumerate(self.elements) if e != 'H']


@dataclass
class Layout:
    chains: list[list[Unit]]
    n_atoms: int


def read_layout(system_dir) -> Layout:
    """Chain/unit/atom layout from AutoPoly's moltemplate inputs. Polymers are poly_N instances of monomer units;
    molecular liquids are molecule instances of one 'single' unit each (one chain of one unit per molecule)."""
    inp = Path(system_dir)/'moltemplate'/'input'
    system_lt = (inp/'system.lt').read_text()
    polys = re.findall(r'^\s*\S+\s*=\s*new\s+(poly_\d+)\b', system_lt, re.M)
    if not polys:
        return _molecule_layout(inp, system_lt)
    cache = {}
    chains, offset = [], 0
    for poly in polys:
        text = (inp/f'{poly}.lt').read_text()
        mons = sorted((int(i), v) for i, v in re.findall(r'monomer\[(\d+)\]\s*=\s*new\s+(\w+)', text))
        if [i for i, _ in mons] != list(range(len(mons))):
            raise MappingError(f'Non-contiguous monomer indices in {poly}')
        units = []
        for _, variant in mons:
            if variant not in cache:
                cache[variant] = read_unit_lt(inp/f'{variant}.lt')
            m = re.fullmatch(r'monomer_\d+_\d+(le|i|re)(_T1)?', variant)
            if not m:
                raise MappingError(f'Unexpected unit variant name {variant}')
            names, bonds = cache[variant]
            units.append(Unit(variant, POSITION[m.group(1)], names, bonds, offset))
            offset += len(names)
        chains.append(units)
    return Layout(chains, offset)


def _molecule_layout(inp, system_lt) -> Layout:
    mols = re.findall(r'^\s*\S+\s*=\s*new\s+(monomer_[A-Za-z0-9_]+)', system_lt, re.M)
    if not mols:
        raise MappingError(f'No polymer or molecule instances in {inp/"system.lt"}')
    cache, chains, offset = {}, [], 0
    for variant in mols:
        if not re.fullmatch(r'monomer_\d+_\d+single(_T1)?', variant):
            raise MappingError(f'Unexpected molecule variant name {variant}')
        if variant not in cache:
            cache[variant] = read_unit_lt(inp/f'{variant}.lt')
        names, bonds = cache[variant]
        chains.append([Unit(variant, 'single', names, bonds, offset)])
        offset += len(names)
    return Layout(chains, offset)


def read_data(path):
    """Masses, per-atom (type, molecule, charge, xyz in A), bonds and box from a LAMMPS data file."""
    lines = Path(path).read_text().splitlines()
    sections = {'Masses', 'Atoms', 'Bonds', 'Angles', 'Dihedrals', 'Impropers', 'Velocities'}
    section, masses, atoms, bonds, box = None, {}, {}, [], {}
    for raw in lines:
        s = raw.split('#', 1)[0].strip()
        if s in sections:
            section = s
            continue
        if not s:
            continue
        p = s.split()
        if section is None:
            if len(p) == 4 and p[2].endswith('lo'):
                box[p[2][0]] = (float(p[0]), float(p[1]))
        elif section == 'Masses':
            masses[int(p[0])] = float(p[1])
        elif section == 'Atoms':
            atoms[int(p[0])] = (int(p[2]), int(p[1]), float(p[3]), [float(x) for x in p[4:7]])
        elif section == 'Bonds':
            bonds.append((int(p[2]), int(p[3])))
    ids = sorted(atoms)
    if ids != list(range(1, len(ids) + 1)):
        raise MappingError('Atom ids must be 1..N')
    types = np.array([atoms[i][0] for i in ids])
    return {'mass': np.array([masses[t] for t in types]),
            'element': [ELEMENT_BY_MASS.get(round(masses[t], 3), '?') for t in types],
            'molecule': np.array([atoms[i][1] for i in ids]),
            'charge': np.array([atoms[i][2] for i in ids]),
            'xyz_A': np.array([atoms[i][3] for i in ids]),
            'bonds': np.array(bonds, dtype=int) - 1,
            'box_A': np.array([box[c][1] - box[c][0] for c in 'xyz'])}


# ---------------------------------------------------------------- beads
def _smiles_graph(smiles: str):
    """Heavy-atom graph of a complement SMILES (SMILES heavy index -> element) and the
    heavy indices bonded to the first and last [*] (head, tail; None when absent)."""
    import networkx as nx
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise MappingError(f'Bad SMILES {smiles}')
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 0]
    k = {old: i for i, old in enumerate(heavy)}
    g = nx.Graph()
    for old in heavy:
        g.add_node(k[old], element=mol.GetAtomWithIdx(old).GetSymbol())
    for bd in mol.GetBonds():
        i, j = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        if i in k and j in k:
            g.add_edge(k[i], k[j])
    joints = [k[a.GetNeighbors()[0].GetIdx()] for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    return g, joints


def match_unit(unit: 'Unit', smiles: str):
    """Map the unit's .lt heavy atoms to the SMILES heavy-atom indices, or None.

    AutoPoly writes the head connection atom first and the tail connection atom
    second, then the rest; the match is a graph isomorphism on elements and
    bonds with those connection atoms pinned, so it does not rely on order.
    """
    import networkx as nx
    from networkx.algorithms import isomorphism
    ref, joints = _smiles_graph(smiles)
    heavy = unit.heavy
    pos = {old: i for i, old in enumerate(heavy)}
    g = nx.Graph()
    for i, old in enumerate(heavy):
        g.add_node(i, element=unit.elements[old])
    for a, b in unit.bonds:
        if a in pos and b in pos:
            g.add_edge(pos[a], pos[b])
    if g.number_of_nodes() != ref.number_of_nodes():
        return None
    pinned = {}
    if unit.position == 'single':                      # a whole molecule: no connection atoms
        pass
    elif unit.position == 'first':
        pinned = {1: joints[-1]} if len(heavy) > 1 else {}
    elif unit.position == 'middle':
        pinned = {0: joints[0], 1: joints[-1]}
    else:
        pinned = {0: joints[0]}
    matcher = isomorphism.GraphMatcher(g, ref, node_match=lambda x, y: x['element'] == y['element'])
    for iso in matcher.isomorphisms_iter():
        if all(iso[a] == b for a, b in pinned.items()):
            return [iso[i] for i in range(len(heavy))]
    return None


@dataclass
class CGTopologySpec:
    task: str
    bead_types: list[str]                # per bead
    bead_masses_amu: np.ndarray          # per bead
    bead_molecule: np.ndarray            # per bead (0-based chain index)
    atom_bead: np.ndarray                # per AA atom
    bonds: list[tuple[int, int]]
    angles: list[tuple[int, int, int]]
    type_names: list[str] = field(default_factory=list)

    @property
    def exclusions(self):
        excl = {tuple(sorted(b)) for b in self.bonds}
        excl |= {tuple(sorted((a[0], a[2]))) for a in self.angles}
        return sorted(excl)

    def bond_type(self, b):
        return '-'.join(sorted((self.bead_types[b[0]], self.bead_types[b[1]])))

    def angle_type(self, a):
        t = [self.bead_types[i] for i in a]
        return '-'.join(min(t, t[::-1]))

    def to_dict(self):
        return {'task': self.task, 'type_names': self.type_names,
                'bead_types': [self.type_names.index(t) for t in self.bead_types],
                'bead_masses_amu': self.bead_masses_amu.tolist(),
                'bead_molecule': self.bead_molecule.tolist(),
                'bonds': [list(b) for b in self.bonds], 'angles': [list(a) for a in self.angles],
                'exclusions': [list(e) for e in self.exclusions],
                'bond_types': sorted({self.bond_type(b) for b in self.bonds}),
                'angle_types': sorted({self.angle_type(a) for a in self.angles})}


def build_cg_topology(task: str, layout: Layout, data: dict, recipes: dict, mappings: dict) -> CGTopologySpec:
    """Assign every AA atom to a bead and derive CG bonds/angles; verify the layout first."""
    if layout.n_atoms != len(data['mass']):
        raise MappingError(f'Layout has {layout.n_atoms} atoms, data file {len(data["mass"])}')
    spec_units = mappings['tasks'][task]['units']
    matches = {}   # variant -> (unit code, .lt heavy position -> SMILES heavy index)
    atom_bead = np.full(layout.n_atoms, -1)
    bead_types, bead_mol = [], []
    molecule_of_chain = {}
    for c, units in enumerate(layout.chains):
        mol_ids = set(data['molecule'][units[0].first_atom:units[-1].first_atom + len(units[-1].names)])
        if len(mol_ids) != 1:
            raise MappingError(f'Chain {c} spans molecules {sorted(mol_ids)}')
        molecule_of_chain[c] = mol_ids.pop()
        for u in units:
            sl = slice(u.first_atom, u.first_atom + len(u.names))
            if [e for e in data['element'][sl]] != u.elements:
                raise MappingError(f'Element mismatch for chain {c} unit {u.variant}')
            if u.variant not in matches:
                found = [(code, m) for code in spec_units
                         if (m := match_unit(u, recipes['units'][code][u.position])) is not None]
                if len(found) != 1:
                    raise MappingError(f'Unit {u.variant} matches {len(found)} recipe units for {task}')
                matches[u.variant] = found[0]
            u.code, to_smiles = matches[u.variant]
            groups = spec_units[u.code]
            if isinstance(groups, dict):
                groups = groups[u.position]
            heavy = u.heavy
            smiles_to_lt = {s_idx: heavy[k] for k, s_idx in enumerate(to_smiles)}
            local = np.full(len(u.names), -1)
            for btype, idx in groups:
                b = len(bead_types)
                bead_types.append(btype)
                bead_mol.append(c)
                members = range(len(heavy)) if idx == 'all' else idx
                for k in members:
                    local[smiles_to_lt[k]] = b
            for i, e in enumerate(u.elements):
                if e == 'H':
                    partners = [j for a, j in u.bonds if a == i] + [a for a, j in u.bonds if j == i]
                    if len(partners) != 1:
                        raise MappingError(f'H {u.names[i]} in {u.variant} has {len(partners)} bonds')
                    local[i] = local[partners[0]]
            if (local < 0).any():
                raise MappingError(f'Unassigned atoms in {u.variant}: {[u.names[i] for i in np.where(local < 0)[0]]}')
            atom_bead[sl] = local
    masses = np.bincount(atom_bead, weights=data['mass'], minlength=len(bead_types))
    pairs = {tuple(sorted((int(atom_bead[a]), int(atom_bead[b])))) for a, b in data['bonds']
             if atom_bead[a] != atom_bead[b]}
    bonds = sorted(pairs)
    neigh = {}
    for a, b in bonds:
        neigh.setdefault(a, set()).add(b)
        neigh.setdefault(b, set()).add(a)
    angles = sorted({(min(i, k), j, max(i, k)) for j, ns in neigh.items()
                     for i, k in itertools.combinations(sorted(ns), 2)})
    names = sorted(set(bead_types))
    return CGTopologySpec(task, bead_types, masses, np.array(bead_mol), atom_bead, bonds, angles, names)


# ---------------------------------------------------------------- trajectories
def read_lammps_dump(path, n_atoms: int):
    """Yield (timestep, box_A [3], xyz_A [N,3]) from a sorted 'id xu yu zu' text dump."""
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith('ITEM: TIMESTEP'):
                raise MappingError(f'Unexpected dump line {line!r}')
            step = int(f.readline())
            f.readline()
            n = int(f.readline())
            if n != n_atoms:
                raise MappingError(f'Dump has {n} atoms, expected {n_atoms}')
            f.readline()
            box = np.array([np.diff([float(x) for x in f.readline().split()[:2]])[0] for _ in range(3)])
            header = f.readline().split()[2:]
            if header[:4] != ['id', 'xu', 'yu', 'zu']:
                raise MappingError(f'Dump columns {header}, expected id xu yu zu')
            block = np.array(''.join(f.readline() for _ in range(n)).split(), dtype=float).reshape(n, -1)
            if not np.array_equal(block[:, 0], np.arange(1, n + 1)):
                raise MappingError('Dump must be sorted by atom id (dump_modify sort id)')
            yield step, box, block[:, 1:4]


def map_frame(xyz_A, atom_mass, atom_bead, bead_mass):
    """Bead centres of mass in nm (unwrapped input gives unwrapped beads)."""
    out = np.empty((len(bead_mass), 3))
    for d in range(3):
        out[:, d] = np.bincount(atom_bead, weights=atom_mass * xyz_A[:, d], minlength=len(bead_mass))
    return out / bead_mass[:, None] / ANGSTROM_PER_NM


# ---------------------------------------------------------------- observables
def intermolecular_pair_counts(bead_types, molecule, type_names):
    """Number of intermolecular bead pairs per type pair (for RDF normalisation)."""
    t = np.asarray([type_names.index(x) for x in bead_types])
    mol = np.asarray(molecule)
    counts = {}
    n_type = np.bincount(t, minlength=len(type_names)).astype(float)
    per_mol = np.zeros((mol.max() + 1, len(type_names)))
    np.add.at(per_mol, (mol, t), 1)
    for a, b in itertools.combinations_with_replacement(range(len(type_names)), 2):
        if a == b:
            total = (n_type[a]**2 - np.sum(per_mol[:, a]**2)) / 2
        else:
            total = n_type[a] * n_type[b] - np.sum(per_mol[:, a] * per_mol[:, b])
        counts[f'{type_names[a]}-{type_names[b]}'] = total
    return counts


KERNEL_HALF_WIDTH_SIGMA = 6.0          # beyond 6 sigma a bin gets < 1.6e-8 of a value: GradCG's untruncated sum


def kernel_half_width_bins(sigma_bins):
    """Bins on each side of a value's own bin that its Gaussian reaches (0 for hard bins)."""
    return int(np.ceil(KERNEL_HALF_WIDTH_SIGMA * sigma_bins)) if sigma_bins > 0 else 0


def kernel_counts(values, lo, dx, n_bins, sigma_bins=0.0, weights=None):
    """Histogram of `values` on the grid lo + dx [0..n_bins]. sigma_bins = 0: hard bins; > 0: GradCG's kernel
    (custom_quantity: every value adds exp(-(x - c_k)^2 / 2 sigma^2) dx / sqrt(2 pi sigma^2) to every bin centre
    c_k, sigma = sigma_bins bins, not renormalised), evaluated within +- KERNEL_HALF_WIDTH_SIGMA sigma; weight
    falling off the grid is lost, values off the grid still spill into it. Same definition as
    PolymerEnergy.observables on device."""
    x = np.asarray(values, dtype=float).ravel()
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float).ravel()
    t = (x - lo) / dx
    if sigma_bins <= 0:
        k = np.floor(t).astype(int)
        ok = (k >= 0) & (k < n_bins)
        return np.bincount(k[ok], weights=w[ok], minlength=n_bins)[:n_bins].astype(float)
    h = kernel_half_width_bins(sigma_bins)
    out = np.zeros(n_bins)
    for i in range(0, len(x), 200_000):              # chunks keep the [n, 2h+1] kernel small
        tc, wc = t[i:i + 200_000], w[i:i + 200_000]
        ks = np.floor(tc).astype(int)[:, None] + np.arange(-h, h + 1)
        g = np.exp(-0.5 * ((ks + 0.5 - tc[:, None]) / sigma_bins)**2) / (np.sqrt(2 * np.pi) * sigma_bins)
        g *= wc[:, None]
        ok = (ks >= 0) & (ks < n_bins)
        out += np.bincount(ks[ok], weights=g[ok], minlength=n_bins)[:n_bins]
    return out


def trapz_normalised(u, dx):
    """GradCG's BDF / ADF normalisation: u / trapz(u, bin centres) along the last axis (numpy or jax arrays;
    empty rows stay 0)."""
    s = dx * (u.sum(-1, keepdims=True) - 0.5 * (u[..., :1] + u[..., -1:]))
    return u / (s + (s == 0))


def frame_rdf_counts(pos_nm, box_nm, bead_types, molecule, type_names, r_edges, sigma_bins=0.0):
    """Histogram of intermolecular pair distances per type pair for one frame (minimum image);
    sigma_bins > 0: Gaussian kernel (kernel_counts) on the uniform grid r_edges, fed by every pair whose
    Gaussian reaches the grid (r < r_edges[-1] + kernel reach), as GradCG's all-pairs RDF."""
    from scipy.spatial import cKDTree
    x = np.mod(pos_nm, box_nm)
    tree = cKDTree(x, boxsize=box_nm)
    reach = kernel_half_width_bins(sigma_bins) * (r_edges[1] - r_edges[0])
    pairs = tree.query_pairs(r_edges[-1] + reach, output_type='ndarray')
    mol = np.asarray(molecule)
    pairs = pairs[mol[pairs[:, 0]] != mol[pairs[:, 1]]]
    d = x[pairs[:, 0]] - x[pairs[:, 1]]
    d -= box_nm * np.round(d / box_nm)
    r = np.linalg.norm(d, axis=1)
    t = np.asarray([type_names.index(s) for s in bead_types])
    ta, tb = np.minimum(t[pairs[:, 0]], t[pairs[:, 1]]), np.maximum(t[pairs[:, 0]], t[pairs[:, 1]])
    out = {}
    for a, b in itertools.combinations_with_replacement(range(len(type_names)), 2):
        sel = (ta == a) & (tb == b)
        if sigma_bins > 0:
            out[f'{type_names[a]}-{type_names[b]}'] = kernel_counts(
                r[sel], r_edges[0], r_edges[1] - r_edges[0], len(r_edges) - 1, sigma_bins)
        else:
            out[f'{type_names[a]}-{type_names[b]}'] = np.histogram(r[sel], bins=r_edges)[0]   # r >= r_edges[-1] dropped
    return out


def rdf_from_counts(counts, n_pairs, volumes_nm3, r_edges):
    """g(r) from summed histograms; ideal count per frame = n_pairs * shell / V."""
    shell = 4 / 3 * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)
    ideal = n_pairs * shell * np.sum(1.0 / np.asarray(volumes_nm3))
    return counts / ideal


def block_se(x, n_blocks=10):
    """Standard error of the mean from block averages (x: [frames, ...])."""
    x = np.asarray(x, dtype=float)
    n = (len(x) // n_blocks) * n_blocks
    if n < n_blocks:
        return np.full(x.shape[1:], np.nan)
    blocks = x[:n].reshape(n_blocks, n // n_blocks, *x.shape[1:]).mean(axis=1)
    return blocks.std(axis=0, ddof=1) / np.sqrt(n_blocks)


def chain_size(pos_nm, bead_mass, molecule):
    """Per-chain mass-weighted Rg and end-to-end distance (first/last bead), unwrapped beads."""
    mol = np.asarray(molecule)
    rg, ree = [], []
    for m in np.unique(mol):
        idx = np.where(mol == m)[0]
        x, w = pos_nm[idx], bead_mass[idx]
        com = (w[:, None] * x).sum(0) / w.sum()
        rg.append(np.sqrt((w * ((x - com)**2).sum(1)).sum() / w.sum()))
        ree.append(np.linalg.norm(x[-1] - x[0]))
    return np.array(rg), np.array(ree)


def bond_lengths(pos_nm, bonds):
    b = np.asarray(bonds, int).reshape(-1, 2)          # a molecule of one bead has none
    return np.linalg.norm(pos_nm[b[:, 0]] - pos_nm[b[:, 1]], axis=1)


def angle_values(pos_nm, angles):
    a = np.asarray(angles, int).reshape(-1, 3)         # molecules of two beads have none
    if not len(a):
        return np.zeros(0)
    u = pos_nm[a[:, 0]] - pos_nm[a[:, 1]]
    v = pos_nm[a[:, 2]] - pos_nm[a[:, 1]]
    c = (u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
    return np.arccos(np.clip(c, -1, 1))


def compressibility_per_bar(volumes_nm3, temperature_K):
    v = np.asarray(volumes_nm3, dtype=float)
    return float(v.var() / (KB_KJ_PER_MOL_K * temperature_K * v.mean())) / BAR_PER_KJ_MOL_NM3


def read_thermo(log_path):
    """Thermo rows (dict of arrays) of the last 'run' in a LAMMPS log (the NPT production)."""
    blocks, current, header = [], None, None
    for line in Path(log_path).read_text(errors='replace').splitlines():
        p = line.split()
        if p and p[0] == 'Step':
            header, current = p, []
            blocks.append((header, current))
        elif current is not None and p and len(p) == len(header):
            try:
                current.append([float(x) for x in p])
            except ValueError:
                current = None
        elif p and p[0] == 'Loop':
            current = None
    header, rows = blocks[-1]
    arr = np.array(rows)
    return {h: arr[:, i] for i, h in enumerate(header)}
