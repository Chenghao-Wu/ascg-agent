"""Bring-your-own AA reference: the mapped-reference files (as map-aa writes them) from the user's own AA topology
and trajectories, read with MDAnalysis (GROMACS .tpr / .top, LAMMPS data, PSF, ...; .xtc, .trr, .dcd, LAMMPS dumps,
...), and an approved mapping.

Mapping forms:
    residue-atom-names   {"format": "residue-atom-names",
                          "residues": {"<resname or glob>": [{"type": "A", "atoms": ["C1", "C2"]}, ...], ...}}
        beads by atom names inside each residue; hydrogens that are not listed follow their bonded heavy atom; a residue
        name may be a glob (M*). Works for molecules and polymers (one residue per monomer, end residues listed too).
    annotated-smiles / rdkit-heavy-atom-indices (mapping.spec), for molecules (liquids, mixtures): every molecule is
        matched to a species by its heavy-atom graph (element-labelled isomorphism); hydrogens follow their heavy atom.

Molecules are made whole per frame along bonds (minimum image per bond), as in map-aa; boxes must be orthorhombic.
Energy files (optional): GROMACS .xvg (gmx energy) or .edr (needs pyedr), LAMMPS log (units real or metal).
"""
from __future__ import annotations

import fnmatch
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .cg_mapping import AMU_PER_NM3_TO_G_CM3, ANGSTROM_PER_NM, CGTopologySpec, MappingError, map_frame
from .cg_reference import BAR_PER_ATM, make_whole, read_xvg, whole_plan

RESIDUE_NAMES = 'residue-atom-names'
ELEMENT_MASS = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Si': 28.086, 'P': 30.974,
                'S': 32.06, 'Cl': 35.453, 'Br': 79.904}


# ----------------------------------------------------------------------------------------------- reading
LAMMPS_DUMPS = ('.lammpstrj', '.dump', '.lammpsdump')


def lammps_timestep_ps(log):
    """The integration timestep (ps) of a LAMMPS log (units real: fs; metal: ps), or None."""
    text = Path(log).read_text(errors='replace')
    units = re.findall(r'^\s*units\s+(\w+)', text, re.M)
    steps = re.findall(r'^\s*timestep\s+([0-9.eE+-]+)', text, re.M)
    if not steps:
        return None
    return float(steps[-1]) * (1e-3 if (units[-1] if units else 'real') == 'real' else 1.0)


def is_lammps_dump(trajectory):
    first = trajectory[0] if isinstance(trajectory, (list, tuple)) else trajectory
    return bool(first) and Path(str(first)).suffix in LAMMPS_DUMPS


def universe(topology, trajectory=None, *, topology_format=None, trajectory_format=None, timestep_ps=None):
    """MDAnalysis Universe; for LAMMPS dumps timestep_ps (the integration timestep) turns step numbers into ps."""
    import MDAnalysis as mda
    kw = {}
    if topology_format:
        kw['topology_format'] = topology_format
    elif Path(str(topology)).suffix == '.top':                     # a GROMACS .top: the ITP reader, whole system
        kw.update(topology_format='ITP', infer_system=True)
    if trajectory_format:
        kw['format'] = trajectory_format
    elif is_lammps_dump(trajectory):
        kw['format'] = 'LAMMPSDUMP'
    if kw.get('format') == 'LAMMPSDUMP' and timestep_ps:
        kw['dt'] = float(timestep_ps)
    traj = [str(t) for t in trajectory] if isinstance(trajectory, (list, tuple)) else trajectory
    u = mda.Universe(str(topology), traj, **kw) if traj else mda.Universe(str(topology), **kw)
    if not hasattr(u.atoms, 'bonds') or len(u.atoms.bonds) == 0:
        raise MappingError(f'{topology}: the topology has no bonds (give a .tpr, a .top/.itp, a LAMMPS data file or a '
                           'PSF; a .gro or .pdb without connectivity cannot define molecules)')
    return u


def elements(u):
    """Element of every atom: the topology's elements, else guessed from the atom names, else from the masses."""
    n = len(u.atoms)
    el = None
    try:
        el = [str(e).capitalize() for e in u.atoms.elements]
        if not all(e in ELEMENT_MASS for e in el):
            el = None
    except Exception:
        el = None
    if el is None and hasattr(u.atoms, 'names'):
        guess = []
        for name in u.atoms.names:
            hit = re.match(r'([A-Za-z]{1,2})', str(name))
            sym = hit.group(1).capitalize() if hit else ''
            guess.append(sym if sym in ELEMENT_MASS else (sym[0] if sym[:1] in ELEMENT_MASS else '?'))
        if '?' not in guess:
            el = guess
    if el is None:
        el = []
        for mass in u.atoms.masses:
            best = min(ELEMENT_MASS, key=lambda e: abs(ELEMENT_MASS[e] - mass))
            el.append(best if abs(ELEMENT_MASS[best] - mass) < 0.1 else '?')
    if len(el) != n or '?' in el:
        raise MappingError('cannot tell the elements of every atom (no element / name / mass match); give a topology '
                           'with elements or standard atom names')
    return np.array(el)


def _bond_list(u):
    return np.asarray(u.atoms.bonds.indices, int)


def _has_names(u):
    """Residue and atom names in the topology (a LAMMPS data file has neither)."""
    try:
        u.residues.resnames, u.atoms.names
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------------------------- mapping
def _assign_residues(u, el, mapping):
    """atom -> bead, bead types, bead molecule (fragment) from a residue-atom-names mapping."""
    spec = mapping.get('residues') or {}
    if not spec:
        raise MappingError('a residue-atom-names mapping needs "residues": {resname: [{"type", "atoms"}, ...]}')
    if not _has_names(u):
        raise MappingError('this topology has no residue or atom names (e.g. a LAMMPS data file): use a SMILES mapping '
                           '(annotated-smiles) for molecules, or a topology with names')
    bonds = _bond_list(u)
    nbr = [[] for _ in range(len(u.atoms))]
    for a, b in bonds:
        nbr[a].append(b)
        nbr[b].append(a)
    frag_of = np.empty(len(u.atoms), int)
    for k, f in enumerate(u.atoms.fragments):
        frag_of[f.indices] = k
    atom_bead = np.full(len(u.atoms), -1)
    bead_types, bead_mol = [], []
    unknown = Counter()
    for res in u.residues:
        key = res.resname if res.resname in spec else next((k for k in spec if fnmatch.fnmatchcase(res.resname, k)), None)
        if key is None:
            unknown[res.resname] += 1
            continue
        names = {}
        for a in res.atoms:
            if a.name in names:
                raise MappingError(f'residue {res.resname} {res.resid}: atom name {a.name} occurs twice')
            names[a.name] = a.index
        for bead in spec[key]:
            idx = []
            for nm in bead.get('atoms') or []:
                if nm not in names:
                    raise MappingError(f'residue {res.resname} {res.resid}: no atom named {nm!r} '
                                       f'(atoms: {", ".join(names)})')
                i = names[nm]
                if atom_bead[i] >= 0:
                    raise MappingError(f'residue {res.resname} {res.resid}: atom {nm} is in two beads')
                idx.append(i)
            if not idx:
                raise MappingError(f'residue {key}: a bead lists no atoms')
            atom_bead[idx] = len(bead_types)
            bead_types.append(str(bead['type']))
            bead_mol.append(int(frag_of[idx[0]]))
        for nm, i in names.items():                     # hydrogens not listed follow their bonded heavy atom
            if atom_bead[i] < 0 and el[i] == 'H':
                heavy = [j for j in nbr[i] if el[j] != 'H']
                if len(heavy) == 1 and atom_bead[heavy[0]] >= 0:
                    atom_bead[i] = atom_bead[heavy[0]]
        left = [nm for nm, i in names.items() if atom_bead[i] < 0]
        if left:
            raise MappingError(f'residue {res.resname} {res.resid}: atoms {left} are in no bead')
    if unknown:
        raise MappingError('residues without a mapping entry: ' + ', '.join(f'{r} (x{n})' for r, n in unknown.items()))
    return atom_bead, bead_types, np.array(bead_mol)


def _species_graph(sp):
    import networkx as nx
    from rdkit import Chem
    mol = Chem.MolFromSmiles(sp['smiles'])
    if mol is None:
        raise MappingError(f'bad SMILES {sp["smiles"]}')
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise MappingError('a SMILES mapping with attachment points (*) is a polymer repeat unit: give polymers a '
                           f'{RESIDUE_NAMES} mapping (one residue per monomer)')
    bead = {}
    for b, s in enumerate(sp['bead_sites']):
        for k in s['heavy_atom_indices_1based']:
            bead[int(k) - 1] = b
    g = nx.Graph()
    for a in mol.GetAtoms():
        if a.GetAtomicNum() > 1:
            g.add_node(a.GetIdx(), element=a.GetSymbol(), bead=bead[a.GetIdx()])
    for bd in mol.GetBonds():
        i, j = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        if i in g and j in g:
            g.add_edge(i, j)
    return g, [s['bead_type'] for s in sp['bead_sites']]


def _assign_smiles(u, el, mapping):
    """atom -> bead from a SMILES mapping of molecules: each fragment matched to a species by its heavy-atom graph."""
    import networkx as nx
    from networkx.algorithms import isomorphism
    from .spec import normalise
    species = [_species_graph(sp) for sp in normalise(mapping)]
    bonds = _bond_list(u)
    atom_bead = np.full(len(u.atoms), -1)
    bead_types, bead_mol = [], []
    same = lambda a, b: a['element'] == b['element']
    for k, frag in enumerate(u.atoms.fragments):
        idx = frag.indices
        heavy = [i for i in idx if el[i] != 'H']
        g = nx.Graph()
        for i in heavy:
            g.add_node(int(i), element=el[i])
        hs = set(heavy)
        for a, b in bonds:
            if a in hs and b in hs:
                g.add_edge(int(a), int(b))
        sig = (tuple(sorted(Counter(el[heavy]).items())), g.number_of_edges())
        found = None
        for s, (sg, types) in enumerate(species):
            if sg.number_of_nodes() != g.number_of_nodes() or sg.number_of_edges() != g.number_of_edges():
                continue
            gm = isomorphism.GraphMatcher(g, sg, node_match=same)
            m = next(gm.isomorphisms_iter(), None)
            if m is not None:
                found = (s, m)
                break
        if found is None:
            raise MappingError(f'molecule {k} ({"".join(f"{e}{n}" for e, n in sig[0])}, {len(idx)} atoms) matches no '
                               'species of the mapping')
        s, m = found
        sg, types = species[s]
        first = len(bead_types)
        for b, t in enumerate(types):
            bead_types.append(t)
            bead_mol.append(k)
        for i, j in m.items():
            atom_bead[i] = first + sg.nodes[j]['bead']
    for a, b in bonds:                                       # hydrogens follow their heavy atom
        for h, x in ((a, b), (b, a)):
            if el[h] == 'H' and el[x] != 'H':
                atom_bead[h] = atom_bead[x]
    if (atom_bead < 0).any():
        raise MappingError(f'{int((atom_bead < 0).sum())} atoms are in no bead (hydrogens without a heavy atom?)')
    return atom_bead, bead_types, np.array(bead_mol)


def cg_topology(task, u, mapping, el=None):
    """CGTopologySpec of the universe under the mapping (bead order: molecules, then beads within each)."""
    el = elements(u) if el is None else el
    if mapping.get('format') == RESIDUE_NAMES:
        atom_bead, bead_types, bead_mol = _assign_residues(u, el, mapping)
    else:
        atom_bead, bead_types, bead_mol = _assign_smiles(u, el, mapping)
    order = np.lexsort((np.arange(len(bead_types)), bead_mol))       # stable: molecule-major
    remap = np.empty(len(order), int)
    remap[order] = np.arange(len(order))
    atom_bead = remap[atom_bead]
    bead_types = [bead_types[i] for i in order]
    bead_mol = bead_mol[order]
    masses = np.bincount(atom_bead, weights=u.atoms.masses, minlength=len(bead_types))
    pairs = {tuple(sorted((int(atom_bead[a]), int(atom_bead[b])))) for a, b in _bond_list(u)
             if atom_bead[a] != atom_bead[b]}
    bonds = sorted(pairs)
    neigh = {}
    for a, b in bonds:
        neigh.setdefault(a, set()).add(b)
        neigh.setdefault(b, set()).add(a)
    import itertools
    angles = sorted({(min(i, k), j, max(i, k)) for j, ns in neigh.items()
                     for i, k in itertools.combinations(sorted(ns), 2)})
    return CGTopologySpec(task, bead_types, masses, bead_mol, atom_bead, bonds, angles, sorted(set(bead_types)))


# ----------------------------------------------------------------------------------------------- energy files
def read_energy(path):
    """{'time_ps', 'Temp' K, 'Press' atm, optional 'Volume_nm3'} from a gmx energy .xvg, a .edr (pyedr) or a LAMMPS
    log (units real: atm, metal: bar; the time from the logged timestep)."""
    p = Path(path)
    if p.suffix == '.xvg':
        d = read_xvg(p)
        return {'time_ps': d['Step'], 'Temp': d['Temp'], 'Press': d['Press'],
                **({'Volume_nm3': d['Volume_nm3']} if 'Volume_nm3' in d else {})}
    if p.suffix == '.edr':
        try:
            import pyedr
        except ImportError:
            raise MappingError(f'{p}: reading .edr needs pyedr (pip install pyedr), or give `gmx energy` output (.xvg)')
        d = pyedr.edr_to_dict(str(p))
        return {'time_ps': np.asarray(d['Time']), 'Temp': np.asarray(d['Temperature']),
                'Press': np.asarray(d['Pressure']) / BAR_PER_ATM,
                **({'Volume_nm3': np.asarray(d['Volume'])} if 'Volume' in d else {})}
    from .cg_mapping import read_thermo
    text = p.read_text(errors='replace')
    units = re.findall(r'^\s*units\s+(\w+)', text, re.M)
    units = units[-1] if units else 'real'
    if units not in ('real', 'metal'):
        raise MappingError(f'{p}: LAMMPS units {units!r}; only real (atm) and metal (bar) logs are read')
    steps = re.findall(r'^\s*timestep\s+([0-9.eE+-]+)', text, re.M)
    dt_ps = float(steps[-1]) * (1e-3 if units == 'real' else 1.0) if steps else None
    d = read_thermo(p)
    press = d['Press'] if units == 'real' else d['Press'] / BAR_PER_ATM
    out = {'time_ps': d['Step'] * dt_ps if dt_ps else None, 'Temp': d['Temp'], 'Press': press}
    if 'Volume' in d:
        out['Volume_nm3'] = d['Volume'] / ANGSTROM_PER_NM**3
    return out


def _window(th, start_ps):
    if th is None or start_ps <= 0 or th.get('time_ps') is None:
        return th
    keep = th['time_ps'] >= start_ps - 1e-9
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in th.items()}


# ----------------------------------------------------------------------------------------------- frames
def mapped_frames(u, topo, *, start_ps=0.0, stride=1):
    """Bead positions (nm, molecules whole), boxes (nm) and times (ps) of the trajectory."""
    plan = whole_plan(_bond_list(u), len(u.atoms))
    masses = np.asarray(u.atoms.masses, float)
    pos, box, times = [], [], []
    for ts in u.trajectory[::max(1, int(stride))]:
        if ts.time < start_ps - 1e-9:
            continue
        dims = ts.dimensions
        if dims is None or not np.allclose(dims[3:], 90.0):
            raise MappingError('only orthorhombic boxes are supported')
        b = np.asarray(dims[:3], float)
        xyz = make_whole(ts.positions, b, plan)
        pos.append(map_frame(xyz, masses, topo.atom_bead, topo.bead_masses_amu))
        box.append(b / ANGSTROM_PER_NM)
        times.append(float(ts.time))
    if not pos:
        raise MappingError(f'no frames at or after {start_ps} ps')
    return np.array(pos), np.array(box), np.array(times)


# ----------------------------------------------------------------------------------------------- entry points
def inspect(topology, trajectory=None, energy=None, mapping=None, *, topology_format=None, trajectory_format=None,
            timestep_fs=None, max_frames=200):
    """What the files hold, and (with a mapping) the bead assignment it gives."""
    dt_ps = timestep_fs * 1e-3 if timestep_fs else (lammps_timestep_ps(energy) if energy and
                                                     Path(str(energy)).suffix not in ('.xvg', '.edr') else None)
    u = universe(topology, trajectory, topology_format=topology_format, trajectory_format=trajectory_format,
                 timestep_ps=dt_ps)
    el = elements(u)
    frags = u.atoms.fragments
    kinds = Counter()
    for f in frags:
        kinds[''.join(f'{e}{n}' for e, n in sorted(Counter(el[f.indices]).items()))] += 1
    res = {}
    if _has_names(u):
        for r in u.residues:
            d = res.setdefault(r.resname, {'count': 0, 'atoms': [str(n) for n in r.atoms.names]})
            d['count'] += 1
    out = {'topology': str(topology), 'n_atoms': len(u.atoms), 'n_bonds': len(u.atoms.bonds),
           'molecules': {'count': len(frags), 'by_formula': dict(kinds.most_common())},
           'residues': {k: {'count': v['count'], 'atoms': v['atoms'][:40]} for k, v in list(res.items())[:30]},
           'n_residue_names': len(res), 'hydrogens': int((el == 'H').sum()), 'mass_amu': float(u.atoms.masses.sum()),
           **({} if res else {'note': 'no residue or atom names: map molecules with a SMILES mapping'})}
    if trajectory:
        tr = u.trajectory
        n = len(tr)
        sel = np.unique(np.linspace(0, n - 1, min(n, max_frames)).astype(int))
        vols, times = [], []
        for i in sel:
            ts = tr[int(i)]
            vols.append(float(np.prod(ts.dimensions[:3])) / ANGSTROM_PER_NM**3)
            times.append(float(ts.time))
        vols = np.array(vols)
        rho = out['mass_amu'] * AMU_PER_NM3_TO_G_CM3 / vols
        out['trajectory'] = {'frames': n, 'time_ps': [times[0], times[-1]],
                             'frame_spacing_ps': (times[-1] - times[0]) / (n - 1) if n > 1 else None,
                             'volume_nm3': [float(vols.min()), float(vols.max())],
                             'ensemble_guess': 'NPT' if vols.std() / vols.mean() > 1e-6 else 'NVT',
                             'density_g_cm3': [float(rho.min()), float(rho.max())]}
        if not 0.05 < rho.mean() < 3.5:
            out.setdefault('warnings', []).append(f'density {rho.mean():.3g} g/cm3 is implausible: check units')
        if is_lammps_dump(trajectory) and not dt_ps:
            out['trajectory']['time_ps'] = None
            out['trajectory']['frame_spacing_ps'] = None
            out['trajectory']['steps'] = [times[0], times[-1]]
            out.setdefault('warnings', []).append('LAMMPS dump times are step numbers: give the LAMMPS log or the '
                                                  'timestep (fs) to convert them to ps')
    if energy:
        th = read_energy(energy)
        out['energy'] = {'file': str(energy), 'rows': len(th['Temp']),
                         'temperature_K_mean': float(np.mean(th['Temp'])),
                         'pressure_bar_mean': float(np.mean(th['Press']) * BAR_PER_ATM),
                         'time_ps': [float(th['time_ps'][0]), float(th['time_ps'][-1])] if th.get('time_ps') is not None else None}
    if mapping:
        from .spec import _bonded_types
        topo = cg_topology('inspect', u, mapping, el)
        types = topo.bead_types
        per_mol = {}
        for k, f in enumerate(frags):
            beads = np.flatnonzero(topo.bead_molecule == k)
            key = ' '.join(types[b] for b in beads)
            d = per_mol.setdefault(key, {'count': 0, 'beads': len(beads),
                                         'bead_masses_amu': [round(float(topo.bead_masses_amu[b]), 3) for b in beads]})
            d['count'] += 1
        local = {}
        for a, b in topo.bonds:
            local.setdefault(int(topo.bead_molecule[a]), []).append((a, b))
        dih = set()
        for k, edges in local.items():
            beads = sorted({x for e in edges for x in e})
            idx = {b: i for i, b in enumerate(beads)}
            _, _, d = _bonded_types([types[b] for b in beads], [(idx[a], idx[b]) for a, b in edges])
            dih |= set(d)
        bonds_all = _bond_list(u)                      # beads whose heavy atoms are not bonded to each other
        heavy = el != 'H'
        split = Counter()
        for b in range(len(types)):
            members = set(np.flatnonzero((topo.atom_bead == b) & heavy).tolist())
            if len(members) > 1:
                seen, todo = set(), [next(iter(members))]
                inner = [(x, y) for x, y in bonds_all if x in members and y in members]
                while todo:
                    x = todo.pop()
                    if x in seen:
                        continue
                    seen.add(x)
                    todo += [y for a, c in inner for y in ((c,) if a == x else (a,) if c == x else ())]
                if seen != members:
                    split[types[b]] += 1
        if split:
            out.setdefault('warnings', []).extend(
                f'{n} bead(s) of type {t} hold heavy atoms that are not bonded to each other: check that the atom names '
                'follow the chain (their centre of mass can sit between groups)' for t, n in split.items())
        out['mapping'] = {'beads': len(types), 'bead_types': topo.type_names, 'molecule_types': per_mol,
                          'bond_types': sorted({topo.bond_type(b) for b in topo.bonds}),
                          'angle_types': sorted({topo.angle_type(a) for a in topo.angles}),
                          'dihedral_types': sorted(dih)}
    return out


def prepare_byo_reference(task, topology, runs, mapping, out_dir, *, topology_format=None, trajectory_format=None,
                          start_ps=0.0, stride=1, r_max_nm=1.5, dr_nm=0.01, bond_bin_nm=0.002, angle_bin_deg=2.0,
                          n_blocks=10):
    """runs: {state label: {'trajectory': path or [paths], 'energy': path or None}}; one mapping for all states."""
    from .cg_reference import write_reference
    topo, mapped, thermo, n_atoms = None, {}, {}, None
    for T, run in runs.items():
        dt_ps = None
        if is_lammps_dump(run['trajectory']):
            dt_ps = (run['timestep_fs'] * 1e-3 if run.get('timestep_fs') else
                     lammps_timestep_ps(run['energy']) if run.get('energy') else None)
            if dt_ps is None and start_ps > 0:
                raise MappingError(f'state {T}: LAMMPS dump times are step numbers; give the LAMMPS log or the '
                                   'timestep (fs) to discard the first ps')
        u = universe(topology, run['trajectory'], topology_format=topology_format, trajectory_format=trajectory_format,
                     timestep_ps=dt_ps)
        if topo is None:
            topo, n_atoms = cg_topology(task, u, mapping), len(u.atoms)
        elif len(u.atoms) != n_atoms:
            raise MappingError(f'state {T}: {len(u.atoms)} atoms, the first state has {n_atoms}')
        mapped[T] = mapped_frames(u, topo, start_ps=start_ps, stride=stride)
        thermo[T] = _window(read_energy(run['energy']), start_ps) if run.get('energy') else None
    source = {'kind': 'bring-your-own', 'topology': str(topology), 'mapping_format': mapping.get('format'),
              'runs': {str(T): {k: (str(v) if not isinstance(v, (list, tuple)) else [str(x) for x in v])
                                for k, v in r.items() if v} for T, r in runs.items()},
              'start_ps': start_ps, 'stride': stride}
    return write_reference(task, topo, mapped, thermo, out_dir, r_max_nm=r_max_nm, dr_nm=dr_nm,
                           bond_bin_nm=bond_bin_nm, angle_bin_deg=angle_bin_deg, n_blocks=n_blocks, source=source)
