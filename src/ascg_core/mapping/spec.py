"""Mapping specifications: check, normalise and describe an atom-to-bead mapping before any reference exists.

A species is one molecule, or a polymer repeat unit with its two '*' attachment points (head first). Two equivalent
SMILES forms; heavy-atom indices follow the order of the atoms in the SMILES string, attachment points included:

    annotated-smiles           {"format": "annotated-smiles", "smiles": "[CH3:1][CH2:1][CH2:2]...",
                                "bead_types": {"1": "END", "2": "MID", ...}}
        Every heavy atom carries its bead number as the atom-map number, which needs bracket atoms with explicit
        hydrogens ([CH2:1], [cH:2], [OH:3]); '*' attachment points carry none.
    rdkit-heavy-atom-indices   {"format": "rdkit-heavy-atom-indices", "smiles": "CCCCC",
                                "bead_sites": [{"site_id": "E1", "bead_type": "END", "heavy_atom_indices_1based": [1, 2]}, ...]}

A system is one species or {"species": [...]} (mixtures). normalise_species() turns either form into the second
(what map-aa applies, mapping.formats); check() also reports bead masses (hydrogens follow their heavy atom), the
bead graph, the bonded types the model will have (bonds, angles, combined bending-torsion dihedrals on every path of
three bonds; for polymers over four linked repeat units), errors that block the mapping and warnings for the user.
"""
from __future__ import annotations

import itertools
import re

from .cg_mapping import MappingError

ANNOTATED = 'annotated-smiles'
INDICES = 'rdkit-heavy-atom-indices'
MAX_BEAD_TYPES = 16                       # fastMD
USUAL_HEAVY = (2, 5)                      # heavy atoms per bead the skills recommend


def _mol(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        raise MappingError(f'bad SMILES {smiles!r}')
    return mol


def _radicals(mol):
    """Atoms left with unpaired electrons: in a bead-annotated SMILES a bracket atom without its hydrogens ([C:1])."""
    return [f'{a.GetSymbol()}{a.GetIdx() + 1}' for a in mol.GetAtoms() if a.GetNumRadicalElectrons()]


def normalise_species(sp):
    """One species in rdkit-heavy-atom-indices form (annotated-smiles converted; the SMILES keeps its atom order)."""
    fmt = sp.get('format', INDICES)
    if fmt == INDICES:
        if not sp.get('smiles') or not sp.get('bead_sites'):
            raise MappingError('a rdkit-heavy-atom-indices species needs "smiles" and "bead_sites"')
        return dict(sp)
    if fmt != ANNOTATED:
        raise MappingError(f'mapping format {fmt!r}: use {ANNOTATED} or {INDICES}')
    smiles = sp.get('smiles') or ''
    mol = _mol(smiles)
    beads = {}
    for a in mol.GetAtoms():
        n = a.GetAtomMapNum()
        if a.GetAtomicNum() == 0:
            if n:
                raise MappingError(f'attachment point {a.GetIdx() + 1} carries bead number {n}; it belongs to no bead')
            continue
        if not n:
            raise MappingError(f'heavy atom {a.GetSymbol()}{a.GetIdx() + 1} has no bead number (write it as '
                               f'[{a.GetSymbol()}H<n>:<bead>])')
        beads.setdefault(n, []).append(a.GetIdx() + 1)
    types = {str(k): v for k, v in (sp.get('bead_types') or {}).items()}
    missing = [n for n in sorted(beads) if str(n) not in types]
    if missing:
        raise MappingError(f'bead numbers {missing} have no entry in "bead_types"')
    unused = sorted(set(types) - {str(n) for n in beads})
    if unused:
        raise MappingError(f'"bead_types" lists bead numbers no atom carries: {unused}')
    plain = re.sub(r':\d+\]', ']', smiles)            # the same atoms in the same order, without the bead numbers
    out = {'format': INDICES, 'smiles': plain,
           'bead_sites': [{'site_id': f'B{n}', 'bead_type': types[str(n)], 'heavy_atom_indices_1based': beads[n]}
                          for n in sorted(beads)]}
    for k in ('name', 'description'):
        if sp.get(k):
            out[k] = sp[k]
    return out


def normalise(spec):
    """{"species": [...]} or a single species -> list of species in rdkit-heavy-atom-indices form."""
    species = spec.get('species') if isinstance(spec, dict) and 'species' in spec else [spec]
    if not species or not all(isinstance(s, dict) for s in species):
        raise MappingError('a mapping is one species object or {"species": [...]}')
    return [normalise_species(s) for s in species]


def _species_report(sp):
    """Partition, masses and bead graph of one normalised species; raises MappingError on a broken partition."""
    from rdkit import Chem
    smiles, sites = sp['smiles'], sp['bead_sites']
    mol = _mol(smiles)
    rad = _radicals(mol)
    if rad:
        raise MappingError(f'{smiles}: atoms {rad} have unpaired electrons - bracket atoms need their hydrogens '
                           '(e.g. [CH2:1], not [C:1])')
    n = mol.GetNumAtoms()
    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    owner = {}
    for b, s in enumerate(sites):
        for k in s.get('heavy_atom_indices_1based') or []:
            i = int(k) - 1
            if not 0 <= i < n:
                raise MappingError(f'bead {s.get("site_id", b + 1)}: atom {k} is outside 1..{n}')
            if i in dummies:
                raise MappingError(f'bead {s.get("site_id", b + 1)} lists the attachment point {k}')
            if i in owner:
                raise MappingError(f'atom {k} ({mol.GetAtomWithIdx(i).GetSymbol()}) is in two beads')
            owner[i] = b
    left = [f'{mol.GetAtomWithIdx(i).GetSymbol()}{i + 1}' for i in heavy if i not in owner]
    if left:
        raise MappingError(f'{smiles}: heavy atoms {left} are in no bead (every heavy atom belongs to exactly one bead)')
    if dummies and len(dummies) != 2:
        raise MappingError(f'{smiles}: a polymer repeat unit has exactly two attachment points (*), not {len(dummies)}')
    hmol = Chem.AddHs(mol)
    mass = [0.0] * len(sites)
    for a in hmol.GetAtoms():
        if a.GetAtomicNum() == 0:
            continue
        i = a.GetIdx() if a.GetAtomicNum() > 1 else a.GetNeighbors()[0].GetIdx()
        mass[owner[i]] += a.GetMass()
    edges, inner = set(), {b: set() for b in range(len(sites))}
    for bd in mol.GetBonds():
        i, j = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        if i in owner and j in owner:
            if owner[i] != owner[j]:
                edges.add(tuple(sorted((owner[i], owner[j]))))
            else:
                inner[owner[i]].add((i, j))
    beads = []
    for b, s in enumerate(sites):
        atoms = sorted(i for i, o in owner.items() if o == b)
        comp = {atoms[0]: atoms[0]}                            # connectivity of the bead's own atoms (union-find)
        def root(x):
            while comp.setdefault(x, x) != x:
                x = comp[x]
            return x
        for i, j in inner[b]:
            comp[root(i)] = root(j)
        beads.append({'site_id': s.get('site_id', f'B{b + 1}'), 'bead_type': s['bead_type'],
                      'heavy_atoms': [f'{mol.GetAtomWithIdx(i).GetSymbol()}{i + 1}' for i in atoms],
                      'mass_amu': round(mass[b], 3), 'connected': len({root(i) for i in atoms}) == 1})
    joints = [owner[mol.GetAtomWithIdx(d).GetNeighbors()[0].GetIdx()] for d in dummies]   # head bead, tail bead
    return {'smiles': smiles, 'kind': 'repeat_unit' if dummies else 'molecule', 'beads': beads,
            'mass_amu': round(sum(mass), 3), 'edges': sorted(edges), 'joints': joints}


def _bonded_types(types, edges, joints=None, units=4):
    """Bond, angle and dihedral type names of one molecule, or of `units` linked repeat units (tail bead of unit k to
    head bead of unit k + 1)."""
    if joints:
        n, (head, tail) = len(types), joints
        edges = [(a + k * n, b + k * n) for k in range(units) for a, b in edges] + \
                [(tail + k * n, head + (k + 1) * n) for k in range(units - 1)]
        types = types * units
    nbr = {i: set() for i in range(len(types))}
    for a, b in edges:
        nbr[a].add(b)
        nbr[b].add(a)
    canon = lambda t: '-'.join(min(t, t[::-1]))
    bonds = {canon((types[a], types[b])) for a, b in edges}
    angles = {canon((types[a], types[b], types[c])) for b in nbr for a, c in itertools.combinations(sorted(nbr[b]), 2)}
    dihedrals = {canon((types[a], types[b], types[c], types[d]))            # paths a-b-c-d around each bond b-c
                 for b, c in edges for a in nbr[b] - {c} for d in nbr[c] - {b, a}}
    return sorted(bonds), sorted(angles), sorted(dihedrals)


RESIDUE_NAMES = 'residue-atom-names'


def _check_residue_names(spec):
    """Structure of a residue-atom-names mapping; the assignment itself needs the topology (ascg ref inspect)."""
    errors, types = [], set()
    res = spec.get('residues')
    if not isinstance(res, dict) or not res:
        errors.append('a residue-atom-names mapping needs "residues": {resname: [{"type": T, "atoms": [names]}, ...]}')
        res = {}
    for name, beads in res.items():
        if not isinstance(beads, list) or not beads:
            errors.append(f'residue {name}: a list of beads')
            continue
        seen = set()
        for b in beads:
            if not isinstance(b, dict) or not b.get('type') or not b.get('atoms'):
                errors.append(f'residue {name}: every bead needs "type" and "atoms"')
                continue
            types.add(str(b['type']))
            dup = seen & set(b['atoms'])
            if dup:
                errors.append(f'residue {name}: atoms {sorted(dup)} in two beads')
            seen |= set(b['atoms'])
    if len(types) > MAX_BEAD_TYPES:
        errors.append(f'{len(types)} bead types; fastMD allows {MAX_BEAD_TYPES}')
    return {'ok': not errors, 'errors': errors, 'warnings': [], 'bead_types': sorted(types), 'n_bead_types': len(types),
            'species': [], 'normalised': spec,
            'note': 'atom names are checked against the topology by `ascg ref inspect --mapping` (beads per molecule, '
                    'masses, bonded types); hydrogens not listed follow their heavy atom'}


def check(spec, system_class=None):
    """Report on a mapping specification: {'ok', 'errors', 'warnings', 'normalised', 'species', 'bead_types', ...}.
    Never raises on a bad mapping: problems are listed under 'errors'."""
    if isinstance(spec, dict) and spec.get('format') == RESIDUE_NAMES:
        return _check_residue_names(spec)
    try:
        species = normalise(spec)
    except MappingError as e:
        return {'ok': False, 'errors': [str(e)], 'warnings': []}
    errors, warnings, reports = [], [], []
    for k, sp in enumerate(species):
        try:
            r = _species_report(sp)
        except MappingError as e:
            errors.append(f'species {k + 1}: {e}')
            continue
        types = [b['bead_type'] for b in r['beads']]
        r['bonds'], r['angles'], r['dihedrals'] = _bonded_types(types, r['edges'], r['joints'] or None)
        r['beads_per_unit' if r['kind'] == 'repeat_unit' else 'beads_per_molecule'] = len(types)
        for b in r['beads']:
            nh = len(b['heavy_atoms'])
            if not b['connected']:
                warnings.append(f"bead {b['site_id']} ({b['bead_type']}): its atoms are not bonded to each other; "
                                'its centre of mass can sit where no atom is')
            if not USUAL_HEAVY[0] <= nh <= USUAL_HEAVY[1]:
                warnings.append(f"bead {b['site_id']} ({b['bead_type']}) has {nh} heavy atom(s), outside the usual "
                                f'{USUAL_HEAVY[0]}-{USUAL_HEAVY[1]}')
        if r['kind'] == 'molecule' and len(types) == 1:
            warnings.append(f'species {k + 1} is a single bead: no bonded terms; the model is pair potentials only')
        if system_class == 'polymer' and r['kind'] != 'repeat_unit':
            errors.append(f'species {k + 1}: a polymer mapping is a repeat unit with two attachment points (*)')
        if system_class in ('liquid', 'liquid_mixture') and r['kind'] != 'molecule':
            errors.append(f'species {k + 1}: a liquid mapping is a whole molecule (no attachment points)')
        del r['edges'], r['joints']
        reports.append(r)
    all_types = sorted({b['bead_type'] for r in reports for b in r['beads']})
    if len(all_types) > MAX_BEAD_TYPES:
        errors.append(f'{len(all_types)} bead types; fastMD allows {MAX_BEAD_TYPES}')
    by_type = {}
    for r in reports:
        for b in r['beads']:
            by_type.setdefault(b['bead_type'], []).append(b['mass_amu'])
    for t, ms in by_type.items():
        if max(ms) > 1.15 * min(ms):
            warnings.append(f'bead type {t} has masses {min(ms):.2f}-{max(ms):.2f} amu; fastMD uses one mass per type '
                            '(their mean), which changes dynamics but not structure')
    if system_class == 'liquid' and len(species) > 1:
        errors.append('a liquid has one species; a mixture is system class liquid_mixture')
    return {'ok': not errors, 'errors': errors, 'warnings': warnings, 'system_class': system_class,
            'bead_types': all_types, 'n_bead_types': len(all_types), 'species': reports,
            'normalised': {'species': species} if len(species) > 1 else species[0]}
