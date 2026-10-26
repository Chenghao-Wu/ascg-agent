"""Mapping definitions given in other formats, converted to the repeat-unit groups build_cg_topology reads.

annotated-smiles (atom-map numbers give each heavy atom's bead) is converted to rdkit-heavy-atom-indices first
(mapping.spec). rdkit-heavy-atom-indices (the system-library format of proposed mappings): a SMILES plus bead sites, each a bead type
and the 1-based RDKit indices of its heavy atoms in that SMILES; hydrogens follow their heavy atom. For molecular
liquids and mixtures every molecule species becomes one 'single' unit whose reference SMILES is the mapping's own
SMILES (not the build recipe's spelling): build_cg_topology matches each built molecule to it by graph isomorphism, so
the indices refer to the mapping's SMILES.

For polymers the mapping's SMILES is the repeat unit with its two [*] attachment points (RDKit indices count them;
they belong to no bead). polymer_inputs matches it onto the build recipe's first / middle / last unit SMILES, keeping
the chain direction (the atoms next to the head and tail [*]), and returns per-position groups in the recipe's heavy-atom
order (the repeat-unit-groups form). Atoms of an end unit that the repeat unit lacks (end groups) join the nearest
bead; repeat-unit atoms that an end unit lacks are left out (e.g. a chain-end methyl in place of the backbone oxygen).
"""
from __future__ import annotations

from .cg_mapping import MappingError
from .spec import ANNOTATED, normalise_species

RDKIT_SITES = 'rdkit-heavy-atom-indices'


def molecular_inputs(task: str, species: list[dict]) -> tuple[dict, dict]:
    """(recipes, mappings) for build_cg_topology from the rdkit-heavy-atom-indices mappings of the molecule species
    of a liquid (one species) or mixture (several)."""
    from rdkit import Chem
    units_r, units_m = {}, {}
    for i, sp in enumerate(species):
        sp = normalise_species(sp) if sp.get('format') == ANNOTATED else sp
        if sp.get('format', RDKIT_SITES) != RDKIT_SITES:
            raise MappingError(f'{task}: mapping format {sp.get("format")!r} is not {RDKIT_SITES}')
        smiles, sites = sp.get('smiles'), sp.get('bead_sites')
        if not smiles or not sites:
            raise MappingError(f'{task}: species {i} has no SMILES or bead sites')
        if '*' in smiles:
            raise MappingError(f'{task}: {smiles} has attachment points; only molecular systems are converted')
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise MappingError(f'{task}: bad SMILES {smiles}')
        n_heavy = sum(a.GetAtomicNum() > 1 for a in mol.GetAtoms())
        groups = [[s['bead_type'], [int(k) - 1 for k in s['heavy_atom_indices_1based']]] for s in sites]
        members = sorted(k for _, idx in groups for k in idx)
        if members != list(range(n_heavy)):
            raise MappingError(f'{task}: bead sites of {smiles} do not partition its {n_heavy} heavy atoms exactly once')
        code = f'MOL{i}'
        units_r[code] = {'single': smiles}
        units_m[code] = groups
    return {'units': units_r}, {'tasks': {task: {'units': units_m}}}


def library_inputs(task: str, library: dict, recipes: dict) -> tuple[dict, dict]:
    """(recipes, mappings) for build_cg_topology from a system library (schema ascg-library/1) entry: repeat-unit
    groups as they are, with the given recipes; rdkit-heavy-atom-indices via polymer_inputs for polymers (with the given
    recipes) and molecular_inputs for liquids and mixtures (components are library ids)."""
    systems = {s['key']: s for s in library['systems']}
    if task not in systems or not systems[task].get('mapping'):
        raise MappingError(f'{task}: no mapping in the system library')
    m = systems[task]['mapping']
    if m.get('format') == 'ascg-repeat-unit-groups':
        return recipes, {'tasks': {task: {'units': m['units']}}}
    if m.get('format') not in (RDKIT_SITES, ANNOTATED):
        raise MappingError(f'{task}: unsupported mapping format {m.get("format")!r}')
    if systems[task].get('system_class') == 'polymer':
        return polymer_inputs(task, m, recipes)
    by_id = {s.get('id'): s for s in library['systems']}
    species = [by_id[c]['mapping'] for c in m['components']] if m.get('components') else [m]
    return molecular_inputs(task, species)


def _heavy_graph(smiles):
    """(heavy-atom graph with element labels, heavy index of the atom bonded to the first [*] or None, ... last [*]),
    heavy indices in RDKit order with dummies removed; also the RDKit index -> heavy index map."""
    import networkx as nx
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise MappingError(f'bad SMILES {smiles}')
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 0]
    k = {old: i for i, old in enumerate(heavy)}
    g = nx.Graph()
    for old in heavy:
        g.add_node(k[old], element=mol.GetAtomWithIdx(old).GetSymbol())
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in k and j in k:
            g.add_edge(k[i], k[j])
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    joints = [k[a.GetNeighbors()[0].GetIdx()] for a in dummies]
    head = joints[0] if dummies and dummies[0].GetIdx() == min(d.GetIdx() for d in dummies) else None
    return g, joints, k


def _core_match(core, core_joints, unit, unit_joints, position):
    """Map core (repeat unit) heavy atoms onto a unit's heavy atoms with the attachment atoms pinned in chain direction:
    middle pins head and tail, first the tail, last the head. Core atoms the unit lacks (up to two terminal ones) are
    dropped. Returns {core atom: unit atom}."""
    import itertools
    from networkx.algorithms import isomorphism
    head, tail = core_joints[0], core_joints[-1]
    pins = {'middle': {head: unit_joints[0], tail: unit_joints[-1]}, 'first': {tail: unit_joints[-1]},
            'last': {head: unit_joints[0]}}[position]
    same = lambda a, b: a['element'] == b['element']
    removable = [n for n in core.nodes if n not in pins]
    for r in range(0, 3):
        for drop in itertools.combinations(removable, r):
            sub = core.subgraph([n for n in core.nodes if n not in drop])
            gm = isomorphism.GraphMatcher(unit, sub, node_match=same)
            for m in gm.subgraph_monomorphisms_iter():          # unit atom -> core atom
                inv = {c: u for u, c in m.items()}
                if all(inv.get(c) == u for c, u in pins.items()):
                    return inv
    raise MappingError(f'the repeat unit does not match the {position} unit')


def _unit_groups(task, mapping):
    """(core heavy-atom graph, its attachment atoms, [(bead type, [core atoms])], core atom -> bead) of one repeat unit's
    SMILES mapping."""
    from rdkit import Chem
    mapping = normalise_species(mapping) if mapping.get('format') == ANNOTATED else mapping
    smiles, sites = mapping.get('smiles'), mapping.get('bead_sites')
    if not smiles or '*' not in smiles or not sites:
        raise MappingError(f'{task}: no repeat-unit SMILES with attachment points or no bead sites')
    core, core_joints, rdk = _heavy_graph(smiles)
    if len(core_joints) != 2:
        raise MappingError(f'{task}: the repeat unit needs exactly two attachment points')
    mol = Chem.MolFromSmiles(smiles)
    groups = []
    for st in sites:
        idx = []
        for one in st['heavy_atom_indices_1based']:
            a = int(one) - 1
            if mol.GetAtomWithIdx(a).GetAtomicNum() == 0:
                raise MappingError(f'{task}: bead {st.get("site_id")} lists the attachment point {one}')
            idx.append(rdk[a])
        groups.append((st['bead_type'], idx))
    members = sorted(i for _, idx in groups for i in idx)
    if members != list(range(core.number_of_nodes())):
        raise MappingError(f'{task}: bead sites do not partition the repeat unit\'s heavy atoms exactly once')
    return core, core_joints, groups, {i: b for b, (_, idx) in enumerate(groups) for i in idx}


def polymer_inputs(task, mapping, recipes):
    """(recipes, mappings) for build_cg_topology from the SMILES mapping of a polymer's repeat unit (homopolymer), or of
    each unit code ({"units": {code: mapping}}, copolymers): groups per recipe unit position (first / middle / last)
    in the recipe's heavy-atom order. Sequences and random compositions both give their unit codes."""
    import networkx as nx
    t = recipes['tasks'][task]
    codes = sorted(t['random']) if 'random' in t else sorted({tok.split('*')[0] for tok in str(t['sequence']).split()})
    per_code = mapping.get('units')
    if per_code is None:
        one = _unit_groups(task, mapping)
    elif set(per_code) != set(codes):
        raise MappingError(f'{task}: the mapping has units {sorted(per_code)}, the recipe {codes}')
    units = {}
    for code in codes:
        core, core_joints, groups, bead_of = _unit_groups(f'{task} {code}', per_code[code]) if per_code else one
        units[code] = {}
        for position in ('first', 'middle', 'last'):
            unit, joints, _ = _heavy_graph(recipes['units'][code][position])
            match = _core_match(core, core_joints, unit, joints, position)     # core atom -> unit atom
            unit_bead = {u: bead_of[c] for c, u in match.items()}
            lengths = {}
            for u in sorted(set(unit.nodes) - set(unit_bead)):                 # end groups: the nearest bead
                d = nx.single_source_shortest_path_length(unit, u)
                near = min((d[v], unit_bead[v]) for v in unit_bead if v in d)
                lengths[u] = near[1]
            unit_bead.update(lengths)
            out = [[groups[b][0], sorted(u for u, bb in unit_bead.items() if bb == b)] for b in range(len(groups))]
            if any(not idx for _, idx in out):
                raise MappingError(f'{task}: a bead is empty in the {code} {position} unit')
            units[code][position] = out
    return recipes, {'tasks': {task: {'units': units}}}
