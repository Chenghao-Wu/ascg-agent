"""A system from one SMILES per species: the all-atom build (an AutoPoly v2.0 recipe) and, when the SMILES carries
bead numbers, the CG mapping, both derived from the same structure.

    {"system_class": "polymer",                       homopolymer: one species, a repeat unit with two '*', head first
     "species": [{"name": "MMA", "smiles": "*[CH2:1][C:1](*)([CH3:1])[C:2](=[O:2])[O:2][CH3:2]",
                  "bead_types": {"1": "BB", "2": "EST"}}],
     "repeat_units": 10, "tacticity": "atactic", "end_groups": "H",       (defaults)
     "chains": null, "target_cg_beads": 1000}                            (chains from the bead count, >= 20)

    {"system_class": "liquid_mixture",                 liquids and mixtures: whole molecules, no '*'
     "species": [{"name": "hexane", "smiles": "CCCCCC", "fraction": 0.5},
                 {"name": "toluene", "smiles": "Cc1ccccc1", "fraction": 0.5}],
     "molecules": 1000}                                                   (or a "count" per species)

The SMILES is the annotated-smiles mapping form (atom-map number = bead) or plain; with bead numbers the mapping
comes with it. Polymer chain ends: the start unit is the repeat unit with its head '*' capped, the end unit with its
tail '*' capped: end_groups "H" or "CH3", or {"head": ..., "tail": ...}. H caps suit addition (vinyl) polymers; for
condensation polymers give the end units as SMILES with one '*' ("first", "last"), e.g. methyl esters for a
polyester, since an H cap on an acyl or silyl end makes an aldehyde or a silane. Copolymers: one species per
monomer and "sequence" ("S*3 BD*4 S*3", "E VA E VA ...") or "random" ({"E": 0.5, "P": 0.5} with "repeat_units").

derive(spec) never raises on a bad spec: problems are listed under 'errors'. It returns the recipe in the format of
benchmarks/*/aa_recipes*.json (build-opls, neutralize-aa, prepare-gmx-protocol, map-aa read it unchanged) and a
one-system library (ascg-library/1) carrying the mapping (cg-topology, map-aa).
"""
from __future__ import annotations

import re

CLASSES = ('polymer', 'liquid', 'liquid_mixture')
DEFAULTS = {'repeat_units': 10, 'tacticity': 'atactic', 'end_groups': 'H', 'target_cg_beads': 1000, 'min_chains': 20,
            'molecules': 1000}
TACTICITY = ('atactic', 'isotactic', 'syndiotactic')
CAPS = {'H': None, 'CH3': 6}
EQUILIBRATION = {'polymer': 'eq21', 'liquid': 'direct', 'liquid_mixture': 'direct'}
OPLS_ELEMENTS = {'C', 'H', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I'}
MAX_ATOMS = 300_000                           # AA cost: beyond this, ask before simulating


def _plain(smiles):
    """The SMILES without bead numbers, attachment points written [*] (AutoPoly's form); atom order kept."""
    s = re.sub(r':\d+\]', ']', smiles)
    return re.sub(r'(?<!\[)\*(?!\])', '[*]', s)


def _build_smiles(smiles):
    """The SMILES AutoPoly gets: no bead numbers, atom order kept (head '*' first), bracket atoms written plainly where
    their hydrogens are the normal valence ([CH2] -> C), attachment points as [*]."""
    from rdkit import Chem
    s = _plain(smiles)
    m = _mol(s)
    hs = [a.GetTotalNumHs() for a in m.GetAtoms()]
    for a in m.GetAtoms():
        if a.GetAtomicNum() and not a.GetFormalCharge():
            a.SetNoImplicit(False)
            a.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return s
    if [a.GetTotalNumHs() for a in m.GetAtoms()] != hs:
        return s
    return _plain(Chem.MolToSmiles(m, canonical=False))


def _mol(smiles):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        raise ValueError(f'bad SMILES {smiles!r}')
    return m


def cap_unit(unit, keep, cap='H', notes=None):
    """Start unit (keep='tail': the head '*' capped) or end unit (keep='head') of a repeat unit with two '*', head
    first. The cap is H or a methyl; bracket atoms keep their valence (one more explicit H). With a list in `notes`,
    an H cap on a carbonyl carbon (an aldehyde end) or on silicon (a silane end) is noted there."""
    from rdkit import Chem
    mol = _mol(unit)
    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        raise ValueError('a repeat unit needs exactly two attachment points (*)')
    drop, kept = (dummies[0], dummies[1]) if keep == 'tail' else (dummies[1], dummies[0])
    rw = Chem.RWMol(mol)
    d = rw.GetAtomWithIdx(drop)
    d.SetIsotope(0)
    head = d.GetNeighbors()[0].GetIdx()                 # the atom that gets the cap
    if CAPS[cap] is None:
        nb = d.GetNeighbors()[0]
        if notes is not None:
            carbonyl = nb.GetSymbol() == 'C' and any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(nb).GetSymbol() == 'O'
                                                     for b in nb.GetBonds())
            if carbonyl or nb.GetSymbol() == 'Si':
                notes.append(f"{'head' if keep == 'tail' else 'tail'}: an H cap makes "
                             f"{'an aldehyde' if carbonyl else 'a silane (Si-H)'} end; give the end unit "
                             f"(\"{'first' if keep == 'tail' else 'last'}\", e.g. a methyl ester or a trimethylsilyl end)")
        if nb.GetNoImplicit():
            nb.SetNumExplicitHs(nb.GetNumExplicitHs() + 1)
        rw.RemoveAtom(drop)
        shift = lambda i: i - (i > drop)                 # noqa: E731  (indices after the removal)
        cap_side, kept = shift(head), shift(kept)
    else:
        d.SetAtomicNum(CAPS[cap])
        d.SetNoImplicit(False)
        cap_side = drop
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    # AutoPoly orients an end unit along its first atom -> its connection atom: a start unit is written from the
    # capped end, an end unit from its attachment point (as the benchmark's 'CC[*]' / '[*]CC')
    return _plain(Chem.MolToSmiles(out, rootedAtAtom=cap_side if keep == 'tail' else kept))


def end_unit_axis_ok(smiles, which):
    """AutoPoly orients a start unit along atom 0 -> its connection atom, and an end unit along its connection atom
    -> its last atom (after removing the '*'); the two must differ."""
    m = _mol(smiles)
    d = [a for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    if len(d) != 1:
        return False
    nb = d[0].GetNeighbors()[0].GetIdx()
    conn = nb - (nb > d[0].GetIdx())
    return conn != (0 if which == 'first' else m.GetNumAtoms() - 2)


def _with_h(smiles):
    """Atoms (hydrogens included, attachment points not), heavy atoms, mass, formula, elements, net charge and
    unpaired electrons of a molecule or of a unit inside a chain (its '*' bonds go to the next unit, not to H)."""
    from collections import Counter
    from rdkit import Chem
    m = _mol(smiles)
    atoms = [a for a in Chem.AddHs(m).GetAtoms() if a.GetAtomicNum() != 0]
    pt = Chem.GetPeriodicTable()
    n = Counter(a.GetSymbol() for a in atoms)
    order = ['C', 'H'] + sorted(x for x in n if x not in ('C', 'H')) if 'C' in n else sorted(n)
    return {'atoms': len(atoms), 'heavy': sum(1 for a in atoms if a.GetAtomicNum() > 1),
            'mass': sum(pt.GetAtomicWeight(a.GetAtomicNum()) for a in atoms),
            'formula': ''.join(f'{x}{n[x] if n[x] > 1 else ""}' for x in order if x in n), 'elements': set(n),
            'charge': Chem.GetFormalCharge(m), 'radicals': sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())}


def _code(name, k):
    c = re.sub(r'[^A-Za-z0-9]', '', str(name or '')).upper()[:8]
    return c or f'U{k + 1}'


def _expand(sequence, codes):
    """'S*3 BD*4 S*3' (names or codes) -> ['S', 'S', 'S', 'BD', ...] in unit codes."""
    out = []
    for tok in str(sequence).split():
        name, _, count = tok.partition('*')
        if name not in codes:
            raise ValueError(f'sequence: {name!r} is not a species (have {", ".join(sorted(set(codes.values())))})')
        out += [codes[name]] * int(count or 1)
    return out


def _compact(tokens):
    runs = []
    for c in tokens:
        if runs and runs[-1][0] == c:
            runs[-1][1] += 1
        else:
            runs.append([c, 1])
    return ' '.join(f'{c}*{n}' if n > 1 else c for c, n in runs)


def _chain_types(unit_reports, sequence):
    """Counts of bond, angle and dihedral types along one chain of the given unit codes (tail bead of unit k bonded to
    the head bead of unit k + 1)."""
    from collections import Counter
    types, edges, prev_tail = [], [], None
    for code in sequence:
        r = unit_reports[code]
        off = len(types)
        types += [b['bead_type'] for b in r['beads']]
        edges += [(a + off, b + off) for a, b in r['edges']]
        head, tail = r['joints']
        if prev_tail is not None:
            edges.append((prev_tail, head + off))
        prev_tail = tail + off
    nbr = {i: set() for i in range(len(types))}
    for a, b in edges:
        nbr[a].add(b)
        nbr[b].add(a)
    canon = lambda x: '-'.join(min(x, x[::-1]))                    # noqa: E731
    bonds = Counter(canon((types[a], types[b])) for a, b in edges)
    angles = Counter(canon((types[a], types[b], types[c])) for b in nbr
                     for a in nbr[b] for c in nbr[b] if a < c)
    dihedrals = Counter(canon((types[a], types[b], types[c], types[d]))
                        for b, c in edges for a in nbr[b] - {c} for d in nbr[c] - {b, a})
    return bonds, angles, dihedrals


def _ends(plain, sp, top, caps, name, warnings):
    """(first, last) end units of one repeat unit: given ("first" / "last" of the species, or of the system for a
    homopolymer) or capped."""
    notes = []
    first = _plain(sp.get('first') or top.get('first') or '') or cap_unit(plain, 'tail', caps.get('head', 'H'), notes)
    last = _plain(sp.get('last') or top.get('last') or '') or cap_unit(plain, 'head', caps.get('tail', 'H'), notes)
    warnings += [f'{name}: {x}' for x in notes]
    for what, u in (('first', first), ('last', last)):
        if u.count('[*]') != 1:
            raise ValueError(f'the {what} unit needs exactly one attachment point (*)')
        if not end_unit_axis_ok(u, what):
            raise ValueError(f'the {what} unit {u!r}: write it ' + ('from its capped end, e.g. CC[*] rather than [*]CC'
                             if what == 'first' else 'from its attachment point, e.g. [*]CC')
                             + ' (AutoPoly orients end units by their first atom)')
    return first, last


def derive(spec, key='system'):
    """{'ok', 'errors', 'warnings', 'spec' (defaults filled in), 'species' (per-species report), 'size', 'recipe',
    'mapping', 'mapping_check', 'bonded_types', 'library'}."""
    from ..mapping import spec as mspec
    errors, warnings = [], []
    if not isinstance(spec, dict):
        return {'ok': False, 'errors': ['the system is a JSON object (see ascg_core.aa.system_spec)'], 'warnings': []}
    cls = spec.get('system_class')
    if cls not in CLASSES:
        return {'ok': False, 'errors': [f'system_class must be one of {", ".join(CLASSES)}'], 'warnings': []}
    species = spec.get('species') or []
    if not species:
        return {'ok': False, 'errors': ['no species: give each molecule or repeat unit as a SMILES'], 'warnings': []}
    if cls == 'liquid' and len(species) != 1:
        errors.append('a liquid has one species; several are a liquid_mixture')
    copolymer = cls == 'polymer' and len(species) > 1
    if copolymer and not (spec.get('sequence') or spec.get('random')):
        return {'ok': False, 'errors': ['a copolymer needs "sequence" (e.g. "A*5 B*5", "A B A B ...") or "random" '
                                        '({"A": 0.5, "B": 0.5} with "repeat_units")'], 'warnings': []}
    s = {**{k: v for k, v in DEFAULTS.items() if k != 'molecules' or cls != 'polymer'}, **spec}
    if cls != 'polymer':
        for k in ('repeat_units', 'tacticity', 'end_groups', 'target_cg_beads', 'min_chains'):
            s.pop(k, None)
    # bead numbers: all species annotated (the mapping comes with the system) or none
    annotated = [bool(re.search(r':\d+\]', sp.get('smiles', ''))) for sp in species]
    if any(annotated) and not all(annotated):
        errors.append('bead numbers on some species only: annotate every species or none')
    mapping = check = None
    if all(annotated):
        msp = [{'format': mspec.ANNOTATED, 'smiles': sp['smiles'], 'bead_types': sp.get('bead_types')}
               for sp in species]
        mapping = msp[0] if len(msp) == 1 else {'species': msp}
        check = mspec.check(mapping, cls)
        errors += [f'mapping: {e}' for e in check['errors']]
        warnings += [f'mapping: {w}' for w in check['warnings']]
    reports, units, molecules, parts = [], {}, [], {}
    codes = {}
    for k, sp in enumerate(species):
        name = sp.get('name') or f'species{k + 1}'
        try:
            plain = _build_smiles(sp.get('smiles', ''))
            n_star = plain.count('[*]')
            if cls == 'polymer':
                if n_star != 2:
                    raise ValueError('a repeat unit needs exactly two attachment points (*), head first')
                eg = s.get('end_groups', 'H')
                caps = eg if isinstance(eg, dict) else {'head': eg, 'tail': eg}
                if any(caps.get(x, 'H') not in CAPS for x in ('head', 'tail')):
                    raise ValueError(f'end_groups: {" or ".join(CAPS)}, or {{"head": ..., "tail": ...}} (or give '
                                     '"first" / "last" end units)')
                first, last = _ends(plain, sp, {} if copolymer else spec, caps, name, warnings)
                code = _code(sp.get('name'), k)
                if code in units:
                    raise ValueError(f'two species give the unit code {code}: name them differently')
                codes[code] = code
                if sp.get('name'):
                    codes[str(sp['name'])] = code
                units[code] = {'first': first, 'middle': plain, 'last': last}
                parts[code] = {p: _with_h(u) for p, u in units[code].items()}
                info = parts[code]['middle']
                reports.append({'name': name, 'code': code, 'kind': 'repeat_unit', 'unit_formula': info['formula'],
                                'units': units[code]})
            else:
                if n_star:
                    raise ValueError('a liquid species is a whole molecule (no *)')
                info = _with_h(plain)
                reports.append({'name': name, 'kind': 'molecule', 'formula': info['formula'], 'smiles': plain,
                                'atoms': info['atoms'], 'mass': round(info['mass'], 3)})
                molecules.append({'smiles': plain, 'count': sp.get('count'), 'fraction': sp.get('fraction')})
            if info['charge']:
                warnings.append(f'{name}: net charge {info["charge"]:+d}; ions need counter-ions and charge checks')
            if info['radicals']:
                errors.append(f'{name}: atoms with unpaired electrons (bracket atoms need their hydrogens: [CH2])')
            odd = sorted(info['elements'] - OPLS_ELEMENTS)
            if odd:
                warnings.append(f'{name}: {", ".join(odd)} - OPLS-AA in AutoPoly may lack these terms; a published '
                                'force field may be needed (a force-field override)')
            if '@' in plain or '/' in plain or '\\' in plain:
                warnings.append(f'{name}: stereo marks are ignored' + ('; tacticity sets the chain stereochemistry'
                                                                       if cls == 'polymer' else ''))
        except (ValueError, KeyError) as e:
            errors.append(f'{name}: {e}')
    if errors:
        return {'ok': False, 'errors': errors, 'warnings': warnings, 'spec': s, 'species': reports}
    size: dict = {}
    task: dict = {'class': cls}
    bonded = None
    if cls == 'polymer':
        if s['tacticity'] not in TACTICITY:
            errors.append(f'tacticity must be one of {", ".join(TACTICITY)}')
        unit_reports = {}
        if check:
            for r, cr in zip(reports, check['species']):
                unit_reports[r['code']] = mspec._species_report(mspec.normalise_species(
                    {'format': mspec.ANNOTATED, 'smiles': species[reports.index(r)]['smiles'],
                     'bead_types': species[reports.index(r)].get('bead_types')}))
                r['beads_per_unit'] = cr['beads_per_unit']
        per_unit = {r['code']: r.get('beads_per_unit') for r in reports}
        try:
            if spec.get('random'):
                weights = {codes.get(n): float(w) for n, w in spec['random'].items()}
                if None in weights or any(w <= 0 for w in weights.values()):
                    raise ValueError(f'random: weights > 0 for species of the system ({", ".join(units)})')
                n = int(s['repeat_units'])
                tot = sum(weights.values())
                frac = {c: w / tot for c, w in weights.items()}
                mean = lambda part, key: sum(frac[c] * parts[c][part][key] for c in frac)      # noqa: E731
                atoms_chain = mean('first', 'atoms') + (n - 2) * mean('middle', 'atoms') + mean('last', 'atoms')
                mass_chain = mean('first', 'mass') + (n - 2) * mean('middle', 'mass') + mean('last', 'mass')
                beads_chain = n * sum(frac[c] * per_unit[c] for c in frac) if check else None
                task.update(random=weights, length=n)
                seq_desc = f'random, {n} units: ' + ', '.join(f'{c} {100 * f:.0f} %' for c, f in frac.items())
                missing = set(units) - set(weights)
                if missing:
                    raise ValueError(f'random: no weight for {", ".join(sorted(missing))}')
                if check:
                    import itertools
                    bonded = [_chain_types(unit_reports, combo) for combo in itertools.product(sorted(frac), repeat=min(4, n))]
                    bonded = tuple(sorted(set().union(*(b[i] for b in bonded))) for i in range(3))
            else:
                tokens = _expand(s.get('sequence') or f'{next(iter(units))}*{s["repeat_units"]}', codes)
                n = len(tokens)
                if spec.get('repeat_units') and int(spec['repeat_units']) != n and spec.get('sequence'):
                    raise ValueError(f'repeat_units {spec["repeat_units"]} but the sequence has {n} units')
                missing = set(units) - set(tokens)
                if missing:
                    raise ValueError(f'species {", ".join(sorted(missing))} not in the sequence')
                pos = lambda i: 'first' if i == 0 else 'last' if i == n - 1 else 'middle'     # noqa: E731
                atoms_chain = sum(parts[c][pos(i)]['atoms'] for i, c in enumerate(tokens))
                mass_chain = sum(parts[c][pos(i)]['mass'] for i, c in enumerate(tokens))
                beads_chain = sum(per_unit[c] for c in tokens) if check else None
                task['sequence'] = _compact(tokens)
                seq_desc = task['sequence']
                if check:
                    bonded = _chain_types(unit_reports, tokens)
        except (ValueError, KeyError) as e:
            errors.append(str(e))
            return {'ok': False, 'errors': errors, 'warnings': warnings, 'spec': s, 'species': reports}
        s['repeat_units'] = n
        if n < 2:
            errors.append('a chain needs at least 2 repeat units')
        elif n < 5:
            warnings.append(f'{n} repeat units: very short chains; the CG model will be for oligomers')
        if s.get('chains'):
            chains = int(s['chains'])
        elif beads_chain:
            chains = max(int(s['min_chains']), int(round(s['target_cg_beads'] / beads_chain)))
        else:
            chains = max(int(s['min_chains']), int(round(s['target_cg_beads'] / n)))
            warnings.append('no bead numbers in the SMILES: chains sized as one bead per repeat unit; give "chains" '
                            'or annotate the beads')
        if chains < 20:
            warnings.append(f'{chains} chains: fewer than 20 gives poor chain statistics')
        task.update(chains=chains, tacticity=s['tacticity'])
        if not copolymer and check:
            task['beads_per_repeat'] = per_unit[next(iter(units))]
        size = {'chains': chains, 'repeat_units': n, 'sequence': seq_desc, 'atoms': int(round(chains * atoms_chain)),
                'mass_amu': round(chains * mass_chain, 1),
                'cg_beads': int(round(chains * beads_chain)) if beads_chain else None}
        if bonded and copolymer and not spec.get('random'):
            for kind, counts in zip(('bond', 'angle', 'dihedral'), bonded):
                rare = sorted(t for t, c in counts.items() if c * chains < 100)
                if rare:
                    warnings.append(f'{kind} types {", ".join(rare)}: fewer than 100 in the system (e.g. one block '
                                    'junction per chain); their targets are noisier than the others')
        if bonded:
            bonded = {k: sorted(v) for k, v in zip(('bonds', 'angles', 'dihedrals'), bonded)}
    else:
        total = int(s.get('molecules') or DEFAULTS['molecules'])
        counts = [m['count'] for m in molecules]
        if all(c is not None for c in counts):
            counts = [int(c) for c in counts]
        elif all(m['fraction'] is not None for m in molecules):
            fr = [float(m['fraction']) for m in molecules]
            if abs(sum(fr) - 1) > 1e-6:
                warnings.append(f'mole fractions sum to {sum(fr):g}; normalised')
            counts = [int(round(total * f / sum(fr))) for f in fr]
        elif len(molecules) == 1:
            counts = [total]
        else:
            errors.append('a mixture needs a "count" or a "fraction" for every species')
            counts = [0] * len(molecules)
        if any(c < 1 for c in counts) and not errors:
            errors.append('every species needs at least one molecule')
        task['molecules'] = [{'smiles': m['smiles'], 'count': c} for m, c in zip(molecules, counts)]
        for r, c in zip(reports, counts):
            r['count'] = c
        per = check['species'] if check else None
        size = {'molecules': sum(counts), 'atoms': sum(r['atoms'] * c for r, c in zip(reports, counts)),
                'mass_amu': round(sum(r['mass'] * c for r, c in zip(reports, counts)), 1),
                'cg_beads': sum(p['beads_per_molecule'] * c for p, c in zip(per, counts)) if per else None}
        if check:
            bonded = {k: sorted({x for sp in check['species'] for x in sp[k]}) for k in ('bonds', 'angles', 'dihedrals')}
    if size.get('atoms', 0) > MAX_ATOMS:
        warnings.append(f"{size['atoms']} atoms: AA simulations of this size are expensive; consider fewer chains or "
                        'molecules')
    size['box_nm_at_1_g_cm3'] = round((size['mass_amu'] / 602.214076) ** (1 / 3), 2) if size.get('mass_amu') else None
    recipe = {'version': 1, 'provenance': f'derived by ascg from the system {key} (ascg_core.aa.system_spec)',
              'defaults': {'tacticity': s.get('tacticity', 'atactic')}, 'equilibration': dict(EQUILIBRATION),
              'units': units, 'tasks': {key: task}}
    library = None
    if mapping is not None:
        entry = {'key': key, 'id': key, 'name': spec.get('name') or key, 'system_class': cls, 'aliases': [],
                 'resolution': _resolution(check), 'mapping_status': 'proposed',
                 'recipe': {'file': 'recipes.json', 'task': key}, 'force_field_override': None}
        systems = []
        if cls == 'liquid_mixture':
            ids = []
            for k, sp in enumerate(species):
                cid = f'{key}__{_code(sp.get("name"), k).lower()}'
                ids.append(cid)
                systems.append({'key': cid, 'id': cid, 'name': sp.get('name') or cid, 'system_class': 'liquid',
                                'aliases': [], 'mapping_status': 'proposed', 'recipe': None,
                                'force_field_override': None,
                                'mapping': {'format': mspec.ANNOTATED, 'smiles': sp['smiles'],
                                            'bead_types': sp.get('bead_types')}})
            entry['mapping'] = {'format': mspec.ANNOTATED, 'components': ids}
        elif copolymer:
            entry['mapping'] = {'format': mspec.ANNOTATED, 'units': {
                r['code']: {'format': mspec.ANNOTATED, 'smiles': sp['smiles'], 'bead_types': sp.get('bead_types')}
                for r, sp in zip(reports, species)}}
        else:
            entry['mapping'] = mapping
        library = {'schema': 'ascg-library/1', 'name': f'system {key}', 'systems': [entry] + systems}
    return {'ok': not errors, 'errors': errors, 'warnings': warnings, 'spec': s, 'species': reports, 'size': size,
            'recipe': recipe, 'mapping': mapping, 'mapping_check': check, 'bonded_types': bonded, 'library': library}


def _resolution(check):
    sp = check['species']
    if sp and all(x['kind'] == 'repeat_unit' for x in sp) and len(sp) > 1:
        return ', '.join(str(x.get('beads_per_unit')) for x in sp) + ' beads per repeat unit'
    if len(sp) == 1:
        n = sp[0].get('beads_per_unit') or sp[0].get('beads_per_molecule')
        return f"{n} bead{'s' if n != 1 else ''} per {'repeat unit' if sp[0]['kind'] == 'repeat_unit' else 'molecule'}"
    return ', '.join(str(x.get('beads_per_molecule')) for x in sp) + ' beads per molecule'
