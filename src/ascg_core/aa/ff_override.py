"""Force-field override of a converted GROMACS topology (ascg_core.aa.lmp2gmx format), by chemical environment.

For molecules the stock OPLS-AA typing does not cover (PDMS: no siloxane D-unit Si, no Si-O torsions), AutoPoly /
moltemplate still provide the structure and the bond graph; this step then replaces every atom's charge and LJ type and
regenerates all angles and proper dihedrals from the bond graph with the parameters of a published force field
(benchmarks/ff/*.json, LAMMPS real units). A bonded term whose element pattern the force field does not define is an
error (fail closed), so no stock OPLS-AA value survives silently. 1-4 pairs and their OPLS scaling (0.5 / 0.5) come
from lmp2gmx unchanged.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

KJ = 4.184


def _pdms_labels(el, nb):
    """SiM / SiD (trimethylsilyl / dimethylsiloxane Si), O_end (bonded to an SiM) / O, C_M / C_D, H_M / H_D."""
    lab = {}
    for i, e in el.items():
        if e == 'Si':
            lab[i] = 'SiM' if sum(el[j] == 'C' for j in nb[i]) == 3 else 'SiD'
    for i, e in el.items():
        if e == 'O':
            sis = [j for j in nb[i] if el[j] == 'Si']
            if len(sis) != 2:
                raise ValueError(f'PDMS override: O {i} is not a siloxane O (neighbours {[el[j] for j in nb[i]]})')
            lab[i] = 'O_end' if any(lab[j] == 'SiM' for j in sis) else 'O'
        elif e == 'C':
            si = [j for j in nb[i] if el[j] == 'Si']
            if len(si) != 1:
                raise ValueError(f'PDMS override: C {i} is not a methyl on Si')
            lab[i] = 'C_M' if lab[si[0]] == 'SiM' else 'C_D'
    for i, e in el.items():
        if e == 'H':
            c = nb[i][0]
            lab[i] = 'H_M' if lab[c] == 'C_M' else 'H_D'
    missing = [i for i in el if i not in lab]
    if missing:
        raise ValueError(f'PDMS override: atoms without a label: {[el[i] for i in missing][:5]}')
    return lab


SCHEMES = {'pdms': _pdms_labels}
ELEMENT = {1: 'H', 12: 'C', 14: 'N', 16: 'O', 19: 'F', 28: 'Si', 32: 'S', 35: 'Cl'}


def _key(elements, table):
    k = '-'.join(elements)
    r = '-'.join(elements[::-1])
    if k in table:
        return k
    if r in table:
        return r
    raise ValueError(f'force field defines no term for {k}')


def apply(top_in, top_out, spec_file):
    """Rewrite `top_in` (lmp2gmx output) into `top_out` with the force field of `spec_file`; returns a report."""
    from .lmp2gmx import fourier_to_rb
    spec = json.loads(Path(spec_file).read_text())
    label_fn = SCHEMES[spec['scheme']]
    lines = Path(top_in).read_text().splitlines()
    # split into header (defaults + atomtypes), molecule blocks, tail (system / molecules)
    out, i = [], 0
    while i < len(lines) and not lines[i].strip().startswith('[ atomtypes ]'):
        out.append(lines[i]); i += 1
    out += ['[ atomtypes ]', '; name  mass  charge  ptype  sigma(nm)  epsilon(kJ/mol)   (' + spec['name'] + ')']
    for name, a in spec['atoms'].items():
        out.append(f"  ff_{name}  {a['mass']:.4f}  0.0  A  {a['sigma'] / 10:.6f}  {a['epsilon'] * KJ:.6f}")
    while i < len(lines) and not lines[i].strip().startswith('[ moleculetype ]'):
        i += 1
    counts = defaultdict(int)
    n_mol = 0
    while i < len(lines):
        if lines[i].strip().startswith('[ system ]'):
            out += lines[i:]
            break
        assert lines[i].strip() == '[ moleculetype ]', lines[i]
        name = lines[i + 1].split()[0]
        i += 2
        sec, block = None, defaultdict(list)
        while i < len(lines) and not lines[i].strip().startswith(('[ moleculetype ]', '[ system ]')):
            s = lines[i].strip()
            m = re.match(r'\[\s*(\w+)\s*\]', s)
            if m:
                sec = m[1]
            elif s and not s.startswith(';') and sec:
                block[sec].append(s.split())
            i += 1
        if any(len(d) > 4 and d[4] == '4' for d in block['dihedrals']):
            raise ValueError(f'{name}: impropers present; this override regenerates proper dihedrals only')
        atoms = block['atoms']
        el = {int(a[0]): ELEMENT.get(round(float(a[7])), 'X') for a in atoms}
        nb = defaultdict(list)
        for b in block['bonds']:
            x, y = int(b[0]), int(b[1])
            nb[x].append(y); nb[y].append(x)
        lab = label_fn(el, nb)
        q = sum(spec['atoms'][lab[int(a[0])]]['charge'] for a in atoms)
        if abs(q) > 1e-6:
            raise ValueError(f'{name}: override charges sum to {q:+.6f} e')
        out += ['', '[ moleculetype ]', f'  {name}  3', '', '[ atoms ]', '; nr type resnr res atom cgnr charge mass']
        for a in atoms:
            nr = int(a[0]); s = spec['atoms'][lab[nr]]
            out.append(f"  {nr}  ff_{lab[nr]}  1  {a[3]}  {el[nr]}{nr % 1000}  {nr}  {s['charge']:.6f}  {s['mass']:.4f}")
        out += ['', '[ bonds ]']
        for b in block['bonds']:
            x, y = int(b[0]), int(b[1])
            K, r0 = spec['bonds'][_key([el[x], el[y]], spec['bonds'])]
            out.append(f'  {x}  {y}  1  {r0 / 10:.6f}  {2 * K * KJ * 100:.4f}')
            counts['bonds'] += 1
        out += ['', '[ pairs ]'] + [f'  {p[0]}  {p[1]}  1' for p in block['pairs']]
        out += ['', '[ angles ]']                      # regenerated from the bond graph
        for j in sorted(nb):
            for x in nb[j]:
                for y in nb[j]:
                    if x < y:
                        K, th = spec['angles'][_key([el[x], el[j], el[y]], spec['angles'])]
                        out.append(f'  {x}  {j}  {y}  1  {th:.6f}  {2 * K * KJ:.6f}')
                        counts['angles'] += 1
        out += ['', '[ dihedrals ]', '; proper dihedrals as Ryckaert-Bellemans (OPLS K1..K4 converted); zero terms omitted']
        for j in sorted(nb):
            for k in nb[j]:
                if k <= j:
                    continue
                for x in nb[j]:
                    if x == k:
                        continue
                    for y in nb[k]:
                        if y in (j, x):
                            continue
                        c = spec['dihedrals'][_key([el[x], el[j], el[k], el[y]], spec['dihedrals'])]
                        counts['dihedrals_considered'] += 1
                        if any(abs(v) > 0 for v in c):
                            C = fourier_to_rb(*(v * KJ for v in c))
                            out.append(f'  {x}  {j}  {k}  {y}  3  ' + '  '.join(f'{v:.6f}' for v in C))
                            counts['dihedrals'] += 1
        n_mol += 1
    Path(top_out).write_text('\n'.join(out) + '\n')
    return {'force_field': spec['name'], 'source': spec['source'], 'molecules': n_mol, **counts}
