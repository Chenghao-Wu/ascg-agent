#!/usr/bin/env python3
"""OPLS-AA typing / charge audit.

Per-molecule net-charge and atom-environment audit of an all-atom LAMMPS data file (atom_style full), e.g. the
OPLS-AA output of AutoPoly / moltemplate, with hints for known automatic-typing pitfalls.

    python -m ascg_core.aa.opls_typing_audit system.data [--charges system.in.charges] [--tolerance 1e-4] [--json report.json]

Molecules with identical (type, charge) sequences are reported once as a template. For each template: atom count,
net charge, and the table type / charge / environment / count / description, where the environment is the element
followed by its bonded neighbours' elements (e.g. C(CHHO)). Findings:

  NONNEUTRAL  a molecule's net charge exceeds --tolerance (automatic typing produced a charge group that does not sum
              to zero; per-type residual shifts are not attempted);
  HINT_*      an atom whose environment matches a known pitfall of rule-based OPLS-AA typing (see GUIDE.md):
              ESTER_ALKOXY_AS_ETHER, ARYL_ESTER_CARBONYL_AS_ALIPHATIC, NITRILE_ALPHA_AS_ALKANE, UNTYPED.
Hints use the type descriptions of the charges file (moltemplate 'set type N charge Q # text'); without it only
NONNEUTRAL and UNTYPED are evaluated.

Exit codes: 0 every molecule neutral and no hint; 1 findings; 2 input error.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ELEMENTS = {1: 'H', 4: 'He', 7: 'Li', 12: 'C', 14: 'N', 16: 'O', 19: 'F', 23: 'Na', 28: 'Si', 31: 'P', 32: 'S',
            35: 'Cl', 39: 'K', 80: 'Br', 127: 'I'}
SECTIONS = {'Masses', 'Atoms', 'Velocities', 'Bonds', 'Angles', 'Dihedrals', 'Impropers', 'Pair Coeffs',
            'PairIJ Coeffs', 'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs'}


class InputError(Exception):
    pass


def read_data(path):
    sec, data = None, defaultdict(list)
    for raw in Path(path).read_text().splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        if line in SECTIONS:
            sec = line
            continue
        if sec:
            data[sec].append(line.split())
    if not data.get('Atoms'):
        raise InputError(f'{path}: no Atoms section (atom_style full required)')
    if not data.get('Masses'):
        raise InputError(f'{path}: no Masses section')
    mass = {int(r[0]): float(r[1]) for r in data['Masses']}
    atoms = {}
    for r in data['Atoms']:
        if len(r) < 7:
            raise InputError(f'{path}: Atoms line too short for atom_style full: {" ".join(r)}')
        atoms[int(r[0])] = (int(r[1]), int(r[2]), float(r[3]))
    nb = defaultdict(list)
    for r in data.get('Bonds', []):
        a, b = int(r[2]), int(r[3])
        nb[a].append(b)
        nb[b].append(a)
    return mass, atoms, nb


def read_descriptions(path):
    """moltemplate charges file: 'set type N charge Q # description' -> {N: (Q, description)}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r'\s*set\s+type\s+(\d+)\s+charge\s+(\S+)\s*(?:#\s*(.*))?', line)
        if m:
            out[int(m[1])] = (float(m[2]), (m[3] or '').strip())
    return out


def element(m):
    return ELEMENTS.get(int(round(m)), 'X')


def audit(data_file, charges_file=None, tolerance=1e-4):
    mass, atoms, nb = read_data(data_file)
    desc = read_descriptions(charges_file) if charges_file else {}
    el = {i: element(mass[t]) for i, (_, t, _) in atoms.items()}
    env = {i: el[i] + '(' + ''.join(sorted(el[j] for j in nb[i])) + ')' for i in atoms}
    d = lambda i: desc.get(atoms[i][1], (None, ''))[1].lower()

    def carbonyl_c(i):            # sp2 C with a terminal O
        return el[i] == 'C' and any(el[j] == 'O' and len(nb[j]) == 1 for j in nb[i])

    def aromatic_c(i):            # ring C with three heavy/aromatic-like neighbours typed as benzene / aromatic
        return el[i] == 'C' and ('benzene' in d(i) or ' ca ' in f' {d(i)} ' or 'aromatic' in d(i))

    hints = []
    for i in atoms:
        if el[i] == 'X' or (desc and atoms[i][1] not in desc):
            hints.append(('UNTYPED', i))
        if not desc:
            continue
        if el[i] == 'C' and 'ether' in d(i) and len(nb[i]) == 4:
            if any(el[o] == 'O' and len(nb[o]) == 2 and any(carbonyl_c(c) for c in nb[o] if c != i) for o in nb[i]):
                hints.append(('ESTER_ALKOXY_AS_ETHER', i))
        if carbonyl_c(i) and 'ester' in d(i) and not any(w in d(i) for w in ('benzoate', 'aryl', 'aromatic')):
            if any(aromatic_c(j) for j in nb[i]):
                hints.append(('ARYL_ESTER_CARBONYL_AS_ALIPHATIC', i))
        if el[i] == 'C' and 'alkane' in d(i) and sum(el[j] == 'H' for j in nb[i]) == 3:
            if any(el[c] == 'C' and len(nb[c]) == 2 and any(el[n] == 'N' and len(nb[n]) == 1 for n in nb[c])
                   for c in nb[i]):
                hints.append(('NITRILE_ALPHA_AS_ALKANE', i))

    by_mol = defaultdict(list)
    for i in sorted(atoms):
        by_mol[atoms[i][0]].append(i)
    templates = {}
    for mol, ids in by_mol.items():
        sig = tuple((atoms[i][1], round(atoms[i][2], 6)) for i in ids)
        templates.setdefault(sig, []).append(mol)
    hint_by_atom = defaultdict(list)
    for code, i in hints:
        hint_by_atom[i].append(code)
    report = {'data': str(data_file), 'charges': str(charges_file) if charges_file else None, 'tolerance': tolerance,
              'atoms': len(atoms), 'molecules': len(by_mol), 'templates': [], 'findings': []}
    for sig, mols in templates.items():
        ids = by_mol[mols[0]]
        net = sum(atoms[i][2] for i in ids)
        rows = Counter((atoms[i][1], round(atoms[i][2], 6), env[i]) for i in ids)
        t = {'molecules': len(mols), 'example_molecule': mols[0], 'atoms': len(ids), 'net_charge_e': round(net, 6),
             'types': [{'type': ty, 'charge': q, 'environment': e, 'count': n,
                        'description': desc.get(ty, (None, ''))[1]} for (ty, q, e), n in sorted(rows.items())]}
        report['templates'].append(t)
        if abs(net) > tolerance:
            report['findings'].append({'code': 'NONNEUTRAL', 'molecules': len(mols), 'example_molecule': mols[0],
                                       'net_charge_e': round(net, 6)})
        seen = Counter()
        for i in ids:
            for code in hint_by_atom.get(i, []):
                seen[(code, atoms[i][1], env[i])] += 1
        for (code, ty, e), n in sorted(seen.items()):
            report['findings'].append({'code': f'HINT_{code}', 'molecules': len(mols), 'example_molecule': mols[0],
                                       'type': ty, 'environment': e, 'atoms_per_molecule': n,
                                       'description': desc.get(ty, (None, ''))[1]})
    report['verdict'] = 'FINDINGS' if report['findings'] else 'NEUTRAL_NO_HINTS'
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data')
    ap.add_argument('--charges', default=None, help="moltemplate charges file ('set type N charge Q # description')")
    ap.add_argument('--tolerance', type=float, default=1e-4, help='largest accepted |net charge| per molecule (e)')
    ap.add_argument('--json', default=None, help='write the full report here')
    a = ap.parse_args(argv)
    try:
        rep = audit(a.data, a.charges, a.tolerance)
    except (InputError, OSError, ValueError, IndexError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1) + '\n')
    print(f"{rep['data']}: {rep['atoms']} atoms, {rep['molecules']} molecules, {len(rep['templates'])} templates")
    for t in rep['templates']:
        print(f"  template x{t['molecules']} ({t['atoms']} atoms): net charge {t['net_charge_e']:+.4f} e")
        for r in t['types']:
            print(f"    type {r['type']:4d}  q {r['charge']:+.4f}  x{r['count']:<4d} {r['environment']:12s} "
                  f"{r['description'][:60]}")
    for f in rep['findings']:
        print('  ' + ' '.join(f'{k}={v}' for k, v in f.items()))
    print(rep['verdict'])
    return 1 if rep['findings'] else 0


if __name__ == '__main__':
    sys.exit(main())
