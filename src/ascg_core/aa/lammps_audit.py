"""Deterministic, non-destructive checks of AutoPoly-generated LAMMPS data.

Supports atom_style full or charge. Reports unknowns instead of pretending a
file with atom coordinates automatically defines a valid OPLS-AA topology.
"""
from __future__ import annotations
from pathlib import Path
import re
import math
from dataclasses import dataclass, asdict

KNOWN_SECTIONS={'Masses','Atoms','Bonds','Angles','Dihedrals','Impropers',
                'Velocities','Pair Coeffs','Bond Coeffs','Angle Coeffs',
                'Dihedral Coeffs','Improper Coeffs','Ellipsoids','Lines','Triangles'}


class TopologyAuditError(ValueError): pass


@dataclass
class Audit:
    atoms:int
    molecules:int|None
    types:int
    bonds:int
    total_mass_amu:float
    density_g_cm3:float
    net_charge_e:float|None
    charge_source:str
    box_lengths_A:tuple[float,float,float]
    validated_components:tuple[str,...]
    # atom_style full only: net charge of each molecule (min, max) and how many are nonzero
    molecule_charge_range_e:tuple[float,float]|None=None
    charged_molecules:int|None=None

    def to_dict(self):return asdict(self)


def audit_lammps_data(data_file:str|Path,*,expected_density:float|None=None,
                      tolerance:float=0.03) -> Audit:
    """Audit topology and mass density; requires explicit atom_style label.

    If atom_style is not explicit, refuses to infer charge column positions.
    Does not validate force-field chemical semantics; those require a
    chemistry-aware checker and AutoPoly's typed .lt + units.json data.
    """
    path=Path(data_file)
    lines=path.read_text().splitlines()
    sections={}
    header=[]
    section=None
    for raw in lines:
        line=raw.strip()
        if not line:continue
        pre=line.split('#',1)[0].strip()
        name=pre.strip()
        if name in KNOWN_SECTIONS:
            section=name
            sections.setdefault(name,{'title':line,'rows':[]})
            continue
        if section is None:
            header.append(pre)
        elif pre:
            sections[section]['rows'].append(pre)
    counts={}
    box={}
    for line in header:
        m=re.match(r'^(\d+)\s+(atoms|bonds|angles|dihedrals|impropers|atom types|bond types)',line)
        if m:counts[m.group(2)]=int(m.group(1))
        b=re.match(r'^([-+.\dEe]+)\s+([-+.\dEe]+)\s+(xlo xhi|ylo yhi|zlo zhi)$',line)
        if b:box[b.group(3)[0]]=(float(b.group(1)),float(b.group(2)))
    if not {'x','y','z'}.issubset(box):raise TopologyAuditError('Missing orthorhombic box bounds')
    cell=tuple(box[c][1]-box[c][0] for c in 'xyz')
    if min(cell)<=0:raise TopologyAuditError('Nonpositive box dimension')
    if 'Masses' not in sections or 'Atoms' not in sections:
        raise TopologyAuditError('Missing Masses or Atoms section')
    masses={}
    for line in sections['Masses']['rows']:
        bits=line.split()
        if len(bits)<2:raise TopologyAuditError(f'Invalid Masses row: {line}')
        masses[int(bits[0])]=float(bits[1])
    if not masses or any((not math.isfinite(m) or m<=0) for m in masses.values()):
        raise TopologyAuditError('Invalid atom masses')
    atom_header=sections['Atoms']['title'].lower()
    if '# full' in atom_header:
        kind='full'; ncol=7
    elif '# charge' in atom_header:
        kind='charge'; ncol=6
    else:
        raise TopologyAuditError('Require Atoms # full or Atoms # charge (cannot infer charges)')
    seen=set()
    n_mol=set()
    mol_charge={}
    total_mass=0.0
    charge=0.0
    for line in sections['Atoms']['rows']:
        cols=line.split()
        if len(cols)<ncol:raise TopologyAuditError(f'Invalid atom row: {line}')
        idx=int(cols[0])
        if idx in seen:raise TopologyAuditError(f'Duplicate atom id {idx}')
        seen.add(idx)
        if kind=='full':
            n_mol.add(int(cols[1])); ty=int(cols[2]); q=float(cols[3]); xyz=cols[4:7]
            mol_charge[int(cols[1])]=mol_charge.get(int(cols[1]),0.0)+q
        else:
            ty=int(cols[1]); q=float(cols[2]); xyz=cols[3:6]
        if ty not in masses:raise TopologyAuditError(f'Missing mass for atom type {ty}')
        if not math.isfinite(q) or not all(math.isfinite(float(v)) for v in xyz):
            raise TopologyAuditError(f'Nonfinite atom property id={idx}')
        charge+=q
        total_mass+=masses[ty]
    if len(seen)!=counts.get('atoms'):
        raise TopologyAuditError(f"Atom header count {counts.get('atoms')} != {len(seen)}")
    if 'Bonds' in sections:
        rows=sections['Bonds']['rows']
        if len(rows)!=counts.get('bonds',0):
            raise TopologyAuditError('Bond count mismatch')
        for line in rows:
            cols=line.split()
            if len(cols)<4 or int(cols[2]) not in seen or int(cols[3]) not in seen:
                raise TopologyAuditError(f'Invalid bond atom references: {line}')
    elif counts.get('bonds',0):
        raise TopologyAuditError('Missing declared Bonds section')
    density=1.66053906660*total_mass/math.prod(cell)
    if expected_density is not None and abs(density/expected_density-1)>tolerance:
        raise TopologyAuditError(f'Initial density {density:.6g} g/cm3 outside tolerance of {expected_density}')
    qs=list(mol_charge.values())
    return Audit(atoms=len(seen),molecules=len(n_mol) if kind=='full' else None,
                 types=len(masses),bonds=counts.get('bonds',0),
                 total_mass_amu=total_mass,density_g_cm3=density,
                 net_charge_e=charge,charge_source=f'Atoms # {kind}',
                 box_lengths_A=cell,
                 validated_components=('counts','atom types','mass density','charges in data','bond references'),
                 molecule_charge_range_e=(min(qs),max(qs)) if qs else None,
                 charged_molecules=sum(abs(q)>1e-6 for q in qs) if qs else None)


def neutralize_molecules(data_file: str | Path, out_file: str | Path, *,
                         adjust_opls_types=(), tolerance: float = 1e-6) -> dict:
    """Write a copy of a LAMMPS data file in which every molecule has zero net charge.

    The OPLS type of each LAMMPS atom type is the leading integer of its Masses
    comment (AutoPoly/moltemplate '# 182_bCT_...'). In each charged molecule the
    residual charge is removed by an equal shift on that molecule's atoms of
    `adjust_opls_types`; a charged molecule with no such atom is an error. Only
    the Atoms charge column changes. Returns the per-type shifts applied.
    """
    path=Path(data_file)
    lines=path.read_text().splitlines()
    opls={}
    section=None
    atoms=[]   # (line index, atom id, mol, lammps type, charge)
    for i,raw in enumerate(lines):
        line=raw.strip()
        pre=line.split('#',1)[0].strip()
        if pre in KNOWN_SECTIONS:
            section=pre; continue
        if not pre:continue
        if section=='Masses':
            m=re.match(r'\s*#\s*(\d+)',raw.split(pre,1)[1]) if '#' in raw else None
            if m:opls[int(pre.split()[0])]=int(m.group(1))
        elif section=='Atoms':
            cols=pre.split()
            atoms.append((i,int(cols[0]),int(cols[1]),int(cols[2]),float(cols[3])))
    if not atoms:raise TopologyAuditError('No Atoms section (atom_style full required)')
    adjust=set(int(t) for t in adjust_opls_types)
    by_mol={}
    for rec in atoms:by_mol.setdefault(rec[2],[]).append(rec)
    new_q={}
    shifts={}
    for mol,recs in by_mol.items():
        q=sum(r[4] for r in recs)
        if abs(q)<=tolerance:continue
        targets=[r for r in recs if opls.get(r[3]) in adjust]
        if not targets:
            raise TopologyAuditError(f'Molecule {mol} has net charge {q:+.4f} e and no atom of OPLS types {sorted(adjust)}')
        dq=-q/len(targets)
        for r in targets:
            new_q[r[0]]=r[4]+dq
            shifts.setdefault(opls[r[3]],[]).append(dq)
    out=list(lines)
    for i,q in new_q.items():
        cols=lines[i].split('#',1)[0].split()
        cols[3]=f'{q:.10f}'
        out[i]=' '.join(cols)+(' #'+lines[i].split('#',1)[1] if '#' in lines[i] else '')
    Path(out_file).write_text('\n'.join(out)+'\n')
    check=audit_lammps_data(out_file)
    if check.charged_molecules:
        raise TopologyAuditError('Neutralization failed')
    return {'source':str(path),'output':str(out_file),'adjust_opls_types':sorted(adjust),
            'molecules_shifted':len({r[2] for r in atoms if r[0] in new_q}),
            'shift_e_by_opls_type':{str(t):[min(v),max(v)] for t,v in shifts.items()},
            'net_charge_after_e':check.net_charge_e}
