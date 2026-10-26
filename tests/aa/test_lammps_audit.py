from pathlib import Path
import pytest
from ascg_core.aa.lammps_audit import audit_lammps_data,TopologyAuditError

DATA='''LAMMPS data example

2 atoms
1 bonds
1 atom types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 12.0 # C

Atoms # full

1 1 1 -0.1 2.0 2.0 2.0
2 1 1 0.1 3.0 3.0 3.0

Bonds

1 1 1 2
'''

def test_mass_charge_bonds(tmp_path):
    p=tmp_path/'system.data';p.write_text(DATA)
    r=audit_lammps_data(p)
    assert r.atoms==2 and r.bonds==1
    assert r.net_charge_e==pytest.approx(0)
    assert r.density_g_cm3==pytest.approx(24.0*1.66053906660/1000)


def test_density_rejection(tmp_path):
    p=tmp_path/'system.data';p.write_text(DATA)
    with pytest.raises(TopologyAuditError,match='density'):
        audit_lammps_data(p,expected_density=0.9)


def test_wrong_mass_or_bond_reference(tmp_path):
    p=tmp_path/'system.data';p.write_text(DATA.replace('1 1 1 2','1 1 1 4'))
    with pytest.raises(TopologyAuditError,match='Invalid bond'):
        audit_lammps_data(p)
    p.write_text(DATA.replace('Atoms # full','Atoms'))
    with pytest.raises(TopologyAuditError,match='Require Atoms'):
        audit_lammps_data(p)


def test_molecule_charges(tmp_path):
    p=tmp_path/'system.data';p.write_text(DATA.replace('2 1 1 0.1 3.0 3.0 3.0','2 2 1 0.1 3.0 3.0 3.0'))
    r=audit_lammps_data(p)
    assert r.molecules==2 and r.charged_molecules==2
    assert r.molecule_charge_range_e==pytest.approx((-0.1,0.1))


def test_neutralize_molecules(tmp_path):
    from ascg_core.aa.lammps_audit import neutralize_molecules
    data=DATA.replace('1 12.0 # C','1 12.0 # 182_bCT_aCT').replace('2 1 1 0.1 3.0','2 1 1 0.05 3.0')
    p=tmp_path/'system.data';p.write_text(data)
    assert audit_lammps_data(p).charged_molecules==1
    r=neutralize_molecules(p,tmp_path/'n.data',adjust_opls_types=[182])
    a=audit_lammps_data(tmp_path/'n.data')
    assert a.charged_molecules==0 and a.net_charge_e==pytest.approx(0,abs=1e-9)
    assert r['shift_e_by_opls_type']['182']==pytest.approx([0.025,0.025])
    with pytest.raises(TopologyAuditError,match='no atom of OPLS'):
        neutralize_molecules(p,tmp_path/'m.data',adjust_opls_types=[999])
