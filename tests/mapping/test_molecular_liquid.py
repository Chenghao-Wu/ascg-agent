"""Molecular liquids: AutoPoly 'single' molecule layouts and system-library mappings in rdkit-heavy-atom-indices form
(acetone CH3 | C=O | CH3 on a two-molecule build written as AutoPoly writes it)."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('rdkit')
pytest.importorskip('networkx')
from ascg_core.mapping.cg_mapping import MappingError, build_cg_topology, read_data, read_layout  # noqa: E402
from ascg_core.mapping.formats import library_inputs, molecular_inputs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ATOMS = [('C1', 1, -0.18), ('C2', 2, 0.47), ('C3', 1, -0.18), ('O4', 3, -0.47)] + [(f'H{i}', 4, 0.06) for i in range(5, 11)]
BONDS = [('C1', 'C2'), ('C2', 'C3'), ('C2', 'O4'), ('C1', 'H5'), ('C1', 'H6'), ('C1', 'H7'), ('C3', 'H8'), ('C3', 'H9'),
         ('C3', 'H10')]


def unit_lt(variant, order):
    atoms = '\n'.join(f'\t$atom:{ATOMS[i][0]} $mol:... @atom:{ATOMS[i][1]} {ATOMS[i][2]:.4f} 0.0 0.0 0.0' for i in order)
    bonds = '\n'.join(f'\t$bond:{a}{b}\t$atom:{a}\t$atom:{b}' for a, b in BONDS)
    return (f'import "oplsaa.lt"\n{variant} inherits OPLSAA {{\n\n  write("Data Atoms") {{\n{atoms}\n  }}\n\n'
            f"  write('Data Bond List') {{\n{bonds}\n  }}\n}}  # {variant}\n")


def acetone_system(d):
    """Two molecules; the second template lists its atoms in another order (as AutoPoly's _T1 variants may)."""
    inp = d/'moltemplate'/'input'
    inp.mkdir(parents=True)
    orders = {'monomer_0_0single': list(range(10)), 'monomer_0_0single_T1': [3, 1, 0, 2] + list(range(4, 10))}
    for v, order in orders.items():
        (inp/f'{v}.lt').write_text(unit_lt(v, order))
    (inp/'system.lt').write_text('import "oplsaa.lt"\nimport "monomer_0_0single.lt"\n\n'
                                 'molecule_1 = new monomer_0_0single.rot(10,0,0,1).move(0,0,0)\n'
                                 'molecule_2 = new monomer_0_0single_T1.rot(20,0,1,0).move(5,0,0)\n')
    mass = {1: 12.011, 2: 12.011, 3: 15.999, 4: 1.008}
    lines, bl, k = [], [], 0
    for m, order in enumerate(orders.values(), 1):
        name_to_id = {}
        for i in order:
            k += 1
            name, t, q = ATOMS[i]
            name_to_id[name] = k
            lines.append(f'{k} {m} {t} {q} {k * 0.1:.2f} 0.0 0.0')
        bl += [(name_to_id[a], name_to_id[b]) for a, b in BONDS]
    data = ['test', '', f'{k} atoms', f'{len(bl)} bonds', '4 atom types', '1 bond types', '',
            '0 20 xlo xhi', '0 20 ylo yhi', '0 20 zlo zhi', '', 'Masses', ''] + [f'{t} {m}' for t, m in mass.items()]
    data += ['', 'Atoms  # full', ''] + lines + ['', 'Bonds', ''] + [f'{i} 1 {a} {b}' for i, (a, b) in enumerate(bl, 1)]
    (d/'system.neutral.data').write_text('\n'.join(data) + '\n')
    return d


def test_acetone_three_beads_from_the_library(tmp_path):
    lib = json.loads((ROOT/'benchmarks'/'ascg_library.json').read_text())
    recipes, mappings = library_inputs('acetone', lib, {})
    assert recipes == {'units': {'MOL0': {'single': 'CC(=O)C'}}}
    d = acetone_system(tmp_path/'acetone')
    topo = build_cg_topology('acetone', read_layout(d), read_data(d/'system.neutral.data'), recipes, mappings)
    assert topo.bead_types == ['ACT_METHYL', 'ACT_CARBONYL', 'ACT_METHYL'] * 2
    assert np.allclose(topo.bead_masses_amu, [15.035, 28.010, 15.035] * 2)
    assert topo.bonds == [(0, 1), (1, 2), (3, 4), (4, 5)] and topo.angles == [(0, 1, 2), (3, 4, 5)]
    assert list(topo.bead_molecule) == [0, 0, 0, 1, 1, 1]
    # every H follows its methyl carbon; the carbonyl bead is exactly C2 + O4 in both molecules
    assert [int((topo.atom_bead == b).sum()) for b in range(6)] == [4, 2, 4] * 2
    t = topo.to_dict()
    assert t['bond_types'] == ['ACT_CARBONYL-ACT_METHYL'] and t['angle_types'] == ['ACT_METHYL-ACT_CARBONYL-ACT_METHYL']


def test_bad_partitions_and_polymers_are_refused():
    site = lambda t, idx: {'bead_type': t, 'heavy_atom_indices_1based': idx}
    with pytest.raises(MappingError, match='partition'):
        molecular_inputs('x', [{'smiles': 'CC(=O)C', 'bead_sites': [site('A', [1, 2]), site('B', [2, 3, 4])]}])
    with pytest.raises(MappingError, match='partition'):
        molecular_inputs('x', [{'smiles': 'CC(=O)C', 'bead_sites': [site('A', [1, 2])]}])
    with pytest.raises(MappingError, match='attachment'):
        molecular_inputs('x', [{'smiles': '[*]CC[*]', 'bead_sites': [site('A', [1, 2])]}])


def test_mixture_components_resolve_to_their_species():
    lib = json.loads((ROOT/'benchmarks'/'ascg_library.json').read_text())
    recipes, mappings = library_inputs('n_hexane_toluene_mixture', lib, {})
    assert len(recipes['units']) == 2
    types = {b for groups in mappings['tasks']['n_hexane_toluene_mixture']['units'].values() for b, _ in groups}
    assert types == {'HEX_TERMINAL', 'HEX_MIDDLE', 'TOL_METHYL', 'TOL_RING'}


def test_molecules_without_angles_or_bonds():
    """Two-bead molecules (toluene, ethanol, acetonitrile with the proposed mappings) have no angles; the mapping
    helpers and the fit schedule must cope (the 27 polymers all had angles)."""
    from types import SimpleNamespace
    from ascg_core.ibi.runner import _schedule, present_families, whole_cycles
    from ascg_core.mapping.cg_mapping import angle_values, bond_lengths
    pos = np.zeros((4, 3))
    assert angle_values(pos, []).shape == (0,) and bond_lengths(pos, []).shape == (0,)
    two_bead = SimpleNamespace(bond_types=['A-B'], angle_types=[])
    fam = present_families(two_bead, None)
    assert fam == ('pair', 'bond')
    assert _schedule('pair:3,bond:1,angle:1,dihedral:1', 8, fam) == ['pair', 'pair', 'pair', 'bond'] * 2
    assert whole_cycles('pair:3,bond:1,angle:1,dihedral:1', 42, fam) == 44
    three_bead = SimpleNamespace(bond_types=['A-B'], angle_types=['A-B-A'])
    assert present_families(three_bead, None) == ('pair', 'bond', 'angle')     # as before for systems with angles



def test_polymer_repeat_unit_mapping_on_recipe_units():
    """Polymer mappings in rdkit-heavy-atom-indices form (repeat unit with [*]) become per-position groups of the build
    recipe's units: chain direction kept, end groups join the nearest bead, missing repeat-unit atoms are left out."""
    lib = json.loads((ROOT/'benchmarks'/'ascg_library.json').read_text())
    rec = json.loads((ROOT/'benchmarks'/'aa_recipes.json').read_text())
    units = lambda k: library_inputs(k, lib, rec)[1]['tasks'][k]['units']
    pmma = units('pmma')['MMA']
    assert pmma['middle'] == [['PMMA_BACKBONE', [0, 1, 2]], ['PMMA_ESTER', [3, 4, 5, 6]]]
    assert pmma['first'] == pmma['middle'] == pmma['last']                  # H-capped ends: same heavy atoms
    pdms = units('pdms')['DMS']
    assert pdms['first'] == [['PDMS_REPEAT', [0, 1, 2, 3, 4]]]               # Me3Si-O start: extra methyl joins
    assert pdms['last'] == [['PDMS_REPEAT', [0, 1, 2, 3]]]                   # -SiMe3 end: no oxygen, extra methyl
    pet = units('pet')['ETT']
    assert pet['last'][3] == ['PET_ESTER', [12, 13, 14, 15]]                 # methyl ester end joins the last ester
    bpc = units('bpa_pc')['BPC']
    assert bpc['first'][0] == ['BPA_PC_CARBONATE', [0, 1, 2, 3, 4]]         # methyl carbonate start
    assert [len(i) for _, i in bpc['middle']] == [4, 6, 3, 6]
    from ascg_core.mapping.formats import polymer_inputs
    bad = {'smiles': '[*]CC(C)(C(=O)OC)[*]', 'bead_sites': [{'bead_type': 'A', 'heavy_atom_indices_1based': [1, 2]}]}
    with pytest.raises(MappingError, match='attachment point'):
        polymer_inputs('pmma', bad, rec)
    bad['bead_sites'] = [{'bead_type': 'A', 'heavy_atom_indices_1based': [2, 3, 4]}]
    with pytest.raises(MappingError, match='partition'):
        polymer_inputs('pmma', bad, rec)
