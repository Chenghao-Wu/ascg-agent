"""AA->CG mapping on a synthetic AutoPoly-style system (2 PS chains x 3 units)."""
import json
from pathlib import Path

import numpy as np
import pytest

Chem = pytest.importorskip('rdkit.Chem')
from ascg_core.mapping.cg_mapping import (build_cg_topology, frame_rdf_counts,  # noqa: E402
                                      intermolecular_pair_counts, map_frame, read_data,
                                      read_layout, read_lammps_dump, block_se)
from ascg_core.mapping.cg_reference import prepare_reference  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]/'benchmarks'


def paper_mappings():
    """Repeat-unit groups of the benchmark's paper polymers, from the system library."""
    lib = json.loads((ROOT/'ascg_library.json').read_text())
    return {'tasks': {s['key']: {'units': s['mapping']['units']} for s in lib['systems']
                      if (s.get('mapping') or {}).get('format') == 'ascg-repeat-unit-groups'}}
MASS = {'C': 12.011, 'H': 1.008}


def unit_atoms(smiles):
    """AutoPoly .lt order: head connection atom, tail connection atom, other heavy atoms
    in SMILES order, then H; plus bonds and the [*] partner indices."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 0]
    heavy = [i for i in keep if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    joints = [a.GetNeighbors()[0].GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    head = joints[0] if smiles.startswith('[*]') else heavy[0]
    tail = joints[-1] if smiles.endswith('[*]') or smiles.count('[*]') == 2 else None
    front = [head] + ([tail] if tail is not None and tail != head else [])
    heavy = front + [i for i in heavy if i not in front]
    hyd = [i for i in keep if mol.GetAtomWithIdx(i).GetAtomicNum() == 1]
    order = heavy + hyd
    new = {old: k for k, old in enumerate(order)}
    names = [f'{mol.GetAtomWithIdx(i).GetSymbol()}{k + 1}' for k, i in enumerate(order)]
    bonds = [(new[b.GetBeginAtomIdx()], new[b.GetEndAtomIdx()]) for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in new and b.GetEndAtomIdx() in new]
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    joints = [new[a.GetNeighbors()[0].GetIdx()] for a in dummies]
    return names, bonds, joints


def write_system(tmp, n_chains=2):
    recipes = json.loads((ROOT/'aa_recipes.json').read_text())
    s = recipes['units']['S']
    variants = {'monomer_0_0le': s['first'], 'monomer_0_1i': s['middle'], 'monomer_0_2re': s['last']}
    inp = tmp/'moltemplate'/'input'
    inp.mkdir(parents=True)
    units = {}
    for v, smi in variants.items():
        names, bonds, joints = unit_atoms(smi)
        units[v] = (names, bonds, joints)
        atoms = '\n'.join(f'\t$atom:{n} $mol:... @atom:1 0.0 0 0 0' for n in names)
        blist = '\n'.join(f'\t$bond:b{k} $atom:{names[a]} $atom:{names[b]}' for k, (a, b) in enumerate(bonds))
        (inp/f'{v}.lt').write_text(f'{v} inherits OPLSAA {{\n  write("Data Atoms") {{\n{atoms}\n  }}\n'
                                   f"  write('Data Bond List') {{\n{blist}\n  }}\n}}\n")
    order = list(variants)
    for c in range(1, n_chains + 1):
        mons = '\n'.join(f'    monomer[{i}] = new {v}.rot(0,1,0,0).move(0,0,0)' for i, v in enumerate(order))
        (inp/f'poly_{c}.lt').write_text(f'poly_{c} inherits OPLSAA {{\n{mons}\n}}\n')
    (inp/'system.lt').write_text('\n'.join(f'polymer_{c} = new poly_{c}.rot(0,1,0,0)' for c in range(1, n_chains + 1)) + '\n')
    # data file in moltemplate numbering
    elements, mol, bonds = [], [], []
    for c in range(n_chains):
        prev_tail = None
        for v in order:
            names, ub, joints = units[v]
            off = len(elements)
            elements += [n.rstrip('0123456789') for n in names]
            mol += [c + 1] * len(names)
            bonds += [(off + a, off + b) for a, b in ub]
            if prev_tail is not None:
                bonds.append((prev_tail, off + joints[0]))
            prev_tail = off + joints[-1]
    rng = np.random.default_rng(1)
    xyz = rng.uniform(0, 40, size=(len(elements), 3))
    types = {'C': 1, 'H': 2}
    lines = ['test', '', f'{len(elements)} atoms', f'{len(bonds)} bonds', '2 atom types', '1 bond types', '',
             '0 40 xlo xhi', '0 40 ylo yhi', '0 40 zlo zhi', '', 'Masses', '', '1 12.011 # 57_bCT', '2 1.008 # 60_bHC',
             '', 'Atoms # full', '']
    lines += [f'{i + 1} {mol[i]} {types[e]} 0.0 {x:.4f} {y:.4f} {z:.4f}' for i, (e, (x, y, z)) in enumerate(zip(elements, xyz))]
    lines += ['', 'Bonds', ''] + [f'{k + 1} 1 {a + 1} {b + 1}' for k, (a, b) in enumerate(bonds)]
    (tmp/'system.neutral.data').write_text('\n'.join(lines) + '\n')
    return recipes, elements, xyz


def test_layout_topology_and_com(tmp_path):
    recipes, elements, xyz = write_system(tmp_path)
    mappings = paper_mappings()
    data = read_data(tmp_path/'system.neutral.data')
    topo = build_cg_topology('ps', read_layout(tmp_path), data, recipes, mappings)
    assert topo.bead_types == ['A', 'B'] * 6 and topo.type_names == ['A', 'B']
    # middle-unit backbone C2H3 and phenyl C6H5; end units carry one more H on the backbone
    assert topo.bead_masses_amu[2] == pytest.approx(2 * 12.011 + 3 * 1.008)
    assert topo.bead_masses_amu[3] == pytest.approx(6 * 12.011 + 5 * 1.008)
    assert topo.bead_masses_amu[0] == pytest.approx(2 * 12.011 + 4 * 1.008)
    assert len(topo.bonds) == 10 and len(topo.angles) == 10 and len(topo.exclusions) == 20
    assert (0, 2) in topo.bonds and (0, 1) in topo.bonds and (1, 2) not in topo.bonds
    assert topo.bond_type((0, 2)) == 'A-A' and topo.angle_type((1, 0, 2)) == 'A-A-B'
    com = map_frame(data['xyz_A'], data['mass'], topo.atom_bead, topo.bead_masses_amu)
    sel = topo.atom_bead == 3
    expect = (data['mass'][sel, None] * data['xyz_A'][sel]).sum(0) / data['mass'][sel].sum() / 10
    assert com[3] == pytest.approx(expect)


def test_layout_mismatch_is_rejected(tmp_path):
    recipes, _, _ = write_system(tmp_path)
    mappings = paper_mappings()
    data = read_data(tmp_path/'system.neutral.data')
    data['element'][5] = 'O'
    with pytest.raises(ValueError, match='Element mismatch'):
        build_cg_topology('ps', read_layout(tmp_path), data, recipes, mappings)


def test_intermolecular_rdf_of_ideal_gas_is_one():
    rng = np.random.default_rng(0)
    n, box = 3000, np.array([6.0, 6.0, 6.0])
    types = ['A', 'B'] * (n // 2)
    mol = np.repeat(np.arange(n // 10), 10)
    edges = np.arange(0, 2.01, 0.25)
    shell = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    pairs = intermolecular_pair_counts(types, mol, ['A', 'B'])
    acc = {k: 0 for k in pairs}
    for _ in range(5):
        for k, c in frame_rdf_counts(rng.uniform(0, 6, (n, 3)), box, types, mol, ['A', 'B'], edges).items():
            acc[k] = acc[k] + c
    for k in pairs:
        g = acc[k] / (5 * pairs[k] * shell / np.prod(box))
        assert np.allclose(g[1:], 1.0, atol=0.05), (k, g)


def test_block_se():
    # One 10-block estimate scatters ~24 %; its mean over many series must match 1/sqrt(n)
    # (times c4(10) = 0.973, the small-sample bias of a standard deviation).
    rng = np.random.default_rng(3)
    est = [block_se(rng.normal(0, 1, 2000)) for _ in range(400)]
    assert np.mean(est) == pytest.approx(0.973 / np.sqrt(2000), rel=0.05)
    # correlated data: SE must exceed the naive iid value
    x = np.repeat(rng.normal(0, 1, 200), 10)
    assert block_se(x) > 1.5 * x.std() / np.sqrt(len(x))


def test_prepare_reference_end_to_end(tmp_path):
    recipes, elements, xyz = write_system(tmp_path/'sys', n_chains=20)
    mappings = paper_mappings()
    data = read_data(tmp_path/'sys'/'system.neutral.data')
    rng = np.random.default_rng(2)
    for T in (400, 500):
        run = tmp_path/f'T{T}'
        run.mkdir()
        frames = []
        for f in range(20):
            box = 40.0 + 0.1 * f
            x = data['xyz_A'] + rng.normal(0, 0.05, data['xyz_A'].shape)
            body = '\n'.join(f'{i + 1} {a:.4f} {b:.4f} {c:.4f}' for i, (a, b, c) in enumerate(x))
            frames.append(f'ITEM: TIMESTEP\n{1000 * f}\nITEM: NUMBER OF ATOMS\n{len(x)}\n'
                          f'ITEM: BOX BOUNDS pp pp pp\n0 {box}\n0 {box}\n0 {box}\nITEM: ATOMS id xu yu zu\n{body}\n')
        (run/'aa_production.lammpstrj').write_text(''.join(frames))
        rows = '\n'.join(f'{1000 * f} {T} {1 + f % 3} 0.9 1 1 64000 40 40 40' for f in range(20))
        (run/'log.lammps').write_text('Step Temp Press Density PotEng TotEng Volume Lx Ly Lz\n' + rows +
                                      '\nLoop time of 1 on 1 procs\n')
    assert len(list(read_lammps_dump(tmp_path/'T400'/'aa_production.lammpstrj', len(data['mass'])))) == 20
    s = prepare_reference('ps', tmp_path/'sys', {400: tmp_path/'T400', 500: tmp_path/'T500'}, tmp_path/'out',
                          recipes=recipes, mappings=mappings, r_max_nm=1.5)
    assert s['n_beads'] == 120 and s['bond_types'] == ['A-A', 'A-B']
    t = np.load(tmp_path/'out'/'targets_T400.npz')
    edges = t['bond_edges/A-B']
    assert np.sum(t['bond/A-B'] * np.diff(edges)) == pytest.approx(1.0)
    assert np.sum(t['angle/A-A-B'] * np.diff(np.deg2rad(np.arange(0, 181, 2)))) == pytest.approx(1.0)
    start = np.load(tmp_path/'out'/'start_T500.npz')
    assert start['pos_nm'].shape == (120, 3)
    assert s['states']['500']['pressure_bar_mean'] == pytest.approx(np.mean([1 + f % 3 for f in range(20)]) * 1.01325)
    with pytest.raises(ValueError, match='half the smallest box'):
        prepare_reference('ps', tmp_path/'sys', {400: tmp_path/'T400'}, tmp_path/'out2',
                          recipes=recipes, mappings=mappings, r_max_nm=2.5)
