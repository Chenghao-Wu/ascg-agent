"""Bring-your-own AA references (ascg_core.mapping.byo; ascg ref inspect / build): the same molecules and frames give
the same mapped reference from a GROMACS topology with a residue-atom-names mapping and from a LAMMPS data file with
a SMILES mapping."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('MDAnalysis')
pytest.importorskip('rdkit')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore')

from byo_system import build, write_gromacs, write_lammps  # noqa: E402
from ascg_core.mapping import byo  # noqa: E402
from ascg_core.mapping.cg_mapping import MappingError  # noqa: E402

SMILES = {'format': 'annotated-smiles', 'smiles': '[CH3:1][CH2:1][CH2:2][CH2:3][CH3:3]',
          'bead_types': {'1': 'END', '2': 'MID', '3': 'END'}}
RESIDUES = {'format': 'residue-atom-names', 'residues': {'PEN': [{'type': 'END', 'atoms': ['C1', 'C2']},
                                                                 {'type': 'MID', 'atoms': ['C3']},
                                                                 {'type': 'END', 'atoms': ['C4', 'C5']}]}}


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp('byo')
    s = build(n_frames=8)
    write_lammps(s, d/'lmp')
    write_gromacs(s, d/'gmx')
    s = build(n_frames=8, npt=True)
    write_gromacs(s, d/'npt')
    return d


def test_both_front_ends_give_the_same_reference(data, tmp_path):
    a = byo.prepare_byo_reference('pentane', data/'lmp/system.data',
                                  {'300': {'trajectory': str(data/'lmp/traj.lammpstrj'), 'energy': str(data/'lmp/log.lammps')}},
                                  SMILES, tmp_path/'a', r_max_nm=1.2)
    b = byo.prepare_byo_reference('pentane', data/'gmx/system.top',
                                  {'300': {'trajectory': str(data/'gmx/traj.xtc'), 'energy': str(data/'gmx/energy.xvg')}},
                                  RESIDUES, tmp_path/'b', r_max_nm=1.2)
    ma, mb = np.load(tmp_path/'a/mapped_T300.npz'), np.load(tmp_path/'b/mapped_T300.npz')
    assert np.abs(ma['pos_nm'] - mb['pos_nm']).max() < 1e-3                 # xtc precision
    assert json.loads((tmp_path/'a/topology.json').read_text()) == json.loads((tmp_path/'b/topology.json').read_text())
    sa, sb = a['states']['300'], b['states']['300']
    for k in ('density_g_cm3', 'temperature_K_mean', 'pressure_bar_mean', 'frames'):
        assert sa[k] == pytest.approx(sb[k], rel=1e-6)
    topo = json.loads((tmp_path/'a/topology.json').read_text())
    assert topo['bead_masses_amu'][:3] == pytest.approx([29.062, 14.027, 29.062], abs=1e-3)
    assert topo['bond_types'] == ['END-MID'] and topo['angle_types'] == ['END-MID-END']
    assert a['source']['kind'] == 'bring-your-own' and set(np.load(tmp_path/'a/targets_T300.npz').files) >= {
        'rdf/END-END', 'rdf/END-MID', 'rdf/MID-MID', 'bond/END-MID', 'angle/END-MID-END', 'density_g_cm3'}
    for f in ('topology.json', 'summary.json', 'start_T300.npz'):
        assert (tmp_path/'a'/f).exists()


def test_molecules_are_made_whole(data, tmp_path):
    byo.prepare_byo_reference('pentane', data/'gmx/system.top', {'300': {'trajectory': str(data/'gmx/traj.xtc')}},
                              RESIDUES, tmp_path/'w', r_max_nm=1.2)
    pos = np.load(tmp_path/'w/mapped_T300.npz')['pos_nm']
    d = np.linalg.norm(pos[:, 1::3] - pos[:, 0::3], axis=-1)                # END-MID distances: no box jumps
    assert d.max() < 0.35


def test_no_energy_file_and_npt_compressibility(data, tmp_path):
    s = byo.prepare_byo_reference('pentane', data/'npt/system.top', {'300': {'trajectory': str(data/'npt/traj.xtc')}},
                                  RESIDUES, tmp_path/'n', r_max_nm=1.2)['states']['300']
    assert s['ensemble'] == 'NPT' and s['kappa_T_per_bar'] > 0
    assert s['pressure_bar_mean'] is None and s['temperature_K_mean'] is None


def test_lammps_times_and_start_ps(data, tmp_path):
    r = byo.inspect(data/'lmp/system.data', data/'lmp/traj.lammpstrj', data/'lmp/log.lammps')
    assert r['trajectory']['time_ps'] == [0.0, 7.0] and r['trajectory']['frame_spacing_ps'] == 1.0
    r = byo.inspect(data/'lmp/system.data', data/'lmp/traj.lammpstrj')
    assert r['trajectory']['time_ps'] is None and any('step numbers' in w for w in r['warnings'])
    with pytest.raises(MappingError, match='step numbers'):
        byo.prepare_byo_reference('p', data/'lmp/system.data', {'300': {'trajectory': str(data/'lmp/traj.lammpstrj')}},
                                  SMILES, tmp_path/'x', start_ps=2.0, r_max_nm=1.2)
    s = byo.prepare_byo_reference('p', data/'lmp/system.data',
                                  {'300': {'trajectory': str(data/'lmp/traj.lammpstrj'), 'timestep_fs': 1.0}},
                                  SMILES, tmp_path/'y', start_ps=2.0, r_max_nm=1.2)
    assert s['states']['300']['frames'] == 6                                  # frames at 2..7 ps


def test_inspect_reports_residues_and_the_mapping(data):
    r = byo.inspect(data/'gmx/system.top', data/'gmx/traj.xtc', data/'gmx/energy.xvg', RESIDUES)
    assert r['molecules'] == {'count': 27, 'by_formula': {'C5H12': 27}}
    assert r['residues']['PEN']['count'] == 27 and r['residues']['PEN']['atoms'][:5] == ['C1', 'C2', 'C3', 'C4', 'C5']
    assert r['trajectory']['ensemble_guess'] == 'NVT' and r['energy']['temperature_K_mean'] == pytest.approx(303.5)
    m = r['mapping']
    assert m['molecule_types'] == {'END MID END': {'count': 27, 'beads': 3, 'bead_masses_amu': [29.062, 14.027, 29.062]}}
    assert m['bond_types'] == ['END-MID'] and m['dihedral_types'] == []


@pytest.mark.parametrize('mapping, message', [
    ({'format': 'residue-atom-names', 'residues': {'XYZ': [{'type': 'A', 'atoms': ['C1']}]}}, 'without a mapping entry'),
    ({'format': 'residue-atom-names', 'residues': {'PEN': [{'type': 'A', 'atoms': ['C1', 'C9']}]}}, "no atom named 'C9'"),
    ({'format': 'residue-atom-names', 'residues': {'P*': [{'type': 'A', 'atoms': ['C1', 'C2', 'C3']}]}}, 'in no bead'),
    ({'format': 'annotated-smiles', 'smiles': '*[CH2:1][CH2:1]*', 'bead_types': {'1': 'A'}}, 'polymer repeat unit'),
    ({'format': 'annotated-smiles', 'smiles': '[CH3:1][CH2:1][CH3:2]', 'bead_types': {'1': 'A', '2': 'B'}}, 'matches no species'),
])
def test_mapping_errors(data, mapping, message):
    with pytest.raises(MappingError, match=message):
        byo.inspect(data/'gmx/system.top', None, None, mapping)


def test_disconnected_bead_is_flagged(data):
    bad = {'format': 'residue-atom-names', 'residues': {'PEN': [{'type': 'A', 'atoms': ['C1', 'C3']},
                                                               {'type': 'B', 'atoms': ['C2']},
                                                               {'type': 'A', 'atoms': ['C4', 'C5']}]}}
    r = byo.inspect(data/'gmx/system.top', None, None, bad)
    assert any('not bonded to each other' in w for w in r['warnings'])
    assert not any('not bonded' in w for w in byo.inspect(data/'gmx/system.top', None, None, RESIDUES).get('warnings', []))


def test_lammps_data_has_no_names(data):
    with pytest.raises(MappingError, match='no residue or atom names'):
        byo.inspect(data/'lmp/system.data', None, None, RESIDUES)


def test_topology_without_bonds_is_refused(data, tmp_path):
    import MDAnalysis as mda
    u = mda.Universe(str(data/'gmx/system.top'), str(data/'gmx/traj.xtc'), topology_format='ITP', infer_system=True)
    u.atoms.write(str(tmp_path/'conf.gro'))
    with pytest.raises(MappingError, match='no bonds'):
        byo.inspect(tmp_path/'conf.gro', None)


def test_ref_tools_flow_on_a_local_target(data, tmp_path, monkeypatch, capsys):
    """map save (residue form) -> map approve -> ref inspect -> ref build -> aa import, through the CLI."""
    from ascg.cli import main as cli
    monkeypatch.setenv('ASCG_STUDIES', str(tmp_path/'studies'))
    prof = tmp_path/'compute.json'
    prof.write_text(json.dumps({'targets': {'here': {'transport': 'local', 'scheduler': 'none', 'root': str(ROOT),
                                                       'python': sys.executable, 'work_root': str(tmp_path/'work'),
                                                       'gpus': [0]}}}))
    monkeypatch.setenv('ASCG_COMPUTE', str(prof))
    monkeypatch.setenv('PYTHONPATH', str(ROOT/'src'))

    def run(args):
        rc = cli(args)
        return rc, json.loads(capsys.readouterr().out)
    run(['study', 'init', '--study', 'b1'])
    rc, out = run(['map', 'save', '--study', 'b1', '--key', 'pentane', '--spec', json.dumps(RESIDUES),
                   '--system-class', 'liquid'])
    assert rc == 0, out
    rc, out = run(['ref', 'build', '--study', 'b1', '--key', 'pentane', '--target', 'here',
                   '--topology', str(data/'gmx/system.top'), '--run', f'300={data/"gmx/traj.xtc"},{data/"gmx/energy.xvg"}'])
    assert rc != 0 and out.get('refused_by_policy') and 'approve' in out['error']
    run(['map', 'approve', '--study', 'b1', '--key', 'pentane', '--note', 'two types, as proposed'])
    rc, out = run(['ref', 'inspect', '--target', 'here', '--topology', str(data/'gmx/system.top'),
                   '--trajectory', str(data/'gmx/traj.xtc'), '--study', 'b1', '--key', 'pentane'])
    assert rc == 0 and out['result']['mapping']['bead_types'] == ['END', 'MID']
    ref = tmp_path/'refs'/'pentane'
    rc, out = run(['ref', 'build', '--study', 'b1', '--key', 'pentane', '--target', 'here',
                   '--topology', str(data/'gmx/system.top'), '--out', str(ref), '--r-max-nm', '1.2',
                   '--run', f'300={data/"gmx/traj.xtc"},{data/"gmx/energy.xvg"}'])
    assert rc == 0, out
    assert out['result']['states']['300']['frames'] == 8 and (ref/'mapping.json').exists()
    rc, out = run(['aa', 'import', '--study', 'b1', '--task', 'pentane', '--target', 'here', '--ref', str(ref),
                   '--states', '300'])
    assert rc == 0, out
    reg = json.loads((tmp_path/'studies'/'b1'/'registry.json').read_text())
    assert reg['tasks']['pentane']['mapping']['status'] == 'approved'
