"""Mapping specifications in SMILES form: map check / save / approve and the converters map-aa uses."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip('rdkit')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))

from ascg.cli import main as cli  # noqa: E402
from ascg_core.mapping.spec import check, normalise_species  # noqa: E402

PENTANE = {'format': 'annotated-smiles', 'smiles': '[CH3:1][CH2:1][CH2:2][CH2:3][CH3:3]',
           'bead_types': {'1': 'END', '2': 'MID', '3': 'END'}}
PMA = {'format': 'annotated-smiles',                     # poly(methyl acrylate): backbone + ester
       'smiles': '*[CH2:1][CH:1](*)[C:2](=[O:2])[O:2][CH3:2]', 'bead_types': {'1': 'BB', '2': 'ES'}}


def run_cli(args, capsys):
    rc = cli(args)
    return rc, json.loads(capsys.readouterr().out)


def test_annotated_smiles_converts_to_indices_and_reports_masses():
    sp = normalise_species(PENTANE)
    assert sp['smiles'] == '[CH3][CH2][CH2][CH2][CH3]'
    assert [(s['bead_type'], s['heavy_atom_indices_1based']) for s in sp['bead_sites']] == \
        [('END', [1, 2]), ('MID', [3]), ('END', [4, 5])]
    r = check(PENTANE, 'liquid')
    assert r['ok'] and r['bead_types'] == ['END', 'MID']
    s = r['species'][0]
    assert [b['mass_amu'] for b in s['beads']] == pytest.approx([29.062, 14.027, 29.062], abs=0.01)
    assert s['mass_amu'] == pytest.approx(72.15, abs=0.01)
    assert s['bonds'] == ['END-MID'] and s['angles'] == ['END-MID-END'] and s['dihedrals'] == []
    assert any('1 heavy atom' in w for w in r['warnings'])               # the one-carbon middle bead


def test_polymer_repeat_unit_links_head_and_tail():
    r = check(PMA, 'polymer')
    assert r['ok'], r['errors']
    s = r['species'][0]
    assert s['kind'] == 'repeat_unit' and s['beads_per_unit'] == 2
    assert [b['mass_amu'] for b in s['beads']] == pytest.approx([27.046, 59.044], abs=0.01)
    assert s['bonds'] == ['BB-BB', 'BB-ES']
    assert s['angles'] == ['BB-BB-BB', 'BB-BB-ES']
    assert s['dihedrals'] == ['BB-BB-BB-BB', 'BB-BB-BB-ES', 'ES-BB-BB-ES']


@pytest.mark.parametrize('spec, cls, message', [
    ({'format': 'annotated-smiles', 'smiles': '[C:1][C:1][C:2]', 'bead_types': {'1': 'A', '2': 'B'}}, None,
     'unpaired electrons'),
    ({'format': 'rdkit-heavy-atom-indices', 'smiles': 'CCCC',
      'bead_sites': [{'bead_type': 'A', 'heavy_atom_indices_1based': [1, 2]}]}, None, 'in no bead'),
    ({'format': 'rdkit-heavy-atom-indices', 'smiles': 'CCCC',
      'bead_sites': [{'bead_type': 'A', 'heavy_atom_indices_1based': [1, 2, 3]},
                     {'bead_type': 'B', 'heavy_atom_indices_1based': [3, 4]}]}, None, 'in two beads'),
    ({'format': 'annotated-smiles', 'smiles': '[*:1][CH2:1][CH2:1]*', 'bead_types': {'1': 'A'}}, None,
     'belongs to no bead'),
    ({'format': 'annotated-smiles', 'smiles': '[CH3:1][CH3:2]', 'bead_types': {'1': 'A'}}, None, 'no entry'),
    (PENTANE, 'polymer', 'two attachment points'),
    (PMA, 'liquid', 'whole molecule'),
])
def test_bad_mappings_are_errors_not_exceptions(spec, cls, message):
    r = check(spec, cls)
    assert not r['ok'] and any(message in e for e in r['errors']), r['errors']


def test_more_bead_types_than_fastmd_allows():
    smiles = ''.join(f'[CH2:{i}]' for i in range(1, 18))
    smiles = '[CH3:1]' + smiles[7:-8] + '[CH3:17]'
    r = check({'format': 'annotated-smiles', 'smiles': smiles, 'bead_types': {str(i): f'T{i}' for i in range(1, 18)}})
    assert not r['ok'] and any('fastMD allows 16' in e for e in r['errors'])


def test_disconnected_bead_is_a_warning():
    r = check({'format': 'annotated-smiles', 'smiles': '[CH3:1][S:2]([CH3:1])=[O:2]', 'bead_types': {'1': 'M', '2': 'S'}})
    assert r['ok'] and any('not bonded to each other' in w for w in r['warnings'])


def test_converters_accept_annotated_smiles():
    from ascg_core.mapping.formats import molecular_inputs, polymer_inputs
    _, m1 = molecular_inputs('x', [PENTANE])
    _, m2 = molecular_inputs('x', [normalise_species(PENTANE)])
    assert m1 == m2 == {'tasks': {'x': {'units': {'MOL0': [['END', [0, 1]], ['MID', [2]], ['END', [3, 4]]]}}}}
    rec = {'tasks': {'x': {'sequence': 'U*10'}},
           'units': {'U': {'first': 'CC(*)C(=O)OC', 'middle': '*CC(*)C(=O)OC', 'last': '*CCC(=O)OC'}}}
    _, p = polymer_inputs('x', PMA, rec)
    assert p['tasks']['x']['units']['U']['middle'] == [['BB', [0, 1]], ['ES', [2, 3, 4, 5]]]


def test_smiles_mappings_of_polymers_are_supported():
    from ascg.tools.reference_tools import _mapping_supported
    for fmt in ('rdkit-heavy-atom-indices', 'annotated-smiles'):
        for cls in ('polymer', 'liquid', 'liquid_mixture'):
            assert _mapping_supported({'system_class': cls, 'mapping': {'format': fmt}})
    assert not _mapping_supported({'system_class': 'polymer', 'mapping': {'format': 'other'}})


def test_map_check_save_show_approve(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('ASCG_STUDIES', str(tmp_path/'studies'))
    monkeypatch.setenv('ASCG_ACTOR', 'tester')
    run_cli(['study', 'init', '--study', 'm1'], capsys)
    rc, out = run_cli(['map', 'check', '--spec', json.dumps(PMA), '--system-class', 'polymer'], capsys)
    assert rc == 0 and out['result']['ok']
    bad = dict(PENTANE, smiles='[CH3:1][CH2:1][C:2][CH2:3][CH3:3]')
    rc, out = run_cli(['map', 'save', '--study', 'm1', '--key', 'pentane', '--spec', json.dumps(bad),
                       '--system-class', 'liquid'], capsys)
    assert rc != 0 and out.get('refused_by_policy')
    spec_file = tmp_path/'pentane.json'
    spec_file.write_text(json.dumps(PENTANE))
    rc, out = run_cli(['map', 'save', '--study', 'm1', '--key', 'pentane', '--spec', str(spec_file),
                       '--system-class', 'liquid', '--rationale', 'ends vs middle'], capsys)
    assert rc == 0 and out['result']['status'] == 'proposed'
    rc, out = run_cli(['map', 'show', '--task', 'pentane', '--study', 'm1'], capsys)
    assert out['result']['status'] == 'proposed' and 'approval' in out['result']
    rc, out = run_cli(['map', 'approve', '--study', 'm1', '--key', 'pentane', '--note', 'yes, two types'], capsys)
    assert out['result']['status'] == 'approved'
    rec = json.loads((tmp_path/'studies'/'m1'/'mappings'/'pentane.json').read_text())
    assert rec['approved_by'] == 'tester' and rec['normalised']['bead_sites'][1]['heavy_atom_indices_1based'] == [3]
    events = [json.loads(x)['event'] for x in (tmp_path/'studies'/'m1'/'events.jsonl').read_text().splitlines()]
    assert events[-2:] == ['mapping_proposed', 'mapping_approved']
    rc, out = run_cli(['map', 'save', '--study', 'm1', '--key', 'pentane', '--spec', json.dumps(PENTANE),
                       '--system-class', 'liquid'], capsys)                       # a new proposal keeps the history
    rec = json.loads((tmp_path/'studies'/'m1'/'mappings'/'pentane.json').read_text())
    assert rec['status'] == 'proposed' and rec['history'][-1]['status'] == 'approved'
