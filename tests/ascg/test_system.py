"""System form (ascg system check / save / approve; ascg_core.aa.system_spec): one SMILES per species gives the AA
build recipe and, with bead numbers, the mapping. Checked against the hand-written benchmark recipes."""
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip('rdkit')
from rdkit import Chem  # noqa: E402

from ascg.study import Study  # noqa: E402
from ascg_core.aa.system_spec import cap_unit, derive  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RECIPES = json.loads((ROOT/'benchmarks'/'aa_recipes.json').read_text())     # the benchmark's 40 systems
P27 = EXT = RECIPES
canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))   # noqa: E731
MMA = '*[CH2:1][C:1](*)([CH3:1])[C:2](=[O:2])[O:2][CH3:2]'
# end units that are not plain H caps of the repeat unit (the benchmark's chemistry notes): methyl caps, or given
SPECIAL = {'VDF': {'end_groups': {'head': 'H', 'tail': 'CH3'}}, 'BPC': {'end_groups': {'head': 'CH3', 'tail': 'H'}},
           'DMS': {'end_groups': {'head': 'CH3', 'tail': 'H'}, 'last': EXT['units']['DMS']['last']},
           'LA': {'first': P27['units']['LA']['first'], 'last': P27['units']['LA']['last']},
           'CL': {'first': P27['units']['CL']['first'], 'last': P27['units']['CL']['last']},
           'ETT': {'last': EXT['units']['ETT']['last']}}


@pytest.mark.parametrize('code,units', sorted(RECIPES['units'].items()))
def test_end_units_of_every_benchmark_polymer(code, units):
    d = derive({'system_class': 'polymer', 'species': [{'name': code, 'smiles': units['middle']}], 'chains': 25,
                **SPECIAL.get(code, {})}, 'x')
    assert d['ok'], d['errors']
    got = d['recipe']['units'][code]
    assert all(canon(got[p]) == canon(units[p]) for p in ('first', 'middle', 'last')), (got, units)


@pytest.mark.parametrize('code,units', sorted(RECIPES['units'].items()))
def test_end_units_are_oriented_for_autopoly(code, units):
    """AutoPoly grows a chain from the start unit's atom 0 towards its connection atom: a start unit written from its
    attachment point ('[*]CC') would give a zero axis and MC chain growth fails (seen in a build)."""
    from ascg_core.aa.system_spec import end_unit_axis_ok
    for cap in ('H', 'CH3'):
        assert end_unit_axis_ok(cap_unit(units['middle'], 'tail', cap), 'first')
        assert end_unit_axis_ok(cap_unit(units['middle'], 'head', cap), 'last')
    d = derive({'system_class': 'polymer', 'species': [{'smiles': '*CC*'}], 'chains': 25, 'first': '[*]CC'}, 'x')
    assert not d['ok'] and 'capped end' in d['errors'][0]


def test_h_caps_flag_acyl_and_silyl_ends():
    d = derive({'system_class': 'polymer', 'species': [{'name': 'LA', 'smiles': '*OC(C)C(=O)*'}], 'chains': 25}, 'x')
    assert d['ok'] and any('aldehyde' in w for w in d['warnings'])
    assert canon(cap_unit('*CC(*)c1ccccc1', 'tail')) == canon('CC([*])c1ccccc1')


def test_sizes_match_the_benchmark():
    d = derive({'system_class': 'polymer', 'species': [{'name': 'MMA', 'smiles': MMA, 'bead_types': {'1': 'BB', '2': 'E'}}]},
               'pmma')
    assert d['ok'], d['errors']
    task = d['recipe']['tasks']['pmma']
    assert task['sequence'] == 'MMA*10' and task['chains'] == EXT['tasks']['pmma']['chains'] == 50
    assert d['size']['cg_beads'] == 1000 and d['mapping']['format'] == 'annotated-smiles'
    mix = derive({'system_class': 'liquid_mixture', 'species': [{'name': 'hexane', 'smiles': 'CCCCCC', 'fraction': 0.5},
                                                                {'name': 'toluene', 'smiles': 'Cc1ccccc1', 'fraction': 0.5}]},
                 'n_hexane_toluene_mixture')
    assert mix['recipe']['tasks']['n_hexane_toluene_mixture']['molecules'] == \
        EXT['tasks']['n_hexane_toluene_mixture']['molecules']
    hexane = derive({'system_class': 'liquid', 'species': [{'smiles': 'CCCCCC'}]}, 'n_hexane')
    assert hexane['recipe']['tasks']['n_hexane']['molecules'] == EXT['tasks']['n_hexane']['molecules']


def test_derived_files_feed_the_cg_topology_inputs():
    from ascg_core.mapping.formats import library_inputs
    d = derive({'system_class': 'polymer', 'species': [{'name': 'MMA', 'smiles': MMA, 'bead_types': {'1': 'BB', '2': 'E'}}]},
               'pmma')
    _, maps = library_inputs('pmma', d['library'], d['recipe'])
    units = maps['tasks']['pmma']['units']['MMA']
    assert [t for t, _ in units['middle']] == ['BB', 'E'] and sorted(len(a) for _, a in units['middle']) == [3, 4]
    mix = derive({'system_class': 'liquid_mixture', 'species': [
        {'name': 'hexane', 'smiles': '[CH3:1][CH2:1][CH2:2][CH2:2][CH2:3][CH3:3]', 'bead_types': {'1': 'T', '2': 'M', '3': 'T'},
         'count': 500},
        {'name': 'toluene', 'smiles': '[CH3:1][c:1]1[cH:2][cH:2][cH:3][cH:3][cH:1]1', 'bead_types': {'1': 'A', '2': 'B', '3': 'B'},
         'count': 500}]}, 'mix')
    assert mix['ok'], mix['errors']
    library_inputs('mix', mix['library'], mix['recipe'])


def test_errors_block():
    bad = [({'system_class': 'polymer', 'species': [{'smiles': '*CC*'}, {'smiles': '*CC(*)C'}]}, 'needs "sequence"'),
           ({'system_class': 'polymer', 'species': [{'name': 'A', 'smiles': '*CC*'}, {'name': 'B', 'smiles': '*CC(*)C'}],
             'sequence': 'A*5 C*5'}, 'not a species'),
           ({'system_class': 'polymer', 'species': [{'name': 'A', 'smiles': '*CC*'}, {'name': 'B', 'smiles': '*CC(*)C'}],
             'sequence': 'A*10'}, 'not in the sequence'),
           ({'system_class': 'polymer', 'species': [{'smiles': 'CCC'}]}, 'two attachment points'),
           ({'system_class': 'liquid', 'species': [{'smiles': 'C1CC'}]}, 'bad SMILES'),
           ({'system_class': 'liquid_mixture', 'species': [{'smiles': '[CH4:1]'}, {'smiles': 'CCO'}]}, 'every species or none'),
           ({'system_class': 'liquid_mixture', 'species': [{'smiles': 'C'}, {'smiles': 'CCO'}]}, '"count" or a "fraction"')]
    for spec, why in bad:
        d = derive(spec, 'x')
        assert not d['ok'] and any(why in e for e in d['errors']), (spec, d['errors'])


def test_system_tools_flow(tmp_path, monkeypatch, capsys):
    from ascg.cli import main as cli
    monkeypatch.setenv('ASCG_STUDIES', str(tmp_path/'studies'))
    monkeypatch.setenv('ASCG_LIBRARY', 'none')

    def run(args):
        rc = cli(args)
        return rc, json.loads(capsys.readouterr().out)
    spec = {'system_class': 'polymer', 'species': [{'name': 'MMA', 'smiles': MMA, 'bead_types': {'1': 'BB', '2': 'E'}}]}
    rc, out = run(['system', 'check', '--spec', json.dumps(spec), '--key', 'pmma'])
    assert rc == 0 and out['result']['ok'] and out['result']['size']['chains'] == 50
    run(['study', 'init', '--study', 'sy'])
    rc, out = run(['aa', 'plan', '--task', 'pmma', '--states', '400,500', '--study', 'sy'])
    assert rc == 0 and out['result']['known_recipe'] is False
    rc, out = run(['system', 'save', '--study', 'sy', '--key', 'pmma', '--spec', json.dumps(spec), '--rationale', 'test'])
    assert rc == 0 and out['result']['mapping'] == 'proposed', out
    s = Study('sy')
    assert s.system('pmma')['status'] == 'proposed' and s.mapping('pmma')['status'] == 'proposed'
    rc, out = run(['aa', 'plan', '--task', 'pmma', '--states', '400,500', '--study', 'sy'])
    plan = out['result']
    assert plan['known_recipe'] and '<system>/recipes.json' in plan['steps'][0] and 'map-aa' in plan['steps'][-1]
    assert any('still proposed' in n for n in plan['notes'])
    rc, out = run(['system', 'approve', '--study', 'sy', '--key', 'pmma', '--note', 'yes, PMMA 10-mers, two beads'])
    assert rc == 0 and out['result']['status'] == 'approved' and out['result']['mapping'] == 'approved'
    assert s.mapping('pmma')['approval_note'] == 'yes, PMMA 10-mers, two beads'
    events = [json.loads(x)['event'] for x in (s.dir/'events.jsonl').read_text().splitlines()]
    assert {'system_proposed', 'mapping_proposed', 'system_approved', 'mapping_approved'} <= set(events)
    recipes = json.loads((s.dir/'systems'/'pmma'/'recipes.json').read_text())
    assert recipes['tasks']['pmma']['chains'] == 50 and recipes['defaults']['tacticity'] == 'atactic'
    bad = {'system_class': 'polymer', 'species': [{'smiles': 'CCC'}]}
    rc, out = run(['system', 'save', '--study', 'sy', '--key', 'bad', '--spec', json.dumps(bad)])
    assert rc == 2 and out.get('refused_by_policy')


def test_notes_about_a_saved_system(tmp_path, monkeypatch, capsys):
    from ascg.cli import main as cli
    monkeypatch.setenv('ASCG_STUDIES', str(tmp_path/'studies'))
    cli(['study', 'init', '--study', 'n'])
    spec = {'system_class': 'liquid', 'species': [{'smiles': 'CCCCC'}]}
    cli(['system', 'save', '--study', 'n', '--key', 'pentane', '--spec', json.dumps(spec)])
    capsys.readouterr()
    assert cli(['study', 'note', '--study', 'n', '--task', 'pentane', '--message', 'user asked for 300 K']) == 0
    assert cli(['study', 'note', '--study', 'n', '--task', 'nothing', '--message', 'x']) == 1


# the benchmark's copolymers, one bead per repeat unit as in their library mappings
COPOLYMER_UNITS = {'S': '*[CH2:1][CH:1](*)[c:1]1[cH:1][cH:1][cH:1][cH:1][cH:1]1', 'BD': '*[CH2:1][CH:1]=[CH:1][CH2:1]*',
                   'E': '*[CH2:1][CH2:1]*', 'P': '*[CH2:1][CH:1](*)[CH3:1]', 'VA': '*[CH2:1][CH:1](*)[OH:1]'}
COPOLYMERS = {'sbs': {'sequence': 'S*3 BD*4 S*3'}, 'ep': {'random': {'E': 0.5, 'P': 0.5}, 'repeat_units': 10},
              'eva_alt': {'sequence': 'E VA E VA E VA E VA E VA'}, 'eva_periodic': {'sequence': 'E E VA E E VA E E VA'}}
CHAINS = {'sbs': 100, 'ep': 100, 'eva_alt': 100, 'eva_periodic': 111}          # the benchmark builds


def copolymer(key):
    t = P27['tasks'][key]
    names = sorted(set(t['random']) if 'random' in t else {x.split('*')[0] for x in t['sequence'].split()})
    return {'system_class': 'polymer', 'species': [{'name': n, 'smiles': COPOLYMER_UNITS[n], 'bead_types': {'1': n}}
                                                   for n in names], **COPOLYMERS[key]}


@pytest.mark.parametrize('key', sorted(COPOLYMERS))
def test_benchmark_copolymers(key):
    from ascg_core.aa.autopoly_v2 import study_recipe
    from ascg_core.mapping.formats import library_inputs
    d = derive(copolymer(key), key)
    assert d['ok'], d['errors']
    task, ref = d['recipe']['tasks'][key], P27['tasks'][key]
    assert task['chains'] == CHAINS[key] and d['size']['cg_beads'] in (1000, 999)
    for code, u in d['recipe']['units'].items():
        assert all(canon(u[p]) == canon(P27['units'][code][p]) for p in ('first', 'middle', 'last'))
    mine = study_recipe(key, '/tmp/x', recipes=d['recipe'])
    theirs = study_recipe(key, '/tmp/x', recipes={**P27, 'tasks': {key: {**ref, 'chains': CHAINS[key]}}})
    seq = lambda r: [[canon(x) for x in p.sequence] for p in r.polymers]           # noqa: E731
    assert seq(mine) == seq(theirs) and [p.chain_num for p in mine.polymers] == [p.chain_num for p in theirs.polymers]
    _, maps = library_inputs(key, d['library'], d['recipe'])
    assert set(maps['tasks'][key]['units']) == set(d['recipe']['units'])


def test_copolymer_bonded_types_follow_the_sequence():
    sbs = derive(copolymer('sbs'), 'sbs')
    assert sbs['bonded_types']['bonds'] == ['BD-BD', 'BD-S', 'S-S']
    assert 'S-BD-S' not in sbs['bonded_types']['angles'] and 'BD-S-BD' not in sbs['bonded_types']['angles']
    assert not any('fewer than' in w for w in sbs['warnings'])                            # 2 junctions x 100 chains
    di = derive({**copolymer('sbs'), 'sequence': 'S*5 BD*5', 'chains': 30}, 'di')
    assert any('BD-S' in w and 'fewer than 100' in w for w in di['warnings'])           # 1 junction x 30 chains
    ep = derive(copolymer('ep'), 'ep')
    assert {'E-P', 'E-E', 'P-P'} <= set(ep['bonded_types']['bonds']) and 'E-P-E' in ep['bonded_types']['angles']
    alt = derive(copolymer('eva_alt'), 'eva_alt')
    assert alt['bonded_types']['bonds'] == ['E-VA'] and 'E-E' not in alt['bonded_types']['bonds']
