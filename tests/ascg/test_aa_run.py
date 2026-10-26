"""aa build / submit / status / map / cancel: the AA campaign of a study's approved system."""
import json
import sys
import time
from pathlib import Path

import pytest

from ascg.cli import main as cli
from ascg.study import Study

ROOT = Path(__file__).resolve().parents[2]
PENTANE = {'system_class': 'liquid', 'species': [{'name': 'pentane', 'smiles': '[CH3:1][CH2:1][CH2:2][CH2:3][CH3:3]',
                                                  'bead_types': {'1': 'E', '2': 'M', '3': 'E'}}]}


def run(args, capsys):
    rc = cli(args)
    return rc, json.loads(capsys.readouterr().out)


@pytest.fixture
def study(tmp_path, monkeypatch, capsys):
    pytest.importorskip('rdkit')
    monkeypatch.setenv('ASCG_STUDIES', str(tmp_path/'studies'))
    monkeypatch.setenv('ASCG_LIBRARY', 'none')
    prof = tmp_path/'compute.json'
    prof.write_text(json.dumps({'targets': {
        'here': {'transport': 'local', 'scheduler': 'none', 'root': str(ROOT), 'python': sys.executable,
                 'env': f'export PYTHONPATH={ROOT}/src', 'work_root': str(tmp_path/'work'),
                 'aa': {'gmx': 'gmx', 'max_core_h': 500}},
        'cluster': {'transport': 'local', 'scheduler': 'slurm', 'root': str(ROOT), 'python': sys.executable,
                    'work_root': str(tmp_path/'cwork'), 'aa': {'gmx': 'gmx_mpi', 'max_core_h': 500,
                                                               'md': {'qos': 'cpu', 'partition': 'p'}}},
        'noaa': {'transport': 'local', 'scheduler': 'none', 'root': str(ROOT), 'python': sys.executable}}}))
    monkeypatch.setenv('ASCG_COMPUTE', str(prof))
    run(['study', 'init', '--study', 'a'], capsys)
    run(['system', 'save', '--study', 'a', '--key', 'pentane', '--spec', json.dumps(PENTANE)], capsys)
    return tmp_path


def test_build_needs_an_approved_system_and_aa_settings(study, capsys):
    rc, out = run(['aa', 'build', '--study', 'a', '--task', 'pentane', '--target', 'here'], capsys)
    assert rc == 2 and 'not approved' in out['error']
    run(['system', 'approve', '--study', 'a', '--key', 'pentane', '--note', 'yes'], capsys)
    rc, out = run(['aa', 'build', '--study', 'a', '--task', 'pentane', '--target', 'noaa'], capsys)
    assert rc == 2 and 'no AA settings' in out['error']


def test_a_real_job_that_fails_is_reported_with_its_log(study, capsys):
    """Local target: the build job really runs (and fails here without AutoPoly); its status file and log reach
    aa status, and aa submit refuses."""
    run(['system', 'approve', '--study', 'a', '--key', 'pentane', '--note', 'yes'], capsys)
    rc, out = run(['aa', 'build', '--study', 'a', '--task', 'pentane', '--target', 'here'], capsys)
    assert rc == 0, out
    d = Path(out['result']['dir'])
    assert json.loads((d/'inputs'/'recipes.json').read_text())['tasks']['pentane']['molecules'][0]['count'] == 1000
    for _ in range(60):
        rc, out = run(['aa', 'status', '--study', 'a', '--task', 'pentane'], capsys)
        if out['result']['pentane']['phase'] != 'building':
            break
        time.sleep(1)
    st = out['result']['pentane']
    script = (d/'jobs'/'build_01.sh').read_text()
    assert 'build-opls --recipes inputs/recipes.json --mappings inputs/library.json --task-key pentane' in script
    assert 'ASCG_TASK=a/aa:pentane' in script
    if st['phase'] == 'built':                               # AutoPoly available here
        return
    assert st['phase'] == 'build_failed' and st['jobs'][-1]['status'] == 'failed' and st['jobs'][-1]['log_tail']
    assert 'aa build --force' in st['next']
    events = [json.loads(x)['event'] for x in (Study('a').dir/'events.jsonl').read_text().splitlines()]
    assert 'aa_build_submitted' in events and 'aa_build_failed' in events
    rc, out = run(['aa', 'submit', '--study', 'a', '--task', 'pentane', '--states', '300', '--approval', 'go'], capsys)
    assert rc == 2 and 'needs a finished build' in out['error']


def test_campaign_flow_on_a_slurm_target(study, capsys, monkeypatch):
    """Build -> audit gate -> dry run with the cost -> approval -> equilibration then branches (afterok) -> map ->
    the reference, with a fake target."""
    from ascg.tools import aa_run_tools as A
    run(['system', 'approve', '--study', 'a', '--key', 'pentane', '--note', 'yes'], capsys)
    calls, n = [], [0]
    state = {'build': None, 'progress': {}, 'reference': None, 'done': set()}

    def fake(t, verb, p=None, timeout=0):
        calls.append((verb, p))
        if verb == 'aa_stage':
            import hashlib
            return {'md5': {k: hashlib.md5(v.encode()).hexdigest() for k, v in p['files'].items()}}
        if verb == 'aa_launch':
            n[0] += 1
            return {'name': f"{p['stage']}_{n[0]:02d}", 'stage': p['stage'], 'id': str(100 + n[0]),
                    'status_file': '/x', 'log': '/x', 'dirs': p.get('dirs'), 'after': p.get('after')}
        if verb == 'aa_prepare':
            state['progress'] = {'equil': {'steps': 5, 'done': 0, 'next': 'em_sc'},
                                 'T300': {'steps': 3, 'done': 0, 'next': 'pre'}}
            return {'equilibration_dir': 'equil', 'branch_dirs': ['T300']}
        if verb == 'aa_state':
            jobs = [{'name': j['name'], 'status': 'done' if j['name'] in state['done'] else 'submitted'}
                    for j in p['jobs']]
            return {'jobs': jobs, 'build': state['build'],
                    'protocol': {'progress': state['progress']} if state['progress'] else None,
                    'reference': state['reference']}
        raise AssertionError(verb)
    monkeypatch.setattr(A, 'remote', fake)
    rc, out = run(['aa', 'build', '--study', 'a', '--task', 'pentane', '--target', 'cluster'], capsys)
    assert rc == 0, out
    state['done'].add('build_01')
    state['build'] = {'atoms': 17000, 'molecules': 1000, 'charged_molecules': 0,
                      'audit': {'system.neutral.data': {'verdict': 'FINDINGS', 'findings': ['HINT_X']}}}
    rc, out = run(['aa', 'status', '--study', 'a'], capsys)
    assert out['result']['pentane']['phase'] == 'built'
    rc, out = run(['aa', 'submit', '--study', 'a', '--task', 'pentane', '--states', '300', '--dry-run'], capsys)
    assert rc == 2 and 'audit has findings' in out['error']
    state['build']['audit']['system.neutral.data'] = {'verdict': 'NEUTRAL_NO_HINTS', 'findings': []}
    rc, out = run(['aa', 'submit', '--study', 'a', '--task', 'pentane', '--states', '300', '--dry-run'], capsys)
    est = out['result']['plan']['estimate']
    assert rc == 0 and est['atoms'] == 17000 and est['core_h_estimate'] > 0 and out['result']['plan']['eq_ns'] == 4
    rc, out = run(['aa', 'submit', '--study', 'a', '--task', 'pentane', '--states', '300'], capsys)
    assert rc == 2 and '--approval' in out['error']
    rc, out = run(['aa', 'submit', '--study', 'a', '--task', 'pentane', '--states', '300', '--prod-ns', '400',
                   '--approval', 'yes'], capsys)
    assert rc == 2 and 'exceed' in out['error']                                     # over the 500 core-h cap
    rc, out = run(['aa', 'submit', '--study', 'a', '--task', 'pentane', '--states', '300',
                   '--approval', 'yes, 300 K at 1 atm, about 9 h'], capsys)
    assert rc == 0, out
    eq, br = out['result']['jobs']
    assert eq['dirs'] == ['equil'] and br['dirs'] == ['T300'] and br['after'] == eq['id']
    rc, out = run(['aa', 'map', '--study', 'a', '--task', 'pentane'], capsys)
    assert rc == 2 and 'every GROMACS directory' in out['error']
    state['done'] |= {eq['name'], br['name']}
    state['progress'] = {'equil': {'steps': 5, 'done': 5, 'next': None}, 'T300': {'steps': 3, 'done': 3, 'next': None}}
    rc, out = run(['aa', 'map', '--study', 'a', '--task', 'pentane'], capsys)
    assert rc == 0 and dict(calls)['aa_launch']['labels'] == ['300']
    state['done'].add(out['result']['job']['name'])
    state['reference'] = '/w/cg_ref/pentane'
    rc, out = run(['aa', 'status', '--study', 'a', '--task', 'pentane'], capsys)
    st = out['result']['pentane']
    assert st['phase'] == 'mapped' and 'aa import' in st['next'] and '/w/cg_ref/pentane' in st['next']
    reg = Study('a').load()
    assert reg['aa']['pentane']['approvals'][-1]['approval'].startswith('yes, 300 K')


def test_aa_job_script_for_the_mapping_stage(tmp_path):
    from ascg import remote
    d = tmp_path/'c'
    (d/'inputs').mkdir(parents=True)
    (d/'inputs'/'library.json').write_text('{}')
    j = remote.aa_launch({'dir': str(d), 'key': 'k', 'study': 's', 'stage': 'map', 'scheduler': 'none',
                          'python': 'true', 'labels': ['400', '500']})
    text = Path(j['script']).read_text()
    assert '--run 400=protocol/T400 --run 500=protocol/T500 --out cg_ref/k' in text and 'ASCG_TASK=s/aa:k' in text
    for _ in range(50):
        if Path(j['status_file']).exists() and json.loads(Path(j['status_file']).read_text())['status'] != 'running':
            break
        time.sleep(0.1)
    st = remote.aa_state({'dir': str(d), 'key': 'k', 'jobs': [j]})
    assert st['jobs'][0]['status'] in ('done', 'failed')
