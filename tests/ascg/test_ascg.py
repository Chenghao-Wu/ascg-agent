"""ascg tool layer: registry / CLI, policies (fail closed), study registry and protocol lock, catalogue, hook guard,
trace mining, and a local-transport round trip of the host verbs on a synthetic mapped reference."""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ascg import policy
from ascg.cli import main as cli
from ascg.protocols import V1, agentic, cycle_length
from ascg.registry import PolicyError, TOOLS, ToolError, validate
from ascg.study import Study

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT/'plugin'/'ascg'/'hooks'/'guard.py'


def run_cli(args, capsys):
    rc = cli(args)
    return rc, json.loads(capsys.readouterr().out)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv('ASCG_STUDIES', str(tmp_path/'studies'))
    prof = tmp_path/'compute.json'
    prof.write_text(json.dumps({'targets': {'here': {'transport': 'local', 'scheduler': 'none', 'root': str(ROOT),
                                                       'python': sys.executable, 'work_root': str(tmp_path/'work'),
                                                       'gpus': [0]}}}))
    monkeypatch.setenv('ASCG_COMPUTE', str(prof))
    return tmp_path


def test_protocol_matches_the_pipeline():
    from ascg_core.ibi.pipeline import PROTOCOL
    assert V1 == PROTOCOL
    a = agentic()
    assert a['extension']['max'] == 0 and a['gates'] == PROTOCOL['gates']
    assert cycle_length(V1) == 6 and cycle_length(V1, has_dihedrals=False) == 5


def test_registry_validation_and_cli(env, capsys):
    t = TOOLS[('run', 'act')]
    with pytest.raises(ToolError):
        validate(t, {'study': 's'})                                  # missing task, action, rationale
    with pytest.raises(ToolError):
        validate(t, {'study': 's', 'task': 'k', 'action': 'relax_gate', 'rationale': 'x'})
    assert validate(t, {'study': 's', 'task': 'k', 'action': 'extend', 'rationale': 'x', 'cycles': '2'})['cycles'] == 2
    rc, out = run_cli(['nonsense', 'tool'], capsys)
    assert rc == 1 or rc == 0                                        # passed to the legacy CLI or reported
    rc, out = run_cli(['study', 'init', '--study', 's1'], capsys)
    assert rc == 0 and out['result']['protocol']['extension']['max'] == 0
    rc, out = run_cli(['study', 'init', '--study', 's1'], capsys)
    assert rc == 1 and 'exists' in out['error']


def test_policies_fail_closed():
    t = {'name': 'ws', 'gpus': [0, 1, 2], 'gpu_policy': 'avoid_foreign', 'tasks_per_gpu': 1}
    gpus = [{'index': 0, 'foreign': [{'pid': 1}], 'own': []}, {'index': 1, 'foreign': [], 'own': []},
            {'index': 3, 'foreign': [], 'own': []}]
    assert policy.gpu_choice(gpus, t, {}) == 1                       # 0 has someone else's job, 3 is not allowed
    with pytest.raises(PolicyError):
        policy.gpu_choice(gpus, t, {1: 1})                           # our own task already there
    with pytest.raises(PolicyError):
        policy.slurm_capacity({'name': 'c', 'submit': {'max_jobs': 2}}, 2)
    rec = {'interventions': [{'extra_iters': 18}, {'extra_iters': 12}]}
    with pytest.raises(PolicyError):
        policy.action_budget(rec, 'extend', 12)                      # 18 + 12 + 12 > 36
    with pytest.raises(PolicyError):
        policy.action_budget({}, 'loosen_density_gate', 0)
    with pytest.raises(PolicyError):
        policy.own_job({'job': {}})


def test_study_protocol_lock_and_events(env):
    s = Study('lock')
    reg = s.init(agentic())
    s.check_protocol_lock(reg)
    p = s.protocol()
    p['gates']['density_rel_error_max'] = 0.10                       # an edit that would loosen a gate
    (s.dir/'protocol.json').write_text(json.dumps(p))
    with pytest.raises(ToolError):
        s.check_protocol_lock(reg)
    ev = s.event(reg, 'progress', 'pe', {'iteration': 6})
    assert [e['event'] for e in s.events(ev['seq'] - 1)] == ['progress']


def test_act_refused_unless_waiting_and_amend_needs_a_person(env, capsys):
    run_cli(['study', 'init', '--study', 's2'], capsys)
    s = Study('s2')
    reg = s.load()
    reg['tasks']['pe'] = {'key': 'pe', 'target': 'here', 'ref': '/x', 'task_dir': '/y', 'status': 'running'}
    s.save(reg)
    rc, out = run_cli(['run', 'act', '--study', 's2', '--task', 'pe', '--action', 'extend', '--rationale', 'x'], capsys)
    assert rc == 2 and out['refused_by_policy']
    rc, out = run_cli(['amend', 'propose', '--study', 's2', '--title', 't', '--rule-change', 'r', '--evidence', 'e'],
                      capsys)
    assert rc == 0 and out['result']['status'] == 'pending'
    sys.stdin = io.StringIO('A01\n')                                  # not a terminal
    rc, out = run_cli(['amend', 'approve', '--study', 's2', '--id', 'A01'], capsys)
    sys.stdin = sys.__stdin__
    assert rc == 2
    calls = [json.loads(x) for x in (s.dir/'calls.jsonl').read_text().splitlines()]
    assert any(c['tool'] == 'run act' and c['kind'] == 'policy' for c in calls)


LIBRARY = ROOT/'benchmarks'/'ascg_library.json'


@pytest.fixture
def library(monkeypatch):
    """The project's benchmark systems as a user system library (ascg has no built-in systems)."""
    monkeypatch.setenv('ASCG_LIBRARY', str(LIBRARY))
    return LIBRARY


def test_catalogue_aliases(library, capsys):
    rc, out = run_cli(['catalogue', 'search', '--query', 'nylon 6'], capsys)
    assert [m['key'] for m in out['result']['matches']] == ['pcl']
    rc, out = run_cli(['catalogue', 'search', '--query', 'polystyrene at 400 K'], capsys)
    assert [m['key'] for m in out['result']['matches']] == ['ps']


@pytest.mark.parametrize('cmd,blocked', [
    ('ascg amend approve --study s --id A01', True), ('ssh host scancel 1', True), ('pkill -f fastmd', True),
    ('cat ~/.ssh/id_ed25519', True), ('/opt/fastmd/build/fastmd run.conf', True), ('ssh host "rm -rf runs"', True),
    ('ascg study watch --study s --timeout 540', False), ('ssh host "ls runs"', False)])
def test_hook_guard(cmd, blocked):
    p = subprocess.run([sys.executable, str(GUARD), 'bash'], input=json.dumps({'tool_input': {'command': cmd}}),
                       capture_output=True, text=True)
    assert (p.returncode == 2) == blocked, p.stderr


def toy_reference(path):
    """A tiny mapped reference (2 chains of 6 beads, one type) with every file preflight needs."""
    from ascg_core.ibi.selfcheck import toy_system
    sysm, pos, box = toy_system(n_chains=2, length=6)
    path.mkdir(parents=True)
    (path/'topology.json').write_text(json.dumps({
        'type_names': ['A'], 'bead_types': [0] * sysm.n, 'bonds': sysm.bonds.tolist(), 'angles': sysm.angles.tolist(),
        'exclusions': sysm.bonds.tolist() + sysm.excl13.tolist(), 'bead_masses_amu': [28.0] * sysm.n,
        'bead_molecule': sysm.mol.tolist(), 'bond_types': ['A-A'], 'angle_types': ['A-A-A']}))
    states = {}
    for T in (400, 500):
        np.savez(path/f'targets_T{T}.npz', **{'rdf/A-A': np.ones(20), 'bond/A-A': np.ones(20), 'angle/A-A-A': np.ones(20)})
        np.savez(path/f'start_T{T}.npz', pos_nm=pos, box_nm=box)
        np.savez(path/f'mapped_T{T}.npz', pos_nm=np.repeat(pos[None], 8, 0))
        states[str(T)] = {'density_g_cm3': 0.8, 'density_se': 0.001, 'density_drift_g_cm3': 0.0, 'rg_nm': 0.5,
                          'ree_rms_nm': 1.0, 'r_max_over_half_box': 0.5, 'frames': 8}
    (path/'summary.json').write_text(json.dumps({'task': 'toy', 'n_beads': sysm.n, 'type_names': ['A'],
                                                 'bond_types': ['A-A'], 'angle_types': ['A-A-A'], 'states': states}))
    return path


def test_local_transport_round_trip(env, capsys):
    from ascg.compute import remote, target
    ref = toy_reference(env/'ref'/'toy')
    t = target('here')
    assert remote(t, 'ping')['ascg']
    rev = remote(t, 'map_review', {'ref': str(ref), 'task': 'toy'})
    assert rev['ok'] and rev['n_beads'] == 12 and rev['n_dihedrals'] == 6
    run_cli(['study', 'init', '--study', 's3'], capsys)
    base = ['aa', 'import', '--study', 's3', '--task', 'toy', '--target', 'here', '--ref', str(ref)]
    rc, out = run_cli(base, capsys)
    assert rc == 1 and 'states' in out['error']                       # state points are never a default
    rc, out = run_cli(base + ['--states', '400,450'], capsys)
    assert rc == 2 and "no AA reference at ['450']" in out['error']  # an approved state without a reference
    rc, out = run_cli(base + ['--states', '400@1000'], capsys)
    assert rc == 2 and "['400_P1000']" in out['error']               # nor at another pressure
    rc, out = run_cli(base + ['--states', '500'], capsys)             # a single approved state of the reference
    assert rc == 0 and out['result']['state_points'] == ['500'] and out['result']['fit'] == 'single-state IBI'
    assert out['result']['reference_states_not_used'] == ['400'] and out['result']['preflight']['states'] == ['500']
    rc, out = run_cli(base + ['--states', '400,500'], capsys)
    assert rc == 0 and out['result']['preflight']['ok'] and out['result']['qa']['ok']
    reg = Study('s3').load()
    assert reg['tasks']['toy']['status'] == 'registered' and reg['tasks']['toy']['state_points'] == ['400', '500']
    assert [e['event'] for e in Study('s3').events()] == ['study_created', 'task_registered', 'task_registered']
    rv = remote(t, 'targets_review', {'ref': str(ref)})
    assert any(k.startswith('dihedral/') for k in rv['targets']['400'])
    rc, out = run_cli(['study', 'note', '--study', 's3', '--task', 'toy', '--message', 'checked by hand'], capsys)
    assert rc == 0 and Study('s3').events()[-1]['event'] == 'note'


def test_trace_mine(tmp_path):
    from ascg.tools.trace_tools import mine
    lines = [{'type': 'user', 'message': {'role': 'user', 'content': 'coarse-grain PE'}, 'timestamp': 't0'},
             {'type': 'assistant', 'message': {'content': [{'type': 'tool_use', 'id': 'a', 'name': 'Bash',
                                                            'input': {'command': 'ascg study status --study s'}}]}},
             {'type': 'assistant', 'message': {'content': [{'type': 'tool_use', 'id': 'b', 'name': 'Bash',
                                                            'input': {'command': 'ls'}}]}},
             {'type': 'user', 'message': {'content': [{'type': 'tool_result', 'tool_use_id': 'b', 'is_error': True,
                                                       'content': 'boom'}]}, 'timestamp': 't1'}]
    f = tmp_path/'t.jsonl'
    f.write_text('\n'.join(json.dumps(x) for x in lines))
    m = mine(f)
    assert m['ascg_calls'] == {'study status': 1} and m['errors'] == 1 and m['user_messages'] == 1


def test_slurm_directives_follow_the_shebang(tmp_path):
    """Regression (eval_pe): #SBATCH lines after a command are ignored by sbatch."""
    from ascg.remote import job_script
    txt = job_script({'scheduler': 'slurm', 'task': 'pe', 'study': 's', 'env': 'source /x/env.sh',
                      'submit': {'qos': '8gpus', 'partition': 'gpu4090,gpu3090'}}, tmp_path, 1, tmp_path/'spec.json')
    lines = txt.splitlines()
    first_cmd = next(i for i, l in enumerate(lines[1:], 1) if not l.startswith('#'))
    assert all(l.startswith('#SBATCH') for l in lines[1:first_cmd])
    assert '#SBATCH --partition=gpu4090,gpu3090' in lines[:first_cmd] and '#SBATCH --qos=8gpus' in lines[:first_cmd]


def test_workstation_job_leader_carries_the_marker(env):
    """Regression (eval_pe): the recorded PID must be the live session leader with the ASCG_TASK marker."""
    import time
    from ascg.remote import _proc_marker, launch
    if not Path('/proc').exists():
        pytest.skip('needs /proc (Linux)')
    td = env/'work'/'s'/'t'
    res = launch({'mode': 'fit', 'study': 's', 'task': 't', 'task_dir': str(td), 'ref': '/nonexistent',
                  'protocol': {}, 'scheduler': 'none', 'env': 'sleep 3'})
    time.sleep(0.5)
    assert _proc_marker(res['pid']) == 's/t'
    os.kill(res['pid'], 0)                                              # alive (the env line sleeps)
    os.killpg(res['pid'], 15)


def test_import_refuses_to_orphan_jobs(env, capsys):
    """Regression (eval_pe): re-importing a task that has jobs would lose track of them."""
    run_cli(['study', 'init', '--study', 's4'], capsys)
    s = Study('s4')
    reg = s.load()
    reg['tasks']['pe'] = {'key': 'pe', 'target': 'here', 'ref': '/x', 'task_dir': '/y', 'status': 'error',
                          'jobs': [{'pid': 1}]}
    s.save(reg)
    rc, out = run_cli(['aa', 'import', '--study', 's4', '--task', 'pe', '--target', 'here', '--ref', '/x',
                       '--states', '400,500'], capsys)
    assert rc == 2 and 'orphan' in out['error']


def test_continuation_stage_counts_in_costs_and_bonds_default_run():
    """Regressions reported by evaluation agents (eval_sbs_cases): every stage a continuation runs is counted in the
    cost report, and diag bonds needs no --run."""
    import inspect
    from ascg import remote
    from ascg.protocols import continuation_plan
    from ascg_core.ibi import pipeline as P
    assert "for k in ('stage1', 'stage2', 'extension', 'validate')" in inspect.getsource(P.write_reports)
    for a in ({'action': 'extend', 'target_iters': 48}, {'action': 'rewind_smaller_timestep', 'target_iters': 42}):
        for at in (10, 30):
            assert {st['stage'] for st in continuation_plan(agentic(), a, at)} <= {'stage1', 'stage2', 'extension'}
    assert "continuation_plan(proto, a, at)" in inspect.getsource(remote.job)
    assert "(p.get('run') or '')" in inspect.getsource(remote._bonds_one)


def test_code_digest_covers_the_numerical_core():
    from ascg.remote import _code_digest
    d = _code_digest()
    assert 'ascg/remote.py' in d['files'] and 'ascg_core/ibi/pipeline.py' in d['files'] and len(d['digest']) == 16


def test_array_parameters_tolerate_split_json():
    """Regression (eval_sbs_cases): '--iterations "[35 36]"' arrived as tokens '[35', '36]'."""
    t = TOOLS[('diag', 'pressure')]
    for v in ('[35, 40]', '35,40', '35 40', ['[35', '40]'], ['[35,', '40]'], '[35 40]'):
        assert [int(x) for x in validate(t, {'study': 's', 'task': 'k', 'iterations': v})['iterations']] == [35, 40]


def test_digest_before_the_first_iteration(tmp_path):
    """Regression (eval_pe, eval_sbs): digest of a queued / starting task crashed on the missing state file."""
    from ascg.remote import digest
    d = digest({'task_dir': str(tmp_path)})
    assert d['completed'] is None and 'joblog' in d['note']


def test_launch_refuses_gates_the_classifier_cannot_honour(env, capsys):
    """Regression (eval_pe2 reviewer): classify() uses the pipeline's v1 gates, so a study protocol with other gates
    must be refused, not mislabelled."""
    run_cli(['study', 'init', '--study', 's5'], capsys)
    s = Study('s5')
    reg = s.load()
    p = s.protocol()
    p['gates']['density_rel_error_max'] = 0.10
    (s.dir/'protocol.json').write_text(json.dumps(p))
    reg['protocol']['hash'] = __import__('ascg.study', fromlist=['canonical_hash']).canonical_hash(p)
    reg['tasks']['pe'] = {'key': 'pe', 'target': 'here', 'ref': '/x', 'task_dir': '/y', 'status': 'registered',
                          'preflight_ok': True}
    s.save(reg)
    rc, out = run_cli(['run', 'submit', '--study', 's5', '--task', 'pe'], capsys)
    assert rc == 2 and 'classifier' in out['error']


def test_library_catalogue_plan_and_mapping(library, capsys):
    """Library systems are found, mapped (proposed) and planned with their own protocol and force
    field, with library-relative paths."""
    rc, out = run_cli(['catalogue', 'search', '--query', 'PMMA'], capsys)
    assert [m['key'] for m in out['result']['matches']] == ['pmma']
    rc, out = run_cli(['catalogue', 'search', '--query', 'acetone at 300 K'], capsys)
    m = out['result']['matches'][0]
    assert m['key'] == 'acetone' and m['system_class'] == 'liquid' and m['id'] == 'B37'
    rc, out = run_cli(['map', 'show', '--task', 'n_hexane'], capsys)
    sites = out['result']['mapping']['bead_sites']
    assert [s['site_id'] for s in sites] == ['HT1', 'HM', 'HT2'] and 'note' not in out['result']   # liquid: supported
    assert 'not approved' in out['result']['approval']
    rc, out = run_cli(['map', 'show', '--task', 'pmma'], capsys)
    assert 'note' not in out['result']                    # polymer in SMILES form: supported (mapping.formats)
    rc, out = run_cli(['aa', 'plan', '--task', 'acetone', '--states', '300'], capsys)
    r = out['result']
    assert r['equilibration'] == 'direct' and any('--equilibration direct' in s and '--states 300' in s for s in r['steps'])
    assert any('map-aa' in s and '--mappings <library>/ascg_library.json' in s and '--run 300=' in s for s in r['steps'])
    rc, out = run_cli(['aa', 'plan', '--task', 'pdms', '--states', '400,500'], capsys)
    assert any('--ff-override <library>/ff/pdms_oplsaa_xiang2025.json' in s for s in out['result']['steps'])
    assert any('--recipes <library>/aa_recipes.json' in s for s in out['result']['steps'])
    rc, out = run_cli(['aa', 'plan', '--task', 'pe', '--states', '400,400@1000'], capsys)
    assert any('--run 400_P1000=' in s for s in out['result']['steps'])
    rc, out = run_cli(['aa', 'plan', '--task', 'brand_new', '--states', '300'], capsys)
    assert out['result']['known_recipe'] is False


def test_no_library_means_no_systems(tmp_path, monkeypatch, capsys):
    """Without a configured library ascg knows no system: the user specifies it."""
    monkeypatch.delenv('ASCG_LIBRARY', raising=False)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.chdir(tmp_path)
    rc, out = run_cli(['catalogue', 'search', '--query', 'polystyrene'], capsys)
    assert rc == 0 and out['result']['matches'] == [] and out['result']['library'] is None
    assert 'specify the system' in out['result']['note']
    rc, out = run_cli(['map', 'show', '--task', 'ps'], capsys)
    assert rc == 1 and 'from the user' in out['error']
    rc, out = run_cli(['aa', 'plan', '--task', 'ps', '--states', '400,500'], capsys)
    assert out['result']['known_recipe'] is False
    (tmp_path/'.config'/'ascg').mkdir(parents=True)                # a default library, switched off explicitly
    (tmp_path/'.config'/'ascg'/'library.json').symlink_to(LIBRARY)
    rc, out = run_cli(['catalogue', 'search', '--query', 'polystyrene'], capsys)
    assert [m['key'] for m in out['result']['matches']] == ['ps']
    monkeypatch.setenv('ASCG_LIBRARY', 'none')
    rc, out = run_cli(['catalogue', 'search', '--query', 'polystyrene'], capsys)
    assert out['result']['matches'] == [] and out['result']['library'] is None


def test_plugin_has_no_library_systems(capsys):
    """Leakage check: the plugin (skills, case libraries, agents, commands, hooks) and the tool schemas name no system
    of the benchmark library, so the skills stay general and the benchmark stays a test set."""
    import re
    names = set()
    for s in json.loads(LIBRARY.read_text())['systems']:
        names |= {str(n).strip() for n in (s['key'], s.get('name', ''), *s.get('aliases', []))}
    names = {n for n in names if len(n) >= 3}                     # 'pe', 'ps', 'pa' are too short to scan for
    texts = {str(p.relative_to(ROOT)): p.read_text() for p in (ROOT/'plugin').rglob('*')
             if p.is_file() and '__pycache__' not in p.parts}
    rc, out = run_cli(['tools', '--schema'], capsys)
    texts['ascg tools --schema'] = json.dumps(out)
    hits = [(n, where) for n in sorted(names) for where, t in texts.items()
            if re.search(r'(?<![A-Za-z0-9_])' + re.escape(n) + r'(?![A-Za-z0-9_])', t, re.I)]
    assert not hits, hits
    assert not any(re.search(r'benchmarks/|aa_recipes|cg_mappings', t) for t in texts.values())


def test_aa_audit_flags_a_mistyped_ester(env, capsys):
    """ascg aa audit (local transport) on a moltemplate-style build: ethyl acetate with its O-CH2 typed as ether."""
    d = env/'build'
    d.mkdir()
    atoms = [('C', -0.18, 'n-CH3 all-atom C: alkanes')] + [('H', 0.06, 'AA H on C-alpha in ketone')] * 3 + \
            [('C', 0.49, 'AA C: esters'), ('O', -0.41, 'AA =O: esters'), ('O', -0.33, 'AA -OR: ester -'),
             ('C', 0.14, 'C(H2OR): ethyl ether')] + [('H', 0.03, "alkoxy H's in esters")] * 2 + \
            [('C', -0.18, 'n-CH3 all-atom C: alkanes')] + [('H', 0.06, 'H all-atom H: alkanes')] * 3
    bonds = [(1, 2), (1, 3), (1, 4), (1, 5), (5, 6), (5, 7), (7, 8), (8, 9), (8, 10), (8, 11), (11, 12), (11, 13), (11, 14)]
    types = {}
    for e, q, dsc in atoms:
        types.setdefault((e, q, dsc), len(types) + 1)
    mass = {'C': 12.011, 'H': 1.008, 'O': 15.999}
    lines = ['test', '', f'{len(atoms)} atoms', '', 'Masses', ''] + [f'{t} {mass[e]}' for (e, q, dsc), t in types.items()]
    lines += ['', 'Atoms  # full', ''] + [f'{i} 1 {types[a]} {a[1]} 0 0 0' for i, a in enumerate(atoms, 1)]
    lines += ['', 'Bonds', ''] + [f'{i} 1 {a} {b}' for i, (a, b) in enumerate(bonds, 1)]
    (d/'system.data').write_text('\n'.join(lines) + '\n')
    (d/'system.in.charges').write_text('\n'.join(f'set type {t} charge {q} # {dsc}' for (e, q, dsc), t in types.items()))
    rc, out = run_cli(['aa', 'audit', '--target', 'here', '--system-dir', str(d)], capsys)
    r = out['result']['system.data']
    assert rc == 0 and r['verdict'] == 'FINDINGS' and abs(r['templates'][0]['net_charge_e'] + 0.05) < 1e-9
    assert {f['code'] for f in r['findings']} == {'NONNEUTRAL', 'HINT_ESTER_ALKOXY_AS_ETHER'}


def test_rewind_into_stage1_finishes_stage1_first(tmp_path, monkeypatch):
    """A rewind that lands inside stage 1 resumes stage 1 (its schedule, no pressure correction) before stage 2, with
    the reduced timestep in both (found by the acetone_xec agent: stage 1 had been cut from 15 to 10 iterations)."""
    from ascg import remote
    from ascg_core.ibi import pipeline as P
    td = tmp_path/'task'
    (td/'workflow').mkdir(parents=True)
    for k in (10, 20):
        (td/'cg'/'ibi'/f'iter_{k:03d}').mkdir(parents=True)
        (td/'cg'/'ibi'/f'iter_{k:03d}'/'metrics.json').write_text(json.dumps({'dt_ps': 0.0015}))
    (td/'workflow'/'state.json').write_text(json.dumps({'completed': 11, 'history': [],
                                                        'settings': {'dihedrals': None, 'model': 'tdep'}}))
    calls = []
    monkeypatch.setattr(P, 'preflight', lambda ref, states=None: {'ok': True, 'n_dihedrals': 0, 'states': ['300']})
    monkeypatch.setattr(P, '_call', lambda log, stage, argv, retries: calls.append((stage, argv)))
    monkeypatch.setattr(P, 'converged', lambda *a, **k: (False, 'stop before validation'))
    proto = agentic()
    action = {'action': 'rewind_smaller_timestep', 'to_iteration': 10, 'dt_factor': 0.67, 'target_iters': 42,
              'rationale': 'test'}
    out = remote.job({'task_dir': str(td), 'ref': str(tmp_path/'ref'), 'protocol': proto, 'mode': 'continue',
                      'task': 't', 'action': action})
    assert out['status'] == 'awaiting_decision'
    opt = lambda argv, k: argv[argv.index(k) + 1] if k in argv else None
    got = [(s, opt(a, '--iters'), opt(a, '--schedule'), opt(a, '--pressure-correction'), opt(a, '--dt')) for s, a in calls]
    s1, s2 = proto['stage1'], proto['stage2']
    assert got == [('rewind', None, None, None, None),
                   ('stage1', str(s1['iters']), s1['schedule'], s1['pressure_correction'], '0.001005'),
                   ('stage2', '42', s2['schedule'], s2['pressure_correction'], '0.001005')]
    calls.clear()                                   # a rewind inside stage 2 goes straight on with stage 2
    remote.job({'task_dir': str(td), 'ref': str(tmp_path/'ref'), 'protocol': proto, 'mode': 'continue', 'task': 't',
                'action': {**action, 'to_iteration': 20}})
    assert [s for s, _ in calls] == ['rewind', 'stage2']



# ---------------------------------------------------------------- CLI conveniences and agent-facing behaviour
def test_cli_positional_alias_and_usage(library, capsys):
    """Calls agents made that failed: a positional value, --target for --targets, and errors without the usage."""
    rc, out = run_cli(['catalogue', 'search', 'acetone'], capsys)
    assert rc == 0 and [m['key'] for m in out['result']['matches']] == ['acetone']
    from ascg.registry import _alias
    assert _alias(TOOLS[('compute', 'hosts')], {'target': ['x']}) == {'targets': ['x']}
    assert _alias(TOOLS[('diag', 'bonds')], {'iterations': [1]}) == {'iterations': [1]}        # both exist: untouched
    rc, out = run_cli(['run', 'act', '--study', 's'], capsys)
    assert rc == 1 and out['usage'].startswith('ascg run act --study <string> --task <string> --action <string>')
    rc, out = run_cli(['study', 'list', 'extra'], capsys)
    assert rc == 1 and 'unexpected argument' in out['error']


def test_continuation_plan_and_cost_projection():
    from ascg.protocols import continuation_plan, project_cost
    p = agentic()
    rew = {'action': 'rewind_smaller_timestep', 'target_iters': 42}
    assert [(s['stage'], s['from'], s['to']) for s in continuation_plan(p, rew, 10)] == [('stage1', 11, 15),
                                                                                            ('stage2', 16, 42)]
    assert [(s['stage'], s['from'], s['to']) for s in continuation_plan(p, rew, 20)] == [('stage2', 21, 42)]
    ext = {'action': 'extend', 'target_iters': 48}
    assert [(s['stage'], s['from'], s['to']) for s in continuation_plan(p, ext, 42)] == [('extension', 43, 48)]
    assert project_cost(p, None, 10) is None
    c = project_cost(p, {'spent_gpu_h': 0.5, 's_per_iteration': 120.0, 'fit_ps_per_iteration': 550.0}, 30, 5)
    val = 120.0 * 3 * 1200 / 550 / 3600
    assert c['validation_gpu_h_est'] == round(val, 2) and c['remaining_iterations'] == 30
    assert c['projected_total_gpu_h'] == [round(0.5 + 30 * 120 / 3600 + val, 2), round(0.5 + 34 * 120 / 3600 + val, 2)]


def acetone_like_run(td, running=True):
    """A run state shaped like acetone_xec: stage 1 to iteration 10, a rewind, stage 2 from 11 at a smaller dt."""
    (td/'workflow').mkdir(parents=True)
    base = {'schedule': 'pair', 'pressure_correction': 'none', 'lam': 0.2, 'dt': None, 'eq_ps': 50.0, 'prod_ps': 500.0,
            'model': 'tdep', 'dihedrals': None, 'temperatures': [300], 'states': ['300']}
    inv = [{'time': 't0', 'start_iteration': 0, 'settings': base},
           {'time': 't1', 'start_iteration': 11, 'settings': {**base, 'schedule': 'pair:3,bond:1,angle:1,dihedral:1',
                                                              'pressure_correction': 'virial', 'dt': 0.001005}}]
    hist = [{'iteration': i, 'family': 'pair', 'wall_s': 100.0 if i <= 10 else 120.0, 'p_bar_300': 10.0,
             'rdf_300': {'A-A': 0.02}} for i in range(0, 21)]
    state = {'settings': base, 'invocations': inv, 'history': hist, 'completed': 20,
             'rewinds': [{'time': 't1', 'from': 10, 'to': 10, 'reason': 'C4 genuine instability',
                          'failed': {'reason': 'broken bonds'}}]}
    (td/'workflow'/'state.json').write_text(json.dumps(state))
    for i in (1, 11):
        (td/'cg'/'ibi'/f'iter_{i:03d}').mkdir(parents=True)
        (td/'cg'/'ibi'/f'iter_{i:03d}'/'metrics.json').write_text(json.dumps({'dt_ps': 0.0015 if i == 1 else 0.001005}))
    (td/'workflow'/'stage_timings.json').write_text(json.dumps({'stage1': 1000.0, 'rewind': 0.1, 'stage2': 1100.0}))
    import time as _t
    start = _t.strftime('%Y-%m-%d %H:%M:%S', _t.localtime(_t.time() - 360))
    (td/'workflow'/'provenance.jsonl').write_text(json.dumps({'time': start, 'event': 'stage start', 'stage': 'stage2'})
                                                  + '\n')
    (td/'workflow'/'ascg_job.json').write_text(json.dumps({'status': 'running' if running else 'done'}))
    return state


def test_run_history_cost_and_stage_boundaries(tmp_path):
    from ascg import remote
    td = tmp_path/'task'
    s = acetone_like_run(td)
    inv = remote._invocations(td, s)
    assert [(v['from'], v['to'], v['pressure_correction'], v['dt_ps']) for v in inv] == [(0, 10, 'none', 0.0015),
                                                                                       (11, 20, 'virial', 0.001005)]
    assert remote._stage1_end(td, s) == 10                               # not the protocol's 15
    c = remote._cost(td, s)
    assert 2100 / 3600 + 0.09 < c['spent_gpu_h'] < 2100 / 3600 + 0.11  # finished stages + the running one (~6 min)
    assert abs(c['s_per_iteration'] - (2 * 100 + 10 * 120) / 12) < 1e-9 and c['fit_ps_per_iteration'] == 550.0
    d = remote.digest({'task_dir': str(td)})
    assert d['settings']['pressure_correction'] == 'virial' and d['settings']['dt'] == 0.001005   # latest, not stage 1
    assert d['stage1_end'] == 10 and len(d['invocations']) == 2
    (td/'workflow'/'ascg_job.json').write_text(json.dumps({'status': 'done'}))
    assert abs(remote._cost(td, s)['spent_gpu_h'] - 2100 / 3600) < 1e-9


def test_targets_review_separates_noise_from_resolution(tmp_path):
    """The old curvature 'roughness' flagged narrow smooth peaks (acetone) and missed noise; block SEs measure noise."""
    from ascg import remote
    ref = tmp_path/'ref'
    ref.mkdir()
    x = np.arange(90)
    narrow = np.exp(-0.5 * ((x - 41) / 1.0)**2)                  # smooth peak ~2 bins wide (as acetone's angle)
    broad = np.exp(-0.5 * ((x - 45) / 12.0)**2)
    np.savez(ref/'targets_T300.npz', **{'angle/N-N-N': narrow, 'angle_se/N-N-N': 0.002 * narrow,
                                       'bond/B-B': broad, 'bond_se/B-B': 0.08 * broad,
                                       'rdf/N-N': np.ones(50), 'rdf_se/N-N': 0.001 * np.ones(50)})
    (ref/'summary.json').write_text(json.dumps({'states': {'300': {'r_max_over_half_box': 0.6}}}))
    monkey = remote._dihedral_review
    remote._dihedral_review = lambda ref, temps: {}
    try:
        out = remote.targets_review({'ref': str(ref)})
    finally:
        remote._dihedral_review = monkey
    t = out['targets']['300']
    assert t['angle/N-N-N']['rel_se'] == 0.002 and t['angle/N-N-N']['bins_fwhm'] < 5
    assert len(out['warnings']) == 1 and 'bond/B-B' in out['warnings'][0] and 'noisy' in out['warnings'][0]
    assert len(out['notes']) == 1 and 'angle/N-N-N' in out['notes'][0] and 'under-resolved' in out['notes'][0]


def test_verify_explain_default_and_status_budget(env, capsys):
    rc, out = run_cli(['study', 'init', '--study', 'sv'], capsys)
    rc, out = run_cli(['study', 'verify', '--study', 'sv'], capsys)
    assert rc == 0 and out['result']['ok'] and out['result']['lock_ok'] and out['result']['gates_match_classifier']
    s = Study('sv')
    reg = s.load()
    reg['tasks']['k'] = {'key': 'k', 'target': 'here', 'status': 'running', 'interventions': [
        {'action': 'rewind_smaller_timestep', 'extra_iters': 2, 'target_iters': 42}],
        'remote': {'completed': 30, 'cost': {'spent_gpu_h': 1.0, 's_per_iteration': 120.0,
                                              'fit_ps_per_iteration': 550.0}}}
    s.save(reg)
    from ascg.tools.study_tools import summary
    b = summary(s.load(), s.protocol())['k']['budget']
    assert b['actions_left'] == 2 and b['extra_iterations_left'] == 34
    assert b['cost']['remaining_iterations'] == 12 and b['cost']['projected_total_gpu_h'][0] > 1.0 + 12 * 120 / 3600
    rc, out = run_cli(['study', 'explain', '--study', 'sv'], capsys)        # the only task is the default
    assert rc == 0 and out['result']['task'] == 'k'
    reg = s.load()
    reg['tasks']['k2'] = {'key': 'k2', 'target': 'here', 'status': 'registered'}
    s.save(reg)
    rc, out = run_cli(['study', 'explain', '--study', 'sv'], capsys)
    assert rc == 1 and "one of ['k', 'k2']" in out['error']
    (s.dir/'protocol.json').write_text(json.dumps({**s.protocol(), 'gates': {'density_rel_error_max': 0.5,
                                                                            'rdf_nrmse_max': 0.15}}))
    rc, out = run_cli(['study', 'verify', '--study', 'sv'], capsys)
    assert not out['result']['ok'] and not out['result']['lock_ok'] and not out['result']['gates_match_classifier']


def test_act_dry_run_plans_and_projects(env, capsys, monkeypatch):
    from ascg.tools import run_tools
    rc, out = run_cli(['study', 'init', '--study', 'sa'], capsys)
    s = Study('sa')
    reg = s.load()
    reg['tasks']['k'] = {'key': 'k', 'target': 'here', 'status': 'stopped_on_gate', 'task_dir': '/x', 'ref': '/r'}
    s.save(reg)
    monkeypatch.setattr(run_tools, 'remote', lambda t, verb, p, timeout=0: {
        'completed': 11, 'settings': {'dihedrals': None},
        'cost': {'spent_gpu_h': 0.3, 's_per_iteration': 80.0, 'fit_ps_per_iteration': 550.0}})
    launched = []
    monkeypatch.setattr(run_tools, '_launch', lambda *a, **k: launched.append(a) or {'id': 1})
    rc, out = run_cli(['run', 'act', '--study', 'sa', '--task', 'k', '--action', 'rewind_smaller_timestep',
                       '--to-iteration', '10', '--dt-factor', '0.67', '--rationale', 'C4', '--dry-run'], capsys)
    r = out['result']
    assert rc == 0 and r['dry_run'] and r['budget_check'] == 'ok' and not launched
    assert [(st['stage'], st['from'], st['to']) for st in r['plan']] == [('stage1', 11, 15), ('stage2', 16, 42)]
    assert r['projection']['s_per_iteration'] == round(80.0 / 0.67, 1) and r['projection']['remaining_iterations'] == 32
    assert not Study('sa').load()['tasks']['k'].get('interventions')        # nothing recorded
    rc, out = run_cli(['run', 'act', '--study', 'sa', '--task', 'k', '--action', 'rewind_smaller_timestep',
                       '--to-iteration', '10', '--dt-factor', '0.67', '--rationale', 'C4'], capsys)
    assert rc == 0 and launched and out['result']['plan'] == r['plan']


def test_compute_hosts_summarises_other_jobs(env, capsys, monkeypatch):
    from ascg.tools import compute_tools
    prof = json.loads(Path(os.environ['ASCG_COMPUTE']).read_text())
    prof['targets']['here']['scheduler'] = 'slurm'
    Path(os.environ['ASCG_COMPUTE']).write_text(json.dumps(prof))
    jobs = [{'id': '1', 'name': 'ascg-acetone', 'state': 'RUNNING'}, {'id': '2', 'name': 'prepcpu_0', 'state': 'RUNNING'},
            {'id': '3', 'name': 'pilotw', 'state': 'PENDING'}]
    monkeypatch.setattr(compute_tools, 'remote', lambda t, verb, timeout=0: [] if verb == 'gpus' else jobs)
    rc, out = run_cli(['compute', 'hosts', '--target', 'here'], capsys)
    h = out['result']['here']
    assert [j['id'] for j in h['ascg_jobs']] == ['1'] and h['other_jobs'] == {'count': 2, 'by_state': {'RUNNING': 1,
                                                                                                         'PENDING': 1}}


def test_reports_carry_the_fit_history(tmp_path):
    from ascg_core.ibi.pipeline import fit_history, write_reports
    td = tmp_path/'task'
    acetone_like_run(td, running=False)
    h = fit_history(json.loads((td/'workflow'/'state.json').read_text()))
    assert [(v['from'], v['to'], v['dt_ps']) for v in h['invocations']] == [(0, 10, None), (11, 20, 0.001005)]
    assert h['rewinds'][0]['to'] == 10 and h['rewinds'][0]['failed'] == 'broken bonds'
    write_reports(tmp_path/'ref', td, {'task': 'acetone', 'class': 'validated', 'reasons': []})
    us = (td/'report'/'user_summary.md').read_text()
    assert 'rewound to iteration 10' in us and 'iterations 11-20' in us and 'timestep 1.005 fs' in us
    assert '"rewinds"' in (td/'report'/'technical_report.md').read_text()


def test_slurm_launch_records_its_directives(tmp_path, monkeypatch):
    """The job record carries the #SBATCH directives (a reviewer could not see the QoS a job ran under)."""
    from ascg.remote import launch
    b = tmp_path/'bin'
    b.mkdir()
    (b/'sbatch').write_text('#!/bin/sh\necho 4242\n')
    (b/'sbatch').chmod(0o755)
    monkeypatch.setenv('PATH', f'{b}:{os.environ["PATH"]}')
    res = launch({'mode': 'fit', 'study': 's', 'task': 't', 'task_dir': str(tmp_path/'t'), 'ref': '/nonexistent',
                  'protocol': {}, 'scheduler': 'slurm', 'env': '', 'submit': {'qos': '8gpus', 'partition': 'gpu4090'}})
    assert res['id'] == '4242' and '--qos=8gpus' in res['sbatch'] and '--partition=gpu4090' in res['sbatch']


def test_moved_task_finds_its_mapping_and_cancel_clears_the_queue(env, library, capsys, monkeypatch):
    """n_hexane_xec session 2: under a new task key, map review lost the library mapping (looked up by the study key),
    and a cancelled task kept its stale queue estimate."""
    from ascg.tools import reference_tools, run_tools
    rc, out = run_cli(['study', 'init', '--study', 'mv'], capsys)
    s = Study('mv')
    reg = s.load()
    reg['tasks']['n_hexane_2'] = {'key': 'n_hexane_2', 'target': 'here', 'ref': '/r', 'task_dir': '/t',
                                  'status': 'queued', 'queue': {'state': 'PENDING'},
                                  'job': {'id': '9', 'mode': 'fit'}}
    s.save(reg)
    monkeypatch.setattr(reference_tools, 'remote', lambda t, verb, p, timeout=0: {'ok': True, 'reference_task': 'n_hexane'})
    rc, out = run_cli(['map', 'review', '--study', 'mv', '--task', 'n_hexane_2'], capsys)
    lm = out['result']['library_mapping']
    assert lm['key'] == 'n_hexane' and [x['site_id'] for x in lm['mapping']['bead_sites']] == ['HT1', 'HM', 'HT2']
    monkeypatch.setattr(run_tools, 'remote', lambda t, verb, p, timeout=0: {'cancelled': '9', 'rc': 0})
    monkeypatch.setattr(run_tools, 'own_job', lambda rec: rec['job'])
    rc, out = run_cli(['run', 'cancel', '--study', 'mv', '--task', 'n_hexane_2'], capsys)
    rec = Study('mv').load()['tasks']['n_hexane_2']
    assert rc == 0 and rec['status'] == 'cancelled' and rec['queue'] is None



def test_mcp_server_serves_the_registry(env, library, capsys):
    """The MCP adapter (for sandboxed harnesses such as Codex): same tools and results as the CLI, no amend approve,
    parameter errors with the usage line, policy refusals as errors."""
    from ascg.mcp import handle
    init = handle({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-06-18'}})
    assert init['result']['serverInfo']['name'] == 'ascg' and 'tools' in init['result']['capabilities']
    assert handle({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) is None
    names = {t['name'] for t in handle({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'})['result']['tools']}
    assert {'study_status', 'run_submit', 'diag_digest', 'catalogue_search'} <= names and 'amend_approve' not in names
    call = lambda name, args: handle({'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call',
                                      'params': {'name': name, 'arguments': args}})['result']
    r = call('catalogue_search', {'query': 'hexane'})
    assert not r['isError'] and json.loads(r['content'][0]['text'])['result']['matches'][0]['key'] == 'n_hexane'
    r = call('study_status', {})
    assert r['isError'] and json.loads(r['content'][0]['text'])['usage'].startswith('ascg study status')
    run_cli(['study', 'init', '--study', 'mcp'], capsys)
    s = Study('mcp')
    reg = s.load()
    reg['tasks']['k'] = {'key': 'k', 'target': 'here', 'status': 'running'}
    s.save(reg)
    r = call('run_act', {'study': 'mcp', 'task': 'k', 'action': 'extend', 'rationale': 'x'})
    assert r['isError'] and json.loads(r['content'][0]['text'])['refused_by_policy']
    assert handle({'jsonrpc': '2.0', 'id': 4, 'method': 'tools/call', 'params': {'name': 'amend_approve'}})['error']


def test_qos_choice_and_launch_with_several_qos(env, capsys, monkeypatch):
    """A target may list several QoS (e.g. 8gpus, 4gpus): each job goes to the first with GPUs left under its per-user
    cap, counting the account's running and pending jobs."""
    from ascg.policy import qos_choice
    jobs = [{'qos': '8gpus', 'state': 'RUNNING', 'gres': 'gres/gpu:a800:1'}] * 6 + \
           [{'qos': '8gpus', 'state': 'PENDING', 'gres': 'gres/gpu:1'}] * 2 + \
           [{'qos': '4gpus', 'state': 'PENDING', 'gres': 'gres/gpu:1'}]
    assert qos_choice(['8gpus', '4gpus'], jobs, {'8gpus': 8, '4gpus': 4}) == ('4gpus', {'8gpus': 0, '4gpus': 3})
    assert qos_choice(['8gpus', '4gpus'], [], {'8gpus': 8, '4gpus': 4})[0] == '8gpus'
    full = jobs + [{'qos': '4gpus', 'state': 'RUNNING', 'gres': 'gres/gpu:4'}]
    assert qos_choice(['8gpus', '4gpus'], full, {'8gpus': 8, '4gpus': 4})[0] == '8gpus'   # both full: most room
    assert qos_choice(['a', 'b'], jobs, {'a': None, 'b': None})[0] == 'a'                  # caps unknown
    from ascg.tools import run_tools
    prof = json.loads(Path(os.environ['ASCG_COMPUTE']).read_text())
    prof['targets']['here'].update(scheduler='slurm', submit={'qos': ['8gpus', '4gpus'], 'max_jobs': 3})
    Path(os.environ['ASCG_COMPUTE']).write_text(json.dumps(prof))
    run_cli(['study', 'init', '--study', 'q'], capsys)
    s = Study('q')
    reg = s.load()
    reg['tasks']['k'] = {'key': 'k', 'target': 'here', 'status': 'registered', 'task_dir': '/t', 'ref': '/r',
                         'preflight_ok': True}
    s.save(reg)
    sent = {}
    def fake_remote(t, verb, p=None, timeout=0):
        if verb == 'code_digest':                        # the target runs this checkout's code
            from ascg.tools.compute_tools import _local_digest
            return {'digest': _local_digest(), 'deploy': None}
        if verb == 'slurm_jobs':
            return jobs
        if verb == 'qos_limits':
            return {'8gpus': 8, '4gpus': 4}
        sent.update(p)
        return {'id': '77', 'script': '/s', 'job_n': 1}
    monkeypatch.setattr(run_tools, 'remote', fake_remote)
    rc, out = run_cli(['run', 'submit', '--study', 'q', '--task', 'k'], capsys)
    assert rc == 0 and sent['submit']['qos'] == '4gpus' and Study('q').load()['tasks']['k']['job']['qos'] == '4gpus'


def test_restart_smaller_timestep_after_an_iteration0_gate_stop(env, capsys, monkeypatch, tmp_path):
    """A fit stopped on the bond gate before iteration 0 completed restarts from the
    Boltzmann-inverted start at a smaller timestep (also when it was escalated for that reason); the failed attempt
    is archived. Refused once an iteration has completed (rewind instead)."""
    from ascg import remote
    from ascg.tools import run_tools
    from ascg_core.ibi import pipeline as P
    run_cli(['study', 'init', '--study', 'r0'], capsys)
    s = Study('r0')
    reg = s.load()
    reg['tasks']['k'] = {'key': 'k', 'target': 'here', 'status': 'escalated', 'task_dir': '/t', 'ref': '/r',
                         'outcome': {'class': 'needs expert input', 'reasons': ['no action at iteration 0']}}
    s.save(reg)
    digest = {'completed': -1, 'failed': {'iteration': 0, 'reason': 'broken bonds'}, 'settings': {'dihedrals': None},
              'cost': {'spent_gpu_h': 0.02, 's_per_iteration': 60.0, 'fit_ps_per_iteration': 550.0}}
    monkeypatch.setattr(run_tools, 'remote', lambda t, verb, p, timeout=0: digest)
    sent = {}
    monkeypatch.setattr(run_tools, '_launch', lambda s_, reg_, key, mode, extra=None: sent.update(extra) or {'id': 1})
    rc, out = run_cli(['run', 'act', '--study', 'r0', '--task', 'k', '--action', 'restart_smaller_timestep',
                       '--dt-factor', '0.6', '--rationale', 'C4 at iteration 0', '--dry-run'], capsys)
    r = out['result']
    assert rc == 0 and [(st['stage'], st['from'], st['to']) for st in r['plan']] == [('stage1', 0, 15), ('stage2', 16, 42)]
    rc, out = run_cli(['run', 'act', '--study', 'r0', '--task', 'k', '--action', 'restart_smaller_timestep',
                       '--dt-factor', '0.6', '--rationale', 'C4 at iteration 0'], capsys)
    rec = Study('r0').load()['tasks']['k']
    assert rc == 0 and sent['action']['dt_factor'] == 0.6 and rec['outcome'] is None and len(rec['interventions']) == 1
    assert any(e['event'] == 'reopened' for e in Study('r0').events(0))
    digest.update(completed=3, failed={'iteration': 4})                       # an iteration completed: rewind instead
    reg = s.load(); reg['tasks']['k']['status'] = 'stopped_on_gate'; s.save(reg)
    rc, out = run_cli(['run', 'act', '--study', 'r0', '--task', 'k', '--action', 'restart_smaller_timestep',
                       '--dt-factor', '0.6', '--rationale', 'x'], capsys)
    assert rc == 2 and 'rewind instead' in out['error']
    # the job side: the failed attempt is archived and the fit restarts from iteration 0 at the smaller timestep
    td = tmp_path/'task'
    (td/'workflow').mkdir(parents=True)
    (td/'workflow'/'state.json').write_text(json.dumps({'completed': -1, 'history': [], 'failed': {'iteration': 0},
                                                        'settings': {'dihedrals': None, 'model': 'tdep'}}))
    (td/'cg'/'ibi'/'iter_000'/'run_T300').mkdir(parents=True)
    (td/'cg'/'ibi'/'iter_000'/'run_T300'/'run.json').write_text(json.dumps({'dt_ps': 0.0025}))
    calls = []
    monkeypatch.setattr(P, 'preflight', lambda ref, states=None: {'ok': True, 'n_dihedrals': 0, 'states': ['300']})
    monkeypatch.setattr(P, '_call', lambda log, stage, argv, retries: calls.append((stage, argv)))
    monkeypatch.setattr(P, 'converged', lambda *a, **k: (False, 'stop before validation'))
    remote.job({'task_dir': str(td), 'ref': str(tmp_path/'ref'), 'protocol': agentic(), 'mode': 'continue', 'task': 't',
                'action': {'action': 'restart_smaller_timestep', 'dt_factor': 0.6, 'target_iters': 42, 'rationale': 'x'}})
    opt = lambda argv, k: argv[argv.index(k) + 1] if k in argv else None
    assert [(st, opt(a, '--iters'), opt(a, '--dt')) for st, a in calls] == [('stage1', '15', '0.001500'),
                                                                            ('stage2', '42', '0.001500')]
    assert not (td/'cg'/'ibi').exists() and len(list((td/'cg').glob('ibi_restart_*'))) == 1
    assert len(list((td/'workflow').glob('state_restart_*.json'))) == 1



def test_deploy_doctor_and_code_check_on_a_local_target(tmp_path, monkeypatch, capsys):
    """compute deploy copies and verifies this checkout's code; doctor reports it; run submit refuses a target whose
    code differs."""
    monkeypatch.setenv('ASCG_STUDIES', str(tmp_path/'studies'))
    root = tmp_path/'host'
    prof = tmp_path/'compute.json'
    prof.write_text(json.dumps({'targets': {'box': {'transport': 'local', 'scheduler': 'none', 'root': str(root),
                                                      'python': sys.executable, 'env': f'export PYTHONPATH={root}/src',
                                                      'work_root': str(tmp_path/'work'), 'gpus': [0]}}}))
    monkeypatch.setenv('ASCG_COMPUTE', str(prof))
    monkeypatch.delenv('FASTMD_BINARY', raising=False)
    rc, out = run_cli(['compute', 'deploy', '--target', 'box'], capsys)
    assert rc == 0, out
    r = out['result']
    assert r['verified'] and r['copied'] and r['matches_local'] and (root/'src/ascg_core/mapping/spec.py').exists()
    assert json.loads((root/'.ascg_deploy.json').read_text())['code_digest'] == r['code_digest']
    rc, out = run_cli(['compute', 'doctor', '--target', 'box', '--no-engine'], capsys)
    checks = {c['check']: c for c in out['result']['checks']}
    assert checks['connection']['ok'] and checks['code']['ok']
    assert not checks['fastMD']['ok'] and 'fix' in checks['fastMD']
    f = root/'src/ascg/policy.py'
    f.write_text(f.read_text() + '\n# changed on the host\n')
    rc, out = run_cli(['compute', 'doctor', '--target', 'box', '--no-engine'], capsys)
    code = next(c for c in out['result']['checks'] if c['check'] == 'code')
    assert not code['ok'] and 'compute deploy' in code['fix']
    run_cli(['study', 'init', '--study', 'dc'], capsys)
    s = Study('dc')
    reg = s.load()
    reg['tasks']['k'] = {'key': 'k', 'target': 'box', 'status': 'registered', 'task_dir': str(tmp_path/'t'),
                         'ref': '/r', 'preflight_ok': True}
    s.save(reg)
    rc, out = run_cli(['run', 'submit', '--study', 'dc', '--task', 'k'], capsys)
    assert rc != 0 and out.get('refused_by_policy') and 'deploy it first' in out['error']


def test_fastmd_version_is_checked_by_ancestry(tmp_path, monkeypatch):
    """The doctor's fastMD version check accepts the pinned commit and any later one, not only a prefix match."""
    from ascg import remote
    src = tmp_path/'fastMD'
    (src/'build').mkdir(parents=True)
    (src/'build'/'fastmd').write_text('binary')
    git = lambda *a: subprocess.run(['git', '-C', str(src), '-c', 'user.name=t', '-c', 'user.email=t@t', *a],
                                    capture_output=True, text=True, check=True).stdout.strip()
    git('init', '-q')
    (src/'a.txt').write_text('1')
    git('add', 'a.txt')
    git('commit', '-q', '-m', 'pinned')
    pinned = git('rev-parse', 'HEAD')
    (src/'a.txt').write_text('2')
    git('commit', '-q', '-am', 'later')
    later = git('rev-parse', 'HEAD')
    monkeypatch.setenv('FASTMD_BINARY', str(src/'build'/'fastmd'))
    assert remote.fastmd_info({'ref': pinned[:7]})['includes_ref'] is True       # a later source
    assert remote.fastmd_info({'ref': later[:7]})['includes_ref'] is True
    assert remote.fastmd_info({'ref': 'deadbeef'})['includes_ref'] is None        # commit unknown there
    git('checkout', '-q', pinned)
    assert remote.fastmd_info({'ref': later[:7]})['includes_ref'] is False        # source older than the pin


def test_compute_add_writes_to_ascg_compute_even_before_it_exists(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('HOME', str(tmp_path/'home'))
    monkeypatch.chdir(tmp_path)
    prof = tmp_path/'new'/'compute.json'
    monkeypatch.setenv('ASCG_COMPUTE', str(prof))
    rc, out = run_cli(['compute', 'add', '--name', 'box', '--transport', 'local'], capsys)
    assert rc == 0 and out['result']['saved'] == str(prof)
    assert 'box' in json.loads(prof.read_text())['targets'] and not (tmp_path/'home'/'.config').exists()


def test_init_codex_writes_workspace_and_profile(tmp_path, monkeypatch, capsys):
    """`ascg init codex`: skills, AGENTS.md, rules and a Codex profile whose MCP server and pinned paths lie outside
    the agent's writable roots; refused when they do not."""
    tomllib = pytest.importorskip('tomllib')                                   # Python >= 3.11
    from ascg import agents
    ws, home, studies = tmp_path/'ws', tmp_path/'codex', tmp_path/'studies'
    monkeypatch.setenv('ASCG_COMPUTE', str(tmp_path/'compute.yaml'))
    monkeypatch.setenv('ASCG_LIBRARY', 'none')
    rc, out = run_cli(['init', 'codex', str(ws), '--studies', str(studies), '--codex-home', str(home)], capsys)
    assert rc == 1 and 'outside its writable roots' in out['error'] and 'studies' in out['error']   # tmp_path is $TMPDIR
    monkeypatch.setattr(agents, 'writable_roots', lambda w: [Path(w).resolve()])
    rc, out = run_cli(['init', 'codex', '--workspace', str(ws), '--studies', str(studies), '--codex-home', str(home)],
                      capsys)
    assert rc == 0, out
    assert (ws/'.agents'/'skills'/'ascg-study'/'SKILL.md').exists() and (ws/'.codex'/'rules'/'ascg.rules').exists()
    assert 'MCP tools' in (ws/'AGENTS.md').read_text()
    prof = tomllib.loads((home/'ascg.config.toml').read_text())
    srv = prof['mcp_servers']['ascg']
    assert srv['args'] == ['-m', 'ascg.mcp'] and srv['env']['ASCG_STUDIES'] == str(studies.resolve())
    assert srv['env']['ASCG_LIBRARY'] == 'none' and prof['projects'][str(ws.resolve())]['trust_level'] == 'trusted'
    assert prof['sandbox_mode'] == 'workspace-write' and out['result']['start'].startswith('codex -p ascg -C ')
    flags = agents.codex_flags(prof)
    assert '-c' in flags and any(f.startswith('mcp_servers.ascg.env.ASCG_STUDIES=') for f in flags)
    (ws/'AGENTS.md').write_text('my own notes\n')                               # a foreign AGENTS.md is kept
    rc, out = run_cli(['init', 'codex', str(ws), '--studies', str(studies), '--codex-home', str(home)], capsys)
    assert rc == 1 and 'not written by ascg' in out['error']
    rc, out = run_cli(['init', 'codex', str(ws), '--studies', str(ws/'studies'), '--codex-home', str(home)], capsys)
    assert rc == 1 and 'studies' in out['error']                                  # inside the workspace


def test_setup_actions_are_not_served_to_agents():
    """Amendment approval, compute setup and agent setup are a person's actions: not MCP tools, blocked by the guard."""
    from ascg import mcp
    served = set(mcp._tools())
    assert {'compute_list', 'compute_doctor', 'map_check'} <= served
    assert not {'amend_approve', 'compute_add', 'compute_deploy'} & served
    guard = Path(__file__).resolve().parents[2]/'plugin'/'ascg'/'hooks'/'guard.py'
    for cmd in ('ascg compute add --name x --transport local --env "sh x"', 'ascg compute deploy --target x',
                'ascg init codex /w', 'ascg amend approve --study s --id A01'):
        r = subprocess.run([sys.executable, str(guard), 'bash'], input=json.dumps({'tool_input': {'command': cmd}}),
                           capture_output=True, text=True)
        assert r.returncode == 2, cmd
    r = subprocess.run([sys.executable, str(guard), 'bash'], input=json.dumps({'tool_input': {'command': 'ascg compute list'}}),
                       capture_output=True, text=True)
    assert r.returncode == 0


def test_one_version_everywhere():
    import ascg
    import ascg_core
    root = Path(__file__).resolve().parents[2]
    plugin = json.loads((root/'plugin'/'ascg'/'.claude-plugin'/'plugin.json').read_text())
    market = json.loads((root/'.claude-plugin'/'marketplace.json').read_text())
    pyproject = (root/'pyproject.toml').read_text()
    v = ascg.__version__
    assert ascg_core.__version__ == v == plugin['version'] == market['plugins'][0]['version']
    assert f'version = "{v}"' in pyproject


def test_eval_checks_read_the_call_log(tmp_path, monkeypatch):
    """eval/score.py: task checks against the ASCG_CALL_LOG that every tool call writes."""
    import importlib.util
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('ascg_eval_score', root/'eval'/'score.py')
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    log = tmp_path/'calls.jsonl'
    monkeypatch.setenv('ASCG_CALL_LOG', str(log))
    monkeypatch.setenv('ASCG_LIBRARY', 'none')
    for smi in ('[CH3:1][CH2:1][CH2:2][CH2:3][CH2:3][CH3:3]', '[CH3:1][CH2:1][CH2:2][CH2:2][CH2:3][CH3:3]'):
        assert cli(['map', 'check', '--system_class', 'liquid', '--spec',
                    json.dumps({'format': 'annotated-smiles', 'smiles': smi, 'bead_types': {'1': 'E', '2': 'M', '3': 'E'}})]) == 0
    calls = score._jsonl(log)
    assert [c['tool'] for c in calls] == ['map check', 'map check'] and calls[0]['result']['ok']
    two_two_two = "sorted(len(b['heavy_atoms']) for b in r['species'][0]['beads']) == [2, 2, 2]"
    assert score.check({'type': 'result', 'tool': 'map check', 'expr': two_two_two}, calls, None)[0]
    assert not score.check({'type': 'result', 'tool': 'map check', 'expr': "len(r['species']) == 2"}, calls, None)[0]
    first_is_2_1_3 = "sorted(len(b['heavy_atoms']) for b in r['species'][0]['beads']) == [1, 2, 3]"
    assert not score.check({'type': 'result', 'tool': 'map check', 'expr': first_is_2_1_3}, calls, None)[0]
    assert score.check({'type': 'result', 'tool': 'map check', 'expr': first_is_2_1_3, 'which': 'any'}, calls, None)[0]
    assert score.check({'type': 'not_called', 'tool': ['map save', 'ref build']}, calls, None)[0]
    assert not score.check({'type': 'not_called', 'tool': ['map check']}, calls, None)[0]
    assert score.check({'type': 'called', 'tool': 'map check', 'ok': True}, calls, None)[0]
    assert score.check({'type': 'final', 'regex': 'densit'}, calls, 'The Density is too low')[0]


def test_report_task_fetches_the_model_files(env, capsys):
    run_cli(['study', 'init', '--study', 'rep'], capsys)
    td = env/'work'/'t'
    (td/'report'/'model_files'/'T300'/'tables').mkdir(parents=True)
    (td/'report'/'user_summary.md').write_text('# T: VALIDATED\n')
    (td/'report'/'model_files'/'README.md').write_text('model\n')
    (td/'report'/'model_files'/'T300'/'run_npt.conf').write_text('ensemble npt_nh\n')
    (td/'report'/'model_files'/'T300'/'tables'/'pair_A_A.table').write_text('1 0.3 1.0 0.0\n')
    s = Study('rep')
    reg = s.load()
    reg['tasks']['k'] = {'key': 'k', 'target': 'here', 'status': 'done', 'task_dir': str(td), 'ref': '/r'}
    s.save(reg)
    rc, out = run_cli(['report', 'task', '--study', 'rep', '--task', 'k', '--model-files'], capsys)
    assert rc == 0, out
    local = Path(out['result']['model_files'])
    assert (local/'T300'/'tables'/'pair_A_A.table').read_text() == '1 0.3 1.0 0.0\n'
    assert out['result']['user_summary'].startswith('# T: VALIDATED')
