"""Protocol v1: the deterministic per-task workflow (docs/PROTOCOL_v1.md). It runs a fit as a fixed script and
is the numerical core the agentic layer (the ascg tools and plugin) drives.

    python -m ascg_core.ibi.pipeline preflight --ref runs/cg_ref/<task> [--states 400,500] [--out f.json]
    python -m ascg_core.ibi.pipeline run --ref runs/cg_ref/<task> --task-dir runs/ibi_v1/<task> [--gpus inherit]
                                         [--states 450 | --states 400,400@1000]
    python -m ascg_core.ibi.pipeline aggregate --root runs/ibi_v1 --out runs/ibi_v1/report

`run`: preflight (deterministic compatibility gate) -> engine selfcheck -> stage 1 (pair-only IBI) -> stage 2
(pair:3,bond:1,angle:1,dihedral:1 with the virial pressure correction, whole cycles) -> convergence check on the
final in-loop NPT check (one bounded extension of three cycles if it fails) -> held-out validation -> figure ->
outcome class, model export and user-facing reports. Infrastructure errors (fastMD / GPU / file) are retried a
bounded number of times (the runner resumes after the last complete iteration); scientific failures (broken bonds,
gates missed after the budget) are reported, never retried.

States: the approved (temperature, pressure) points of the task (--states, N >= 1; default every state of the
reference; ibi.states). Every gate applies at every state; NPT runs use each state's pressure.

Outcome classes: validated (both gates at every state for every seed and chain size within
the band), needs expert input (gates pass, chain size outside the band), failed under budget (gates missed after
the budget, or a deterministic stop such as broken bonds), unsupported (preflight: engine or data limits).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from . import states as st

PROTOCOL = {
    'version': 'v1',
    'model': 'tdep', 'dihedrals': 'cbt',
    'stage1': {'iters': 15, 'schedule': 'pair', 'pressure_correction': 'none'},
    'stage2': {'iters': 42, 'schedule': 'pair:3,bond:1,angle:1,dihedral:1', 'pressure_correction': 'virial'},
    'npt_every': 6,
    'extension': {'iters': 18, 'max': 1, 'density_trigger': 0.04},
    'validation': {'seeds': 3, 'eq_ps': 200.0, 'prod_ps': 1000.0},
    'gates': {'density_rel_error_max': 0.05, 'rdf_nrmse_max': 0.15},
    'chain_band': {'rg_rel_error_max': 0.10},
    'infrastructure_retries': 2,
    'max_bead_types': 16,
}
_pretty = lambda T: st.parse_one(T).pretty()      # '400 K' / '400 K, 1000 atm'
CLASSES = ('validated', 'needs expert input', 'failed under budget', 'unsupported')


# ----------------------------------------------------------------------------------------------- preflight
def preflight(ref_dir, temperatures=None):
    # temperatures: the approved state points (ibi.states spec: '400,500', '400@1000', labels or a list)
    """Deterministic compatibility gate of one mapped reference for the approved states (default: every state of the
    reference): files, fastMD limits, topology / target coverage, reference statistics. Returns
    {'ok', 'checks': [{'name', 'ok', 'detail'}], 'temperatures', ...}."""
    from .system import load_system
    from . import dihedrals as dmod
    ref = Path(ref_dir)
    checks = []

    def check(name, ok, detail=''):
        checks.append({'name': name, 'ok': bool(ok), 'detail': str(detail)})
        return ok

    out = {'ref': str(ref.resolve()), 'task': ref.name}
    try:
        have = st.of_reference(ref)
        temps = st.parse(temperatures) if temperatures else have
    except (OSError, KeyError, ValueError) as e:
        check('state points', False, f'{type(e).__name__}: {e}')
        out.update(ok=False, checks=checks)
        return out
    out['temperatures'] = out['states'] = st.labels(temps)
    extra = [T for T in temps if T not in have]
    if not check('approved states have AA references', not extra,
                 f'no reference at {st.describe(extra)} (reference states: {st.describe(have)})' if extra else
                 f'{st.describe(temps)} (reference: {st.describe(have)})'):
        out.update(ok=False, checks=checks)
        return out
    need = ['topology.json', 'summary.json'] + st.missing_files(ref, temps)
    missing = [f for f in need if not (ref/f).exists()]
    if not check('reference files', not missing, f'missing {missing}' if missing else 'all present'):
        out.update(ok=False, checks=checks)
        return out
    try:
        system = load_system(ref)
    except Exception as e:                       # e.g. bonds missing from the exclusions
        check('topology', False, f'{type(e).__name__}: {e}')
        out.update(ok=False, checks=checks)
        return out
    summary = json.loads((ref/'summary.json').read_text())
    nt = len(system.type_names)
    check('bead types within fastMD limit', nt <= PROTOCOL['max_bead_types'], f'{nt} <= {PROTOCOL["max_bead_types"]}')
    check('beads', system.n > 0 and np.all(system.masses > 0), f'{system.n} beads, total {system.total_mass:.0f} amu')
    for T in temps:
        try:
            t = np.load(ref/f'targets_T{T}.npz')
            miss = ([f'rdf/{k}' for k in system.pair_keys if f'rdf/{k}' not in t.files]
                    + [f'bond/{k}' for k in system.bond_types if f'bond/{k}' not in t.files]
                    + [f'angle/{k}' for k in system.angle_types if f'angle/{k}' not in t.files])
            check(f'targets {_pretty(T)} cover every interaction', not miss, miss or 'complete')
            s = np.load(ref/f'start_T{T}.npz')
            check(f'start {_pretty(T)} matches topology', len(s['pos_nm']) == system.n, f"{len(s['pos_nm'])} vs {system.n}")
            with np.load(ref/f'mapped_T{T}.npz') as m:
                check(f'mapped trajectory {_pretty(T)} readable', m['pos_nm'].shape[1] == system.n, m['pos_nm'].shape)
        except Exception as e:                   # unreadable / corrupted reference files
            check(f'reference files {_pretty(T)} readable', False, f'{type(e).__name__}: {e}')
        rs = summary['states'].get(str(T), {})
        check(f'AA density {_pretty(T)}', rs.get('density_g_cm3', 0) > 0, f"{rs.get('density_g_cm3', 0):.4f} g/cm3")
        rmax = rs.get('r_max_over_half_box')
        check(f'RDF range within half box {_pretty(T)}', rmax is None or rmax <= 1.0, rmax)
    quads, dtype, names = dmod.enumerate_dihedrals(system)
    check('dihedrals', True, f'{len(quads)} of {len(names)} types' if len(quads) else 'none (dihedral steps dropped)')
    out.update(ok=all(c['ok'] for c in checks), checks=checks, n_beads=system.n, type_names=system.type_names,
               bond_types=system.bond_types, angle_types=system.angle_types, dihedral_types=names,
               n_dihedrals=int(len(quads)))
    return out


# ----------------------------------------------------------------------------------------------- run one task
class _Log:
    """workflow/provenance.jsonl (events), retries.jsonl, stage_timings.json."""

    def __init__(self, task_dir):
        self.wf = Path(task_dir)/'workflow'
        self.wf.mkdir(parents=True, exist_ok=True)
        self.timings_file = self.wf/'stage_timings.json'
        self.timings = json.loads(self.timings_file.read_text()) if self.timings_file.exists() else {}

    def event(self, what, **kw):
        with (self.wf/'provenance.jsonl').open('a') as f:
            f.write(json.dumps({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'event': what, **kw}, default=str) + '\n')

    def retry(self, **kw):
        with (self.wf/'retries.jsonl').open('a') as f:
            f.write(json.dumps({'time': time.strftime('%Y-%m-%d %H:%M:%S'), **kw}, default=str) + '\n')

    def timing(self, stage, seconds):
        self.timings[stage] = self.timings.get(stage, 0.0) + seconds
        self.timings_file.write_text(json.dumps(self.timings, indent=2) + '\n')


class ScientificFailure(RuntimeError):
    """A deterministic stop (broken bonds, ...): reported, never retried."""


def _call(log, stage, argv, retries):
    """runner.main(argv) with bounded retries of infrastructure errors; SystemExit('TASK FAILED ...') is scientific."""
    from . import runner
    for attempt in range(retries + 1):
        t0 = time.time()
        log.event('stage start', stage=stage, attempt=attempt + 1, argv=argv)
        try:
            runner.main(argv)
            log.timing(stage, time.time() - t0)
            log.event('stage done', stage=stage, seconds=round(time.time() - t0, 1))
            return
        except SystemExit as e:
            log.timing(stage, time.time() - t0)
            msg = str(e.code)
            if e.code in (None, 0):
                return
            log.event('stage failed', stage=stage, kind='scientific', message=msg)
            raise ScientificFailure(msg) from None
        except Exception as e:                       # fastMD crash, GPU / file errors: resume after the last iteration
            log.timing(stage, time.time() - t0)
            log.retry(stage=stage, attempt=attempt + 1, error=f'{type(e).__name__}: {e}',
                      traceback=traceback.format_exc(limit=3))
            log.event('stage error', stage=stage, kind='infrastructure', message=f'{type(e).__name__}: {e}')
            if attempt == retries:
                raise
            time.sleep(30)


def _last_npt(state):
    key = f'density_error_{st.of_run(state)[0]}'
    h = [e for e in state['history'] if key in e]
    return h[-1] if h else None


def converged(task_dir, trigger=PROTOCOL['extension']['density_trigger'], rdf_max=PROTOCOL['gates']['rdf_nrmse_max']):
    """(ok, reason) from the in-loop NPT check of the final iteration, at every state of the run."""
    state = json.loads((Path(task_dir)/'workflow'/'state.json').read_text())
    temps = st.of_run(state)
    e = _last_npt(state)
    if e is None or e['iteration'] != state['completed']:
        return False, 'no NPT check at the final iteration'
    bad = [f"{_pretty(T)} density {100 * e[f'density_error_{T}']:+.1f} %" for T in temps
           if abs(e[f'density_error_{T}']) > trigger]
    bad += [f"{_pretty(T)} worst RDF {max(e[f'rdf_{T}'].values()):.3f}" for T in temps
            if max(e[f'rdf_{T}'].values()) > rdf_max]
    return (not bad), ('; '.join(bad) if bad else f'in-loop NPT within the trigger at {st.describe(temps)}')


def classify(validation, chain_band=PROTOCOL['chain_band']['rg_rel_error_max']):
    """Outcome class and the reasons, from final/validation.json."""
    if validation is None:
        return 'failed under budget', ['no validation']
    if validation.get('failed'):
        return 'failed under budget', [validation['failed']]
    reasons, chain = [], []
    if not validation.get('states'):
        return 'failed under budget', ['no validated state']
    for T in st.labels(st.parse(list(validation['states']))):  # every state the tables were fitted to
        s = validation['states'][T]
        if abs(s['density_error_mean']) > PROTOCOL['gates']['density_rel_error_max'] or not s['pass_all_seeds']:
            reasons.append(f"{_pretty(T)}: density {100 * s['density_error_mean']:+.1f} %, worst RDF {s['rdf_nrmse_worst']:.3f}")
        rg = np.mean([m['rg_nm'] for m in s['seeds']])
        err = (rg - s['aa']['rg_nm']) / s['aa']['rg_nm']
        if abs(err) > chain_band:
            chain.append(f"{_pretty(T)}: Rg {rg:.3f} nm vs AA {s['aa']['rg_nm']:.3f} ({100 * err:+.0f} %)")
    if reasons:
        return 'failed under budget', reasons
    if chain:
        return 'needs expert input', chain
    return 'validated', []


def export_model(ref, task_dir):
    """report/model_files/T<T> per state: the state's tables, a data file (mapped-AA start), a ready NPT config."""
    from . import fastmd_io as fio
    from .system import load_start, load_system
    from . import dihedrals as dmod
    task = Path(task_dir)
    final = task/'cg'/'ibi'/'final'
    state = json.loads((task/'workflow'/'state.json').read_text())
    system = load_system(ref)
    style = dmod.style_of(state['settings'].get('dihedrals'))
    if style:
        system.add_dihedrals(style)
    out = task/'report'/'model_files'
    temps = st.of_run(state)
    for T in temps:
        d = out/f'T{T}'
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(final/f'tables_T{T}', d/'tables')
        pos, box = load_start(ref, T)
        system.write_data(d/'system.data', pos, box, velocities=fio.maxwell_velocities(system.masses, T, 1))
        conf = (final/f'holdout_T{T}'/'seed_0'/'run.conf').read_text()      # the held-out NPT input, relative paths
        (d/'run_npt.conf').write_text(conf.replace(str((final/f'tables_T{T}').resolve()) + '/', 'tables/'))
    model = ('single-state IBI' if len(temps) == 1 else 'multistate IBI, U(x, T) = A(x) + B(x) T')
    span = (f'(any temperature between {st.label(temps[0])} and {st.label(temps[-1])} K)' if len(temps) > 1 else
            f'(fitted at {st.label(temps[0])} K only: B = 0)')
    (out/'README.md').write_text(
        f'# {system.task.upper()} CG model (protocol {PROTOCOL["version"]}, {model}; states {st.describe(temps)})\n\n'
        'One directory per state point. Run with fastMD main b35e30a or later (`dihedral_table ... cbt`, NPT '
        'barostat fix):\n\n'
        f'    cd T{temps[0]} && fastmd run_npt.conf\n\n'
        'Units: nm, ps, amu, kJ/mol; thermostat/barostat targets are k_B T and pressure in kJ/mol(/nm^3). Tables: '
        'pair (energy, force), bond (energy, force), angle (degrees, force per degree), dihedral (degrees, CBT). '
        'Bond type `EXCL13` is a zero-energy bond that excludes 1-3 pairs from the pair potential. '
        f'A and B of every table {span} are in ../../cg/ibi/final/tables/potentials_tdep.json.\n')
    return out


def _fs(dt_ps):
    return f'{1000 * dt_ps:.3f}'.rstrip('0').rstrip('.')


def fit_history(state):
    """The run's invocations (iterations, schedule, pressure correction, damping, timestep as set) and rewinds, for the
    reports: a continuation after a rewind or with a smaller timestep is part of how the model was made."""
    inv = state.get('invocations') or []
    rows = []
    for i, v in enumerate(inv):
        s = v.get('settings') or {}
        end = inv[i + 1].get('start_iteration', 0) - 1 if i + 1 < len(inv) else state.get('completed')
        rows.append({'from': v.get('start_iteration'), 'to': end, 'schedule': s.get('schedule'),
                     'pressure_correction': s.get('pressure_correction'), 'lam': s.get('lam'),
                     'dt_ps': s.get('dt')})
    rew = [{'time': r.get('time'), 'from': r.get('from'), 'to': r.get('to'), 'reason': r.get('reason'),
            'failed': (r.get('failed') or {}).get('reason')} for r in state.get('rewinds', [])]
    return {'invocations': rows, 'rewinds': rew}


def write_reports(ref, task_dir, outcome):
    """report/outcome.json, costs.json, user_summary.md (plain language), technical_report.md."""
    task = Path(task_dir)
    rep = task/'report'
    rep.mkdir(parents=True, exist_ok=True)
    log = _Log(task)
    stp = task/'workflow'/'state.json'
    state = json.loads(stp.read_text()) if stp.exists() else {'history': [], 'completed': -1}
    val_file = task/'cg'/'ibi'/'final'/'validation.json'
    val = json.loads(val_file.read_text()) if val_file.exists() else None
    retries = (task/'workflow'/'retries.jsonl').read_text().splitlines() if (task/'workflow'/'retries.jsonl').exists() else []
    gpu_s = sum(log.timings.get(k, 0.0) for k in ('stage1', 'stage2', 'extension', 'validate'))
    costs = {'gpu_hours': gpu_s / 3600, 'stage_seconds': log.timings, 'iterations': state.get('completed', -1),
             'infrastructure_retries': len(retries)}
    (rep/'costs.json').write_text(json.dumps(costs, indent=2) + '\n')
    (rep/'outcome.json').write_text(json.dumps({**outcome, 'protocol': PROTOCOL, 'costs': costs}, indent=2, default=str) + '\n')
    lines = [f"# {outcome['task'].upper()}: {outcome['class'].upper()}", '']
    temps = (st.parse(list(val['states'])) if val and val.get('states') else
             st.of_run(state) if state.get('settings') else ())
    at = st.describe(temps) if temps else 'the approved state points'
    plain = {'validated': f'The coarse-grained model reproduces the atomistic density and structure at {at} within '
                          'the pre-registered limits.',
             'needs expert input': 'The model meets the density and structure limits, but the chains are noticeably '
                                   'larger or smaller than in the atomistic reference; please review before use.',
             'failed under budget': 'No model met the limits within the fixed compute budget.',
             'unsupported': 'This task could not be run with the current engine or reference data.'}
    lines += [plain[outcome['class']], '']
    for r in outcome.get('reasons', []):
        lines.append(f'- {r}')
    if val and val.get('states'):
        lines += ['', '| | ' + ' | '.join(f'{_pretty(T)}' for T in temps) + ' |', '|---|' + '---|' * len(temps)]
        s = {str(T): val['states'][str(T)] for T in temps}
        lines.append('| density error (limit 5 %) | ' + ' | '.join(f"{100 * s[T]['density_error_mean']:+.1f} %" for T in s) + ' |')
        lines.append('| worst pair RDF NRMSE (limit 0.15) | ' + ' | '.join(f"{s[T]['rdf_nrmse_worst']:.3f}" for T in s) + ' |')
        lines.append('| chain Rg, CG / AA (nm) | ' + ' | '.join(
            f"{np.mean([m['rg_nm'] for m in s[T]['seeds']]):.3f} / {s[T]['aa']['rg_nm']:.3f}" for T in s) + ' |')
        lines += ['', f"Held-out check: {val['seeds']} independent NPT runs per state that were not used in fitting."]
    hist = fit_history(state)
    if hist['rewinds'] or len(hist['invocations']) > 1:
        lines += ['', 'Fitting history:']
        for r in hist['rewinds']:
            lines.append(f"- rewound to iteration {r['to']}: {(r['reason'] or '')[:200]}")
        for v in hist['invocations']:
            dt = 'the default timestep' if v['dt_ps'] is None else f"timestep {_fs(v['dt_ps'])} fs"
            lines.append(f"- iterations {v['from']}-{v['to']}: {v['schedule']}, pressure correction "
                         f"{v['pressure_correction']}, {dt}")
        if val and val.get('dt_ps'):
            lines.append(f"- held-out validation and the model files: timestep {_fs(val['dt_ps'])} fs")
    lines += ['', f"Compute: {costs['gpu_hours']:.2f} GPU-hours, {costs['iterations']} IBI iterations, "
                  f"{costs['infrastructure_retries']} automatic retries.",
              'Model files: report/model_files (tables, data file and a ready fastMD NPT input per temperature); '
              'figure: report/structure_density.png; details: report/technical_report.md.']
    (rep/'user_summary.md').write_text('\n'.join(lines) + '\n')
    tech = [f"# {outcome['task'].upper()} technical report (protocol {PROTOCOL['version']})", '',
            '## Protocol', '```json', json.dumps(PROTOCOL, indent=2), '```', '',
            '## Outcome', '```json', json.dumps(outcome, indent=2, default=str), '```', '',
            '## Fitting history (invocations and rewinds)', '```json', json.dumps(fit_history(state), indent=2,
                                                                                  default=str), '```', '',
            '## Engine', '```json', json.dumps(state.get('fastmd', {}), indent=2), '```', '',
            '## Costs', '```json', json.dumps(costs, indent=2), '```']
    if val:
        tech += ['', '## Validation (per seed)', '```json', json.dumps(val['states'], indent=1)[:20000], '```']
    (rep/'technical_report.md').write_text('\n'.join(tech) + '\n')


def run_task(ref, task_dir, *, gpus='inherit', ibi_extra=(), validate_extra=(), protocol=PROTOCOL, log_print=print,
             temperatures=None):
    """The protocol-v1 workflow for one task at its approved states (default: every state of the reference);
    returns the outcome dict (also in report/outcome.json)."""
    ref, task = Path(ref), Path(task_dir)
    log = _Log(task)
    outcome = {'task': ref.name, 'ref': str(ref.resolve()), 'task_dir': str(task.resolve())}
    log.event('task start', protocol=protocol['version'], temperatures=temperatures)
    t0 = time.time()
    pf = preflight(ref, temperatures)
    (task/'workflow'/'preflight.json').write_text(json.dumps(pf, indent=2) + '\n')
    log.timing('preflight', time.time() - t0)
    if not pf['ok']:
        outcome.update({'class': 'unsupported', 'reasons': [f"{c['name']}: {c['detail']}" for c in pf['checks'] if not c['ok']]})
        write_reports(ref, task, outcome)
        log.event('task done', **outcome)
        return outcome
    from .selfcheck import main as selfcheck
    t1 = time.time()
    if selfcheck([] if pf['n_dihedrals'] else ['--no-dihedrals']) != 0:      # engine gate: infrastructure, not science
        log.event('selfcheck failed')
        raise RuntimeError('fastMD selfcheck failed: wrong or broken fastMD build on this host')
    log.timing('selfcheck', time.time() - t1)
    common = ['--ref', str(ref), '--task-dir', str(task), '--model', protocol['model'], '--dihedrals',
              protocol['dihedrals'] if pf['n_dihedrals'] else 'none', '--npt-every', str(protocol['npt_every']),
              '--states', ','.join(pf['states']), '--gpus', gpus, *ibi_extra]
    outcome['states'] = pf['states']
    s1, s2, ext = protocol['stage1'], protocol['stage2'], protocol['extension']
    retries = protocol['infrastructure_retries']
    try:
        _call(log, 'stage1', ['ibi', *common, '--iters', str(s1['iters']), '--schedule', s1['schedule'],
                              '--pressure-correction', s1['pressure_correction']], retries)
        n2 = s2['iters']
        _call(log, 'stage2', ['ibi', *common, '--iters', str(n2), '--schedule', s2['schedule'],
                              '--pressure-correction', s2['pressure_correction']], retries)
        state = json.loads((task/'workflow'/'state.json').read_text())
        n2 = state['completed']
        for k in range(ext['max']):
            ok, why = converged(task, ext['density_trigger'])
            log.event('convergence check', iteration=n2, ok=ok, reason=why)
            log_print(f'convergence check at iteration {n2}: {why}')
            if ok:
                break
            n2 += ext['iters']
            _call(log, 'extension', ['ibi', *common, '--iters', str(n2), '--schedule', s2['schedule'],
                                     '--pressure-correction', s2['pressure_correction']], retries)
            n2 = json.loads((task/'workflow'/'state.json').read_text())['completed']
        val_file = task/'cg'/'ibi'/'final'/'validation.json'
        state = json.loads((task/'workflow'/'state.json').read_text())
        if not (val_file.exists() and json.loads(val_file.read_text())['iteration'] == state['completed']):
            v = protocol['validation']
            try:
                _call(log, 'validate', ['validate', '--ref', str(ref), '--task-dir', str(task), '--seeds', str(v['seeds']),
                                        '--eq-ps', str(v['eq_ps']), '--prod-ps', str(v['prod_ps']), '--gpus', gpus,
                                        *validate_extra], retries)
            except ScientificFailure:
                pass                                   # broken bonds in validation: recorded in validation.json
        val = json.loads(val_file.read_text()) if val_file.exists() else None
        cls, reasons = classify(val)
        outcome.update({'class': cls, 'reasons': reasons, 'iterations': state['completed']})
    except ScientificFailure as e:
        outcome.update({'class': 'failed under budget', 'reasons': [str(e)]})
    outcome['wall_clock_h'] = (time.time() - t0) / 3600
    finalize(ref, task, outcome, log)
    log.event('task done', **outcome)
    log_print(f"OUTCOME {outcome['task']}: {outcome['class']}" + (f" ({'; '.join(outcome['reasons'])})"
                                                                  if outcome.get('reasons') else ''))
    return outcome


def finalize(ref, task, outcome, log):
    """Figure, model export and reports of a classified task (reporting problems never change the outcome)."""
    try:
        from .plots import task_figure
        if (Path(task)/'cg'/'ibi'/'final'/'validation.json').exists():
            task_figure(ref, task, Path(task)/'report'/'structure_density.png')
            export_model(ref, task)
    except Exception as e:
        log.event('report error', message=f'{type(e).__name__}: {e}')
    write_reports(ref, task, outcome)


def rescore(ref, task_dir, gpus='inherit', reason='protocol v1.1: broken-bond persistence rule'):
    """Re-analyse a finished task's held-out runs under an amended analysis rule (no new sampling); the previous
    validation and outcome are kept as validation_<old>.json / outcome_<old>.json and both are reported."""
    ref, task = Path(ref), Path(task_dir)
    log = _Log(task)
    final, rep = task/'cg'/'ibi'/'final', task/'report'
    old_v, old_o = json.loads((final/'validation.json').read_text()), json.loads((rep/'outcome.json').read_text())
    (final/'validation_v1.json').write_text(json.dumps(old_v, indent=2) + '\n')
    (rep/'outcome_v1.json').write_text(json.dumps(old_o, indent=2) + '\n')
    v = PROTOCOL['validation']
    _call(log, 'rescore', ['validate', '--ref', str(ref), '--task-dir', str(task), '--seeds', str(v['seeds']), '--eq-ps',
                           str(v['eq_ps']), '--prod-ps', str(v['prod_ps']), '--gpus', gpus, '--rescore'], 0)
    cls, reasons = classify(json.loads((final/'validation.json').read_text()))
    outcome = {k: old_o[k] for k in ('task', 'ref', 'task_dir', 'iterations', 'wall_clock_h') if k in old_o}
    outcome.update({'class': cls, 'reasons': reasons, 'amendment': reason,
                    'v1_outcome': {'class': old_o['class'], 'reasons': old_o.get('reasons', [])}})
    finalize(ref, task, outcome, log)
    log.event('rescored', **outcome)
    return outcome


# ----------------------------------------------------------------------------------------------- aggregate
def aggregate(root, out):
    """Benchmark table and figures over runs/<root>/<task>/report/outcome.json."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ..plotting import GRID, INK, INK2, style as _style
    from .plots import state_colors
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(root.glob('*/report/outcome.json')):
        o = json.loads(f.read_text())
        vf = f.parent.parent/'cg'/'ibi'/'final'/'validation.json'
        v = json.loads(vf.read_text()) if vf.exists() else None
        r = {'task': o['task'], 'class': o['class'], 'reasons': o.get('reasons', []), 'gpu_hours': o['costs']['gpu_hours'],
             'v1_class': ((o.get('v1_outcome') or {}).get('class') or (json.loads((f.parent/'outcome_v1.json').read_text())['class']
                                                                       if (f.parent/'outcome_v1.json').exists() else None)),
             'amendment': o.get('amendment'),
             'wall_clock_h': o.get('wall_clock_h'), 'iterations': o['costs']['iterations'],
             'retries': o['costs']['infrastructure_retries']}
        if v and 'states' in v and v['states']:
            r['states'] = st.labels(st.parse(list(v['states'])))
            for T in r['states']:
                s = v['states'][T]
                r[f'density_{T}'] = s['density_error_mean']
                r[f'rdf_{T}'] = s['rdf_nrmse_worst']
                rg = np.mean([m['rg_nm'] for m in s['seeds']])
                r[f'rg_{T}'] = (rg - s['aa']['rg_nm']) / s['aa']['rg_nm']
        elif o.get('states') or o.get('temperatures'):
            r['states'] = st.labels(st.parse(o.get('states') or o['temperatures']))
        rows.append(r)
    (out/'benchmark.json').write_text(json.dumps(rows, indent=2) + '\n')
    counts = {c: sum(r['class'] == c for r in rows) for c in CLASSES}
    md = [f'# Protocol {PROTOCOL["version"]} benchmark: {len(rows)} tasks', '',
          ', '.join(f'{c}: {n}' for c, n in counts.items()), '',
          '| task | class | states (K) | density error | worst RDF | Rg error | GPU-h | iterations |',
          '|---|---|---|---|---|---|---|---|']
    f = lambda r, k, fmt: fmt.format(r[k]) if k in r else '-'
    per = lambda r, key, fmt: ' / '.join(f(r, f'{key}_{T}', fmt) for T in r.get('states', [])) or '-'
    for r in rows:
        cls = r['class'] + (f" (v1 rule: {r['v1_class']})" if r.get('v1_class') and r['v1_class'] != r['class'] else '')
        md.append(f"| {r['task']} | {cls} | {' / '.join(r.get('states', [])) or '-'} | "
                  f"{per(r, 'density', '{:+.1%}')} | {per(r, 'rdf', '{:.3f}')} | {per(r, 'rg', '{:+.0%}')} | "
                  f"{r['gpu_hours']:.2f} | {r['iterations']} |")
    ok = [r for r in rows if r['class'] in ('validated', 'needs expert input')]
    if ok:
        wc = np.array([r['wall_clock_h'] for r in ok if r.get('wall_clock_h') is not None])
        if len(wc):
            md += ['', f'Wall-clock to a model meeting the gates: median {np.median(wc):.2f} h, p90 '
                       f'{np.percentile(wc, 90):.2f} h; GPU-hours total {sum(r["gpu_hours"] for r in rows):.1f}.']
    (out/'benchmark.md').write_text('\n'.join(md) + '\n')
    # figure: one panel per measure; every task on x in catalogue order (level), 400 / 500 K as two series, gate
    # lines; tasks without a validated model are marked instead of left blank
    try:
        cat = json.loads((Path(__file__).resolve().parents[3]/'benchmarks'/'ascg_library.json').read_text())['systems']
        order = {t['key']: (t['id'],) for t in cat}
    except OSError:
        order = {}
    rows_sorted = sorted(rows, key=lambda r: order.get(r['task'], ('Z', 9))[0])
    if rows_sorted:
        _style(plt)
        fig, ax = plt.subplots(3, 1, figsize=(max(8, 0.36 * len(rows_sorted) + 2), 8.6), sharex=True)
        x = np.arange(len(rows_sorted))
        mark = {'needs expert input': 'expert', 'failed under budget': 'failed', 'unsupported': 'unsupported'}
        all_T = st.parse({T for r in rows_sorted for T in r.get('states', [])} or ['400'])
        colors = state_colors(all_T)
        w = 0.76 / max(len(all_T), 1)
        for i, T in enumerate(all_T):
            off = (i - (len(all_T) - 1) / 2) * w
            for a, key, scale in ((ax[0], 'density', 100), (ax[1], 'rdf', 1), (ax[2], 'rg', 100)):
                v = [scale * r[f'{key}_{T}'] if f'{key}_{T}' in r else np.nan for r in rows_sorted]
                a.bar(x + off, v, w * 0.95, color=colors[T], label=f'{_pretty(T)}' if a is ax[0] else None)
        for a, lim, title in ((ax[0], 5, 'held-out NPT density error vs AA (%), gate +-5 %'),
                              (ax[1], 0.15, 'worst pair RDF NRMSE over seeds, gate 0.15'),
                              (ax[2], 100 * PROTOCOL['chain_band']['rg_rel_error_max'],
                               'chain Rg error vs AA (%), band +-10 % (outside: needs expert input)')):
            a.axhline(lim, color=INK2, lw=0.9, ls=(0, (4, 3)))
            if a is not ax[1]:
                a.axhline(-lim, color=INK2, lw=0.9, ls=(0, (4, 3)))
                a.axhline(0, color=INK, lw=0.6)
            a.set_title(title, loc='left', fontsize=9)
            a.grid(axis='y', color=GRID, lw=0.6)
        for j, r in enumerate(rows_sorted):
            if r['class'] in mark and not any(k.startswith('density_') for k in r):   # no model: say why it is empty
                for a in ax:
                    lo, hi = a.get_ylim()
                    a.text(j, lo + 0.04 * (hi - lo), mark[r['class']], rotation=90, ha='center', va='bottom',
                           fontsize=7, color=INK2)
        ax[0].legend(frameon=False, fontsize=8, ncol=min(len(all_T), 6) or 1, loc='upper right')
        labels = [f"{r['task'].upper()}" + ('*' if r.get('v1_class') and r['v1_class'] != r['class'] else '')
                  + (f" ({mark[r['class']]})" if r['class'] in mark else '') for r in rows_sorted]
        ax[2].set_xticks(x, labels, rotation=60, ha='right')
        counts_txt = ', '.join(f'{n} {c}' for c, n in counts.items() if n)
        fig.suptitle(f'Protocol {PROTOCOL["version"]}: {len(rows)} tasks, {counts_txt}\n* outcome changed by a '
                     'protocol amendment (see benchmark.md)', x=0.01, ha='left', fontsize=10, color=INK)
        fig.tight_layout()
        fig.savefig(out/'benchmark.png', dpi=160)
        plt.close(fig)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('preflight')
    p.add_argument('--ref', required=True, nargs='+')
    p.add_argument('--states', '--temperatures', dest='states', default=None,
                   help='approved state points, e.g. 400,500 or 400@1000 (K, atm; default: every state of the reference)')
    p.add_argument('--out', default=None)
    r = sub.add_parser('run')
    r.add_argument('--ref', required=True)
    r.add_argument('--task-dir', required=True)
    r.add_argument('--gpus', default='inherit')
    r.add_argument('--states', '--temperatures', dest='states', default=None,
                   help='approved state points, e.g. 400,500 or 450 or 400,400@1000 (default: every '
                   'state of the reference)')
    r.add_argument('--ibi-extra', default='', help='extra runner ibi options (tests only)')
    r.add_argument('--validate-extra', default='', help='extra runner validate options (tests only)')
    r.add_argument('--override', default='', help='JSON merged into the protocol (tests only), e.g. '
                   '\'{"stage1": {"iters": 2}}\'')
    rs = sub.add_parser('rescore', help='re-analyse held-out runs under an amended rule (no sampling)')
    rs.add_argument('--ref', required=True)
    rs.add_argument('--task-dir', required=True)
    g = sub.add_parser('aggregate')
    g.add_argument('--root', required=True)
    g.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'preflight':
        res = [preflight(x, a.states) for x in a.ref]
        for x in res:
            bad = [f"{c['name']} ({c['detail']})" for c in x['checks'] if not c['ok']]
            print(f"{x['task']:14s} {'OK ' if x['ok'] else 'FAIL'} types {len(x.get('type_names', []))} "
                  f"dihedrals {x.get('n_dihedrals', '-')} {('; '.join(bad)) if bad else ''}")
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=2) + '\n')
        return 0 if all(x['ok'] for x in res) else 1
    if a.cmd == 'rescore':
        o = rescore(a.ref, a.task_dir)
        print(f"RESCORED {o['task']}: {o['class']} (v1: {o['v1_outcome']['class']})")
        return 0
    if a.cmd == 'run':
        proto = json.loads(json.dumps(PROTOCOL))
        for k, v in (json.loads(a.override) if a.override else {}).items():
            proto[k] = {**proto[k], **v} if isinstance(v, dict) else v
        o = run_task(a.ref, a.task_dir, gpus=a.gpus, ibi_extra=a.ibi_extra.split(), validate_extra=a.validate_extra.split(),
                     protocol=proto, temperatures=a.states)
        return 0 if o['class'] in CLASSES else 1
    aggregate(a.root, a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
