"""aa build / submit / status / map / cancel: the all-atom reference of a study's approved system, on one compute
target, through the numerical core's commands (AutoPoly v2.0 build, OPLS audit, GROMACS protocol, map-aa).

A campaign lives in the study registry under reg['aa'][key] and on the target under <aa root>/<study>/<key>. The
target's `aa` settings (the user's compute profile) give the GROMACS environment line and the batch settings of each
stage; defaults below. AA runs need the user's approval with a cost estimate (`aa submit --dry-run` first) and stay
under the target's core-hour cap per task."""
import hashlib
import json
from pathlib import Path

from ..compute import remote, target
from ..registry import P, PolicyError, ToolError, tool
from ..study import Study, now

STAGE_DEFAULTS = {'build': {'cpus': 2, 'mem': '8G', 'time': '02:00:00'},
                  'md': {'cpus': 16, 'mem': '16G', 'time': '2-00:00:00'},
                  'gpu': {'cpus': 8, 'mem': '16G', 'time': '16:00:00', 'gres': 'gpu:1'},
                  'map': {'cpus': 2, 'mem': '24G', 'time': '12:00:00'}}
MAX_CORE_H = 2000                      # per task, unless the target's aa.max_core_h says otherwise
NS_PER_DAY_16_CORES = 7.6e5            # x 1/atoms: PMMA 7600 atoms ~100, acetone 10000 ~84, toluene 15000 ~56 ns/day
GPU_SPEEDUP = 4.0                      # rough; the measured ns/day replaces the estimate once running
DIRECT_EQ_SPEED = 0.6                  # liquids: compressing the dilute box runs slower (n-pentane: 28 vs 56 ns/day)
TERMINAL_EVENTS = ('aa_built', 'aa_build_failed', 'aa_md_done', 'aa_md_failed', 'aa_mapped', 'aa_map_failed')


def _aa_target(name):
    t = target(name)
    if not t.get('aa'):
        raise PolicyError(f'target {name!r} has no AA settings: the user adds them (ascg compute add ... --aa '
                          '\'{"env": "module load gromacs", "gmx": "gmx_mpi", "md": {"qos": ..., "partition": ...}}\'; '
                          'see the compute guide)')
    return t


def _stage_settings(t, stage):
    return {**STAGE_DEFAULTS[stage], **(t['aa'].get(stage) or {})}


def _campaign(s, reg, key):
    c = (reg.get('aa') or {}).get(key)
    if c is None:
        raise ToolError(f'no AA campaign for {key!r} in study {s.name!r} (aa build)')
    return c


def _common(t, c, study, key):
    return {'dir': c['dir'], 'key': key, 'study': study, 'scheduler': t['scheduler'], 'env': t.get('env', ''),
            'python': t.get('python', 'python')}


def _active(c):
    return [j for j in c.get('jobs', []) if j.get('state', {}).get('status') in (None, 'submitted', 'running')]


def _hours(time_str):
    d, _, hms = str(time_str).rpartition('-')
    h, m, *_ = (hms.split(':') + ['0', '0'])
    return (int(d) if d else 0) * 24 + int(h) + int(m) / 60


def _lengths(cls, states, eq_ns, prod_ns, pre_ns):
    """(equilibration ns, [branch ns]) of the protocol prepare-gmx-protocol writes."""
    from ascg_core.aa import gmx_protocol as G
    from ascg_core.ibi import states as st
    temps = st.parse(states)
    soft = sum(p['nsteps'] * p.get('dt', 0) for _, p, _ in G.soft_core_steps() if p.get('integrator') == 'md')
    if cls == 'polymer':
        eq = sum(p['nsteps'] * p['dt'] for _, p, _ in G.eq21_steps(temperature_K=max(x.T for x in temps))) / 1000
    else:
        eq = sum(p['nsteps'] * p['dt'] for _, p, _ in G.direct_steps(temperature_K=max(x.T for x in temps))) / 1000
    return eq + soft / 1000, [pre_ns + eq_ns + prod_ns for _ in temps]


def _estimate(t, atoms, mode, eq_ns, branches, equilibration='eq21'):
    md = _stage_settings(t, 'gpu' if mode == 'gpu' else 'md')
    cpus = int(md['cpus'])
    ns_day = NS_PER_DAY_16_CORES / max(atoms, 1) * (GPU_SPEEDUP if mode == 'gpu' else cpus / 16)
    eq_h = 24 * eq_ns / (ns_day * (DIRECT_EQ_SPEED if equilibration == 'direct' else 1.0))
    br_h = [24 * b / ns_day for b in branches]
    parallel = t['scheduler'] == 'slurm' and mode != 'gpu'
    return {'atoms': atoms, 'ns_per_day_estimate': round(ns_day, 1), 'equilibration_ns': round(eq_ns, 2),
            'branch_ns': branches, 'wall_h_estimate': round(eq_h + (max(br_h) if parallel else sum(br_h)), 1),
            'core_h_estimate': round(cpus * (eq_h + sum(br_h)), 1), 'cpus_per_job': cpus, 'mode': mode,
            'note': 'from the system size and earlier runs; aa status reports the measured ns/day'}


def _launch(t, c, study, key, stage, **kw):
    p = {**_common(t, c, study, key), 'stage': stage, **kw}
    j = remote(t, 'aa_launch', p, timeout=300)
    j['state'] = {'status': 'submitted'}
    c.setdefault('jobs', []).append(j)
    return j


@tool('aa', 'build', 'Build the all-atom system of a study\'s APPROVED system (system save + system approve) on a '
      'target: AutoPoly v2.0 OPLS-AA build, charge neutralization (recorded fixes only), typing / charge audit and, '
      'with a mapping, the CG topology, as one job. No simulation. Then aa status; aa submit after a clean audit.', {
          'study': P(required=True), 'task': P(required=True, help='the system key'),
          'target': P(required=True, help='compute target with AA settings (AutoPoly, GROMACS)'),
          'force': P('boolean', default=False, help='rebuild over a finished build (its runs are then invalid)')},
      mutating=True)
def aa_build(study, task, force, **kw):
    s = Study(study)
    reg = s.load()
    sysrec = s.system(task)
    if sysrec is None:
        raise PolicyError(f'no system {task!r} in study {study!r}: describe it first (system check / save / approve)')
    if sysrec['status'] != 'approved':
        raise PolicyError(f'system {task!r} is not approved: the user approves it first (system approve)')
    t = _aa_target(kw['target'])
    c = (reg.get('aa') or {}).get(task)
    if c and _active(c):
        raise PolicyError(f'AA jobs of {task!r} are still active: aa status, or aa cancel')
    if c and c.get('phase') not in (None, 'build_failed') and not force:
        raise PolicyError(f'{task!r} already has a build ({c.get("phase")}); --force rebuilds and invalidates its runs')
    root = (t['aa'].get('work_root') or t['work_root'].rstrip('/') + '/aa').rstrip('/')
    files = {'recipes.json': Path(sysrec['files']['recipes']).read_text(),
             'system.json': json.dumps(sysrec['spec'], indent=1) + '\n'}
    if sysrec['files'].get('library'):
        files['library.json'] = Path(sysrec['files']['library']).read_text()
    c = {'target': t['name'], 'dir': f'{root}/{study}/{task}', 'system_class': sysrec['system_class'],
         'system_md5': hashlib.md5(json.dumps(sysrec['spec'], sort_keys=True).encode()).hexdigest()[:12],
         'jobs': [], 'phase': 'building', 'created': now(), 'approvals': (c or {}).get('approvals', [])}
    staged = remote(t, 'aa_stage', {'dir': c['dir'], 'files': files}, timeout=120)
    bad = [f for f, text in files.items() if staged['md5'].get(f) != hashlib.md5(text.encode()).hexdigest()]
    if bad:
        raise ToolError(f'inputs differ after the copy: {bad}')
    b = _stage_settings(t, 'build')
    j = _launch(t, c, study, task, 'build', submit=b, cpus=b['cpus'])
    reg.setdefault('aa', {})[task] = c
    s.event(reg, 'aa_build_submitted', task, {'target': t['name'], 'job': j.get('id') or j.get('pid')})
    s.save(reg)
    return {'task': task, 'target': t['name'], 'dir': c['dir'], 'job': j,
            'next': 'aa status (the build takes minutes); read the audit before anything else'}


def _phase(c, st):
    """Campaign phase from the remote state."""
    jobs = {j['name']: j for j in st['jobs']}
    last = {}
    for j in c.get('jobs', []):
        last[j['stage']] = jobs.get(j['name'], {})
    def failed(stage):
        return last.get(stage, {}).get('status') in ('failed', 'lost')
    if st.get('reference') and last.get('map', {}).get('status') == 'done':
        return 'mapped'
    if 'map' in last:
        return 'map_failed' if failed('map') else 'mapping'
    prog = (st.get('protocol') or {}).get('progress') or {}
    if 'md' in last:
        md = [jobs.get(j['name'], {}) for j in c['jobs'] if j['stage'] == 'md']
        if prog and all(v['next'] is None for v in prog.values()):
            return 'md_done'
        if any(x.get('status') in ('failed', 'lost') for x in md) and not any(x.get('status') in ('submitted', 'running')
                                                                              for x in md):
            return 'md_failed'
        return 'md_running' if any(x.get('status') == 'running' for x in md) else 'md_queued'
    if failed('build'):
        return 'build_failed'
    if last.get('build', {}).get('status') == 'done' and st.get('build') and 'error' not in st['build']:
        return 'built'
    return 'building'


def refresh_aa(s, reg, keys=None):
    """Query each campaign's target once, update job states and phases, emit events on phase changes."""
    out = {}
    for key, c in (reg.get('aa') or {}).items():
        if keys and key not in keys:
            continue
        t = target(c['target'])
        try:
            st = remote(t, 'aa_state', {'dir': c['dir'], 'key': key, 'jobs': c.get('jobs', [])}, timeout=300)
        except ToolError as e:
            out[key] = {'error': str(e)[:300], 'phase': c.get('phase')}
            continue
        by_name = {j['name']: j for j in st['jobs']}
        for j in c.get('jobs', []):
            if j['name'] in by_name:
                j['state'] = {k: v for k, v in by_name[j['name']].items() if k not in ('name', 'stage', 'id', 'pid', 'dirs')}
        ph = _phase(c, st)
        if ph != c.get('phase'):
            c['phase'] = ph
            s.event(reg, f'aa_{ph}', key, {'dir': c['dir']})
        c['build'] = st.get('build')
        c['progress'] = (st.get('protocol') or {}).get('progress')
        if st.get('reference'):
            c['reference'] = st['reference']
        out[key] = st
    s.save(reg)
    return out


def _summary(key, c, st):
    b = c.get('build') or {}
    audit = (b.get('audit') or {}).get('system.neutral.data') or {}
    out = {'phase': c.get('phase'), 'target': c['target'], 'dir': c['dir'],
           'build': {k: b.get(k) for k in ('atoms', 'molecules', 'net_charge_e', 'charged_molecules', 'molecules_shifted',
                                            'charge_fix_opls_types', 'cg_topology')} | {
               'audit_verdict': audit.get('verdict'), 'audit_findings': audit.get('findings')} if b else None,
           'progress': c.get('progress'), 'reference': c.get('reference'),
           'jobs': [{'name': j['name'], 'id': j.get('id') or j.get('pid'), **(j.get('state') or {})}
                    for j in c.get('jobs', [])][-6:]}
    nxt = {'built': 'show the user the build and audit; aa submit --dry-run for the cost, then their approval',
           'build_failed': 'read the job log (jobs[].log_tail); fix with the user (typing, recipe), then aa build --force',
           'md_done': 'aa map', 'md_failed': 'read the log tail; aa submit --resume after a time limit or node failure',
           'mapped': f'ascg aa import --study <S> --task {key} --target {c["target"]} --ref {c.get("reference")} '
                     '--states <the approved states>'}
    if c.get('phase') in nxt:
        out['next'] = nxt[c['phase']]
    return out


@tool('aa', 'status', 'AA campaigns of a study (refreshed from the targets): phase (building, built, md_queued, '
      'md_running, md_done, mapping, mapped, or *_failed), build size, charge and audit, GROMACS progress per directory '
      '(steps done, next, ns/day), jobs, the mapped reference and the next step.', {
          'study': P(required=True), 'task': P(help='one system key (default: all)')})
def aa_status(study, task=None):
    s = Study(study)
    reg = s.load()
    if task:
        _campaign(s, reg, task)
    st = refresh_aa(s, reg, [task] if task else None)
    return {k: _summary(k, c, st.get(k)) for k, c in (reg.get('aa') or {}).items() if not task or k == task}


@tool('aa', 'submit', 'Queue the GROMACS protocol of a built system at the user-approved state points: equilibration '
      '(polymers: RadonPy 21-step schedule at the hottest state; liquids: direct compression), then one branch per '
      'state (pre-equilibration, equilibration, production). --dry-run returns the plan and the cost estimate to show '
      'the user; a real submit needs --approval with the user\'s words. Refuses before a clean build audit, beyond '
      'the target\'s core-hour cap, or while jobs run. --resume requeues what a time limit or node failure left '
      'unfinished (GROMACS continues from checkpoints).', {
          'study': P(required=True), 'task': P(required=True),
          'states': P('array', help='the user-approved state points (T, or T@P in atm), e.g. 400,500 or 300'),
          'approval': P(help="the user's approval of the plan and cost, in their words"),
          'mode': P(enum=('cpu', 'gpu'), help='default: the target\'s aa.mode (cpu)'),
          'eq_ns': P('number', help='branch equilibration (default 10 polymers, 4 liquids)'),
          'prod_ns': P('number', help='production per state (default 10)'),
          'pre_ns': P('number', help='pre-equilibration per state (default 1)'),
          'dry_run': P('boolean', default=False), 'resume': P('boolean', default=False)}, mutating=True)
def aa_submit(study, task, dry_run, resume, states=None, approval=None, mode=None, eq_ns=None, prod_ns=None,
              pre_ns=None):
    from ascg_core.ibi import states as st
    s = Study(study)
    reg = s.load()
    c = _campaign(s, reg, task)
    t = _aa_target(c['target'])
    refresh_aa(s, reg, [task])
    c = reg['aa'][task]
    if _active(c):
        raise PolicyError(f'AA jobs of {task!r} are still active ({c["phase"]}): aa status; aa cancel to stop them')
    mode = mode or t['aa'].get('mode', 'cpu')
    todo = []
    if resume:
        run = c.get('run')
        if not run:
            raise PolicyError('nothing to resume: no protocol was submitted')
        prog = c.get('progress') or {}
        todo = [d for d in [run['equilibration_dir'], *run['branch_dirs']] if (prog.get(d) or {}).get('next')]
        if not todo:
            raise PolicyError('every protocol directory is finished: aa map')
        settings = run
    else:
        if c['phase'] != 'built':
            raise PolicyError(f'{task!r} is {c["phase"]}: aa submit needs a finished build (aa status)')
        audit = ((c.get('build') or {}).get('audit') or {}).get('system.neutral.data') or {}
        if audit.get('verdict') != 'NEUTRAL_NO_HINTS' or (c['build'].get('charged_molecules') or 0):
            raise PolicyError('the build audit has findings: resolve them with the user first (ascg-aa-build): '
                              + json.dumps(audit.get('findings'))[:600])
        if not states:
            raise ToolError('--states: the user-approved state points')
        try:
            temps = st.parse(states)
        except ValueError as e:
            raise PolicyError(f'state points: {e}')
        cls = c['system_class']
        settings = {'states': ','.join(st.labels(temps)), 'states_K': [x.to_dict() for x in temps],
                    'equilibration': 'eq21' if cls == 'polymer' else 'direct',
                    'eq_ns': float(eq_ns if eq_ns is not None else (10 if cls == 'polymer' else 4)),
                    'prod_ns': float(prod_ns if prod_ns is not None else 10), 'pre_ns': float(pre_ns if pre_ns is not None
                                                                                            else 1)}
        if settings['prod_ns'] < 10 or settings['eq_ns'] < (10 if cls == 'polymer' else 4):
            settings['shortened'] = 'shorter than the protocol default: say so to the user and in reports'
        eq_total, branches = _lengths(cls, settings['states'], settings['eq_ns'], settings['prod_ns'],
                                      settings['pre_ns'])
        settings['estimate'] = _estimate(t, int(c['build']['atoms']), mode, eq_total, branches,
                                         settings['equilibration'])
        cap = float(t['aa'].get('max_core_h', MAX_CORE_H))
        if settings['estimate']['core_h_estimate'] > cap:
            raise PolicyError(f"estimated {settings['estimate']['core_h_estimate']} core-hours exceed the target's cap "
                              f'of {cap:g} per task (aa.max_core_h in the user\'s compute profile)')
    if dry_run:
        return {'task': task, 'plan': settings, 'mode': mode, 'resume': resume,
                'next': 'show the user the state points, run lengths and the estimate; submit with --approval "<their '
                        'words>" only after they agree'}
    if resume and not approval:                                # continues the plan the user approved
        approval = f"resume of the plan approved {c['approvals'][-1]['time']}" if c.get('approvals') else None
    if not approval or len(str(approval).strip()) < 3:
        raise PolicyError('--approval: the user\'s words approving this AA plan and its cost (aa submit --dry-run '
                          'shows them)')
    common = {'gmx': t['aa'].get('gmx', 'gmx'), 'mdrun': t['aa'].get('mdrun', 'gmx mdrun'),
              'aa_env': t['aa'].get('env', ''), 'require_cpu_flag': t['aa'].get('require_cpu_flag'), 'mode': mode}
    if not resume:
        proto = remote(t, 'aa_prepare', {'dir': c['dir'], 'key': task, 'python': t.get('python', 'python'),
                                         **{k: settings[k] for k in ('states', 'equilibration', 'eq_ns', 'prod_ns',
                                                                     'pre_ns')},
                                         'ff_override': t['aa'].get('ff_override')}, timeout=900)
        settings.update(equilibration_dir=proto['equilibration_dir'], branch_dirs=proto['branch_dirs'])
        todo = [proto['equilibration_dir'], *proto['branch_dirs']]
    md = _stage_settings(t, 'gpu' if mode == 'gpu' else 'md')
    maxh = max(_hours(md['time']) - 0.5, 0.5) if t['scheduler'] == 'slurm' else None
    eq_dir = settings['equilibration_dir']
    jobs = []
    if t['scheduler'] == 'slurm' and mode == 'cpu':             # equilibration, then the branches in parallel
        after = None
        if eq_dir in todo:
            j = _launch(t, c, study, task, 'md', submit=md, cpus=md['cpus'], dirs=[eq_dir], label=eq_dir, maxh=maxh,
                        **common)
            jobs.append(j)
            after = j['id']
        for b in [d for d in todo if d != eq_dir]:
            jobs.append(_launch(t, c, study, task, 'md', submit=md, cpus=md['cpus'], dirs=[b], label=b, maxh=maxh,
                                after=after, **common))
    else:                                                       # one job runs them in turn (GPU, or a workstation)
        gpu = None
        if mode == 'gpu' and t['scheduler'] == 'none':
            from ..policy import gpu_choice
            gp = remote(t, 'gpus', timeout=120)
            gpu = gpu_choice(gp, t, {g['index']: len(g['own']) for g in gp})
        jobs.append(_launch(t, c, study, task, 'md', submit=md, cpus=md['cpus'], dirs=todo, maxh=maxh, gpu=gpu,
                            **common))
    c['run'] = settings
    c['phase'] = 'md_queued'
    c.setdefault('approvals', []).append({'time': now(), 'approval': approval, 'resume': resume, 'mode': mode,
                                          'states': settings['states'], 'estimate': settings.get('estimate')})
    s.event(reg, 'aa_submitted', task, {'approval': approval, 'resume': resume, 'mode': mode,
                                        'jobs': [j.get('id') or j.get('pid') for j in jobs],
                                        'core_h_estimate': (settings.get('estimate') or {}).get('core_h_estimate')})
    s.save(reg)
    return {'task': task, 'jobs': jobs, 'plan': settings, 'next': 'wait with study watch (AA events) or aa status'}


@tool('aa', 'map', 'Map the finished GROMACS runs of a campaign onto beads with the study\'s mapping (map-aa): mapped '
      'trajectories, targets and start configurations per state, as one job. Then aa import registers the reference.', {
          'study': P(required=True), 'task': P(required=True)}, mutating=True)
def aa_map(study, task):
    s = Study(study)
    reg = s.load()
    c = _campaign(s, reg, task)
    t = _aa_target(c['target'])
    refresh_aa(s, reg, [task])
    c = reg['aa'][task]
    if c['phase'] not in ('md_done', 'map_failed'):
        raise PolicyError(f'{task!r} is {c["phase"]}: aa map needs every GROMACS directory finished')
    m = s.mapping(task)
    if not m or m['status'] != 'approved':
        raise PolicyError(f'the mapping of {task!r} is not approved (system approve or map approve)')
    from ascg_core.ibi import states as st
    labels = st.labels(st.parse(c['run']['states']))
    ms = _stage_settings(t, 'map')
    j = _launch(t, c, study, task, 'map', submit=ms, cpus=ms['cpus'], labels=labels)
    c['phase'] = 'mapping'
    s.event(reg, 'aa_map_submitted', task, {'job': j.get('id') or j.get('pid')})
    s.save(reg)
    return {'task': task, 'job': j, 'next': 'aa status; when mapped, aa import with the approved states'}


@tool('aa', 'cancel', 'Cancel the active AA jobs of a campaign (this study\'s own jobs only).', {
    'study': P(required=True), 'task': P(required=True)}, mutating=True)
def aa_cancel(study, task):
    s = Study(study)
    reg = s.load()
    c = _campaign(s, reg, task)
    t = target(c['target'])
    act = _active(c)
    if not act:
        return {'task': task, 'cancelled': [], 'note': 'no active AA jobs'}
    r = remote(t, 'aa_cancel', {'jobs': act, 'marker': f'{study}/aa:{task}'}, timeout=120)
    for j in act:
        j['state'] = {'status': 'cancelled'}
    s.event(reg, 'aa_cancelled', task, r)
    s.save(reg)
    return {'task': task, **r}
