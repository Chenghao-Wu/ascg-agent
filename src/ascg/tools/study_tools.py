"""study: registry, status (one batched remote call per target), events and event-driven waiting."""
import time

from ..compute import remote, target
from ..config import studies_root
from ..policy import MAX_EXTRA_ITERS, MAX_INTERVENTIONS
from ..protocols import V1, agentic, cycle_length, project_cost
from ..registry import P, ToolError, tool
from ..study import Study, now

TERMINAL = ('done', 'escalated', 'cancelled')
WAKE = ('awaiting_decision', 'stopped_on_gate', 'done', 'error', 'aa_built', 'aa_build_failed', 'aa_md_done',
        'aa_md_failed', 'aa_mapped', 'aa_map_failed')


@tool('study', 'init', 'Create a study: locks the protocol (hash) and creates the registry.', {
    'study': P(required=True), 'description': P(default=''),
    'arm': P(enum=('agentic', 'scripted'), default='agentic', help='agentic: no automatic extension, decisions by the '
             'agent within the budget; scripted: protocol v1 as registered')}, mutating=True)
def study_init(study, description, arm):
    s = Study(study)
    proto = agentic() if arm == 'agentic' else V1
    reg = s.init(proto, description)
    return {'study': study, 'dir': str(s.dir), 'protocol_hash': reg['protocol']['hash'], 'protocol': proto}


@tool('study', 'list', 'Studies under the studies root.')
def study_list():
    root = studies_root()
    return {'root': str(root), 'studies': sorted(p.parent.name for p in root.glob('*/registry.json'))}


def refresh(s, reg):
    """Query every target once, update task records, emit events for status changes."""
    by_target = {}
    for key, rec in reg['tasks'].items():
        if rec.get('job') and rec.get('status') not in TERMINAL + ('registered',):
            by_target.setdefault(rec['target'], []).append(key)
    for tname, keys in by_target.items():
        t = target(tname)
        try:
            st = remote(t, 'task_state', {'tasks': [{'key': k, 'task_dir': reg['tasks'][k]['task_dir'],
                                                     'job': reg['tasks'][k]['job']} for k in keys]}, timeout=300)
        except ToolError as e:
            for k in keys:
                reg['tasks'][k]['last_error'] = str(e)
            continue
        for k in keys:
            rec, d = reg['tasks'][k], st[k]
            rec['remote'] = {kk: d.get(kk) for kk in ('completed', 'last', 'last_npt', 'failed', 'outcome', 'states',
                                                      'cost')}
            rec['queue'] = d.get('queue')
            job = d.get('job') or {}
            js = job.get('status')
            if js in ('done', 'awaiting_decision', 'stopped_on_gate', 'error'):
                new = js
            elif js == 'running':
                new = 'running' if d.get('alive') is not False else 'error'
                if new == 'error':
                    job['message'] = 'job ended without a final status (killed or crashed)'
            elif js == 'submitted':
                new = 'queued' if d.get('alive') else ('error' if d.get('alive') is False else 'queued')
            else:
                new = rec.get('status')
            if job.get('phase') and job['phase'] != rec.get('phase') and new == 'running':
                rec['phase'] = job['phase']                     # e.g. converged -> validating
                s.event(reg, 'phase', k, {'phase': job['phase'], 'message': job.get('message', '')})
            if new != rec.get('status'):
                data = {'message': job.get('message', '')}
                if new == 'done':
                    data['class'] = (d.get('outcome') or {}).get('class') or job.get('outcome')
                    rec['outcome'] = d.get('outcome')
                rec['status'] = new
                s.event(reg, new, k, data)
            done = d.get('completed') or 0
            if done - rec.get('progress_reported', -6) >= 6 and rec['status'] == 'running':
                rec['progress_reported'] = done
                s.event(reg, 'progress', k, {'iteration': done, 'last': (d.get('last') or {}),
                                             'last_npt': d.get('last_npt')})
    s.save(reg)
    active = [k for k, c in (reg.get('aa') or {}).items()
              if c.get('phase') in ('building', 'md_queued', 'md_running', 'mapping')]
    if active:                                         # AA campaigns: one call per campaign
        from .aa_run_tools import refresh_aa
        refresh_aa(s, reg, active)
    return reg


def budget(r, proto):
    """Action budget left and the measured / projected GPU-hours of a task (see protocols.project_cost)."""
    used = r.get('interventions', [])
    out = {'actions_left': MAX_INTERVENTIONS - len(used),
           'extra_iterations_left': MAX_EXTRA_ITERS - sum(i.get('extra_iters', 0) for i in used)}
    rm = r.get('remote') or {}
    if proto and rm.get('cost'):
        if r.get('status') == 'done':
            out['gpu_hours_spent'] = round(rm['cost']['spent_gpu_h'], 2)
        else:
            target_iters = used[-1]['target_iters'] if used else proto['stage2']['iters']
            out['cost'] = project_cost(proto, rm['cost'], target_iters - (rm.get('completed') or 0),
                                       cycle_length(proto))
    return out


def summary(reg, proto=None):
    out = {}
    for k, r in reg['tasks'].items():
        rm = r.get('remote') or {}
        out[k] = {'status': r.get('status'), 'target': r.get('target'), 'iteration': rm.get('completed'),
                  'last': rm.get('last'), 'last_npt': rm.get('last_npt'),
                  'states': r.get('state_points') or r.get('state_points_K') or rm.get('states'),
                  'outcome': (r.get('outcome') or {}).get('class') if r.get('outcome') else None,
                  'interventions': len(r.get('interventions', [])), 'budget': budget(r, proto)}
        if r.get('phase') and r.get('status') == 'running':
            out[k]['phase'] = r['phase']
        if r.get('status') == 'queued' and r.get('queue'):
            out[k]['queue'] = r['queue']           # scheduler's estimate: expected start and wait
    return out


@tool('study', 'status', 'Status of every task (refreshed from the targets).', {'study': P(required=True)})
def study_status(study):
    s = Study(study)
    reg = refresh(s, s.load())
    return {'study': study, 'protocol_hash': reg['protocol']['hash'], 'tasks': summary(reg, s.protocol()),
            'event_cursor': reg.get('event_seq', 0)}


@tool('study', 'events', 'Events since a cursor (study_created, task_registered, submitted, queued, running, progress, phase, '
      'awaiting_decision, stopped_on_gate, done, error, action_taken, escalated, amendment_*, system_*, mapping_*, '
      'aa_*).', {
          'study': P(required=True), 'since': P('integer', default=0)})
def study_events(study, since):
    s = Study(study)
    return {'events': s.events(since), 'cursor': s.load().get('event_seq', 0)}


@tool('study', 'watch', 'Block until an event of the given kinds occurs (or the timeout), refreshing every interval. '
      'Use this instead of polling; keep timeout under the tool-call limit (default 540 s).', {
          'study': P(required=True), 'since': P('integer', help='event cursor (default: current)'),
          'until': P('array', default=list(WAKE)), 'timeout': P('integer', default=540),
          'interval': P('integer', default=60)})
def study_watch(study, until, timeout, interval, since=None):
    s = Study(study)
    reg = s.load()
    cursor = reg.get('event_seq', 0) if since is None else since
    t0 = time.time()
    while True:
        reg = refresh(s, s.load())
        hits = [e for e in s.events(cursor) if e['event'] in until]
        if hits:
            return {'events': hits, 'cursor': reg.get('event_seq', 0), 'timed_out': False,
                    'tasks': summary(reg, s.protocol())}
        if time.time() - t0 + interval > timeout:
            return {'events': [], 'cursor': reg.get('event_seq', 0), 'timed_out': True,
                    'tasks': summary(reg, s.protocol()), 'waited_s': round(time.time() - t0)}
        time.sleep(interval)


@tool('study', 'note', 'Append a note to the event log (corrections, observations, user decisions). The log is '
      'append-only: never edit events.jsonl or registry.json by hand.', {
          'study': P(required=True), 'message': P(required=True),
          'task': P(help='task or system key, if about one (a registered task, a saved system or mapping)')},
      mutating=True)
def study_note(study, message, task=None):
    import os
    s = Study(study)
    reg = s.load()
    if task is not None and task not in reg['tasks'] and not (s.system(task) or s.mapping(task)):
        s.task(reg, task)                                  # raises: no such task, system or mapping
    ev = s.event(reg, 'note', task, {'message': message, 'actor': os.environ.get('ASCG_ACTOR', 'user')})
    return {'event': ev}


@tool('study', 'verify', 'Protocol lock check (read-only): the study\'s protocol file against the hash locked at '
      'study init (or an approved amendment), and its gates against the classifier\'s. Every launch runs the same check.',
      {'study': P(required=True)})
def study_verify(study):
    from ascg_core.ibi.pipeline import PROTOCOL
    s = Study(study)
    reg = s.load()
    out = {'locked_hash': reg['protocol']['hash']}
    try:
        out['protocol_hash'] = s.check_protocol_lock(reg)
        out['lock_ok'] = True
    except ToolError as e:
        out.update(lock_ok=False, error=str(e))
    gates = s.protocol().get('gates')
    out['gates'] = gates
    out['gates_match_classifier'] = gates == PROTOCOL['gates']
    out['ok'] = out['lock_ok'] and out['gates_match_classifier']
    return out


@tool('study', 'explain', 'Everything that defines a task: reference, target, protocol and its hash, amendments, '
      'fastMD build, job history, interventions, the run\'s invocations and current settings.', {
          'study': P(required=True), 'task': P(help='default: the study\'s only task')})
def study_explain(study, task=None):
    s = Study(study)
    reg = s.load()
    if task is None:
        if len(reg['tasks']) != 1:
            raise ToolError(f'give --task: one of {sorted(reg["tasks"])}')
        task = next(iter(reg['tasks']))
    rec = s.task(reg, task)
    out = {'task': task, 'record': rec, 'protocol': s.protocol(), 'protocol_hash': reg['protocol']['hash'],
           'amendments': s.amendments()}
    if rec.get('job'):
        try:
            d = remote(target(rec['target']), 'digest', {'task_dir': rec['task_dir'], 'last': 3}, timeout=120)
            out['run_settings'] = d.get('settings')         # the latest invocation's; None before iteration 0
            out['invocations'] = d.get('invocations')
            out['job'] = d.get('job')
            if d.get('note'):
                out['note'] = d['note']
        except ToolError as e:
            out['remote_error'] = str(e)
    return out


@tool('study', 'adopt', 'Register an existing task directory (e.g. a run made outside this study, or a recorded case for '
      'evaluation) so the diagnostic and run tools can act on it. Its status is read from the directory.', {
          'study': P(required=True), 'task': P(required=True), 'target': P(required=True),
          'ref': P(required=True), 'task_dir': P(required=True),
          'status': P(enum=('auto', 'awaiting_decision', 'stopped_on_gate', 'done'), default='auto')}, mutating=True)
def study_adopt(study, task, ref, task_dir, status, **kw):
    s = Study(study)
    reg = s.load()
    t = target(kw['target'])
    st = remote(t, 'task_state', {'tasks': [{'key': task, 'task_dir': task_dir, 'job': {}}]}, timeout=300)[task]
    if status == 'auto':
        status = ('stopped_on_gate' if st.get('failed') else 'done' if st.get('outcome') else
                  'awaiting_decision' if st.get('completed') is not None else 'registered')
    outcome = st.get('outcome')
    if outcome is None and status == 'done':          # a finished run without an outcome: classify its validation
        c = remote(t, 'classify', {'task_dir': task_dir}, timeout=300)
        outcome = {'class': c['class'], 'reasons': c['reasons'], 'classified_from': c.get('from')} if c['class'] else None
    reg['tasks'][task] = {'key': task, 'target': t['name'], 'ref': ref, 'task_dir': task_dir, 'status': status,
                          'preflight_ok': True, 'adopted': True, 'state_points': st.get('states'),
                          'remote': {k: st.get(k) for k in ('completed', 'last', 'last_npt', 'failed', 'outcome')},
                          'outcome': outcome}
    s.event(reg, 'task_adopted', task, {'status': status, 'failed': bool(st.get('failed'))})
    return {'task': task, 'status': status, 'state': st}
