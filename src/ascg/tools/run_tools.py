"""run: launch, bounded recovery actions, escalation, validation, cancellation. All placement and budget rules are
enforced here (ascg.policy), whatever agent or person calls the tool."""
from ..compute import remote, target
from ..policy import ACTIONS, action_budget, gpu_choice, own_job, qos_choice, slurm_capacity
from ..protocols import continuation_plan, cycle_length, project_cost
from ..registry import P, PolicyError, tool
from ..study import Study, now

ACTIVE = ('submitted', 'queued', 'running')


def _launch(s, reg, key, mode, extra=None):
    rec = s.task(reg, key)
    t = target(rec['target'])
    s.check_protocol_lock(reg)
    from ..protocols import V1
    if any(s.protocol().get(k) != V1.get(k) for k in ('gates', 'chain_band')):
        raise PolicyError('this study\'s gates differ from the engine\'s classifier (protocol v1 gates); unsupported '
                          'by this engine version')
    from .compute_tools import _local_digest
    there, here = remote(t, 'code_digest', timeout=120)['digest'], _local_digest()
    if there != here:                                    # jobs run exactly the code of this checkout
        raise PolicyError(f'the ascg code on {t["name"]} ({there}) differs from this checkout ({here}): deploy it first '
                          f'(ascg compute deploy --target {t["name"]})')
    payload = {'mode': mode, 'study': s.name, 'task': key, 'task_dir': rec['task_dir'], 'ref': rec['ref'],
               'states': rec.get('state_points') or rec.get('state_points_K'), 'protocol': s.protocol(),
               'scheduler': t['scheduler'],
               'submit': t.get('submit') or {}, 'env': t.get('env', ''), **(extra or {})}
    if t['scheduler'] == 'none':
        gpus = remote(t, 'gpus', timeout=120)
        busy = {g['index']: len({x['task'] for x in g['own']}) for g in gpus}    # ascg jobs of any study
        for r in reg['tasks'].values():                                          # just launched, not on the GPU yet
            if r.get('target') == t['name'] and r.get('status') in ('submitted', 'queued') and \
                    (r.get('job') or {}).get('gpu') is not None:
                busy[r['job']['gpu']] = busy.get(r['job']['gpu'], 0) + 1
        payload['gpu'] = gpu_choice(gpus, t, busy)
    else:
        slurm_capacity(t, sum(1 for r in reg['tasks'].values() if r.get('target') == t['name']
                              and r.get('status') in ACTIVE))
        qos = (t.get('submit') or {}).get('qos')
        if isinstance(qos, str) and ',' in qos:
            qos = [q.strip() for q in qos.split(',') if q.strip()]
        if isinstance(qos, list):                         # several QoS: the one with room under its per-user cap
            chosen, room = qos_choice(qos, remote(t, 'slurm_jobs', timeout=120),
                                      remote(t, 'qos_limits', {'qos': qos}, timeout=120))
            payload['submit'] = {**payload['submit'], 'qos': chosen}
            payload['qos_room'] = {q: (None if r == float('inf') else r) for q, r in room.items()}
    res = remote(t, 'launch', payload, timeout=300)
    rec['job'] = {**res, 'mode': mode, 'gpu': payload.get('gpu'), 'submitted': now(),
                  **({'qos': payload['submit']['qos']} if t['scheduler'] == 'slurm' else {})}
    rec.setdefault('jobs', []).append(rec['job'])
    rec['status'] = 'submitted'
    s.event(reg, 'submitted', key, {'mode': mode, 'target': t['name'], 'gpu': payload.get('gpu'),
                                    'job': res.get('id') or res.get('pid')})
    return rec['job']


@tool('run', 'preflight', 'Compatibility gate of a registered task (files, engine limits, coverage, dihedrals).', {
    'study': P(required=True), 'task': P(required=True)})
def run_preflight(study, task):
    s = Study(study)
    rec = s.task(s.load(), task)
    return remote(target(rec['target']), 'preflight', {'ref': rec['ref'],
                                                       'temperatures': rec.get('state_points') or rec.get('state_points_K')},
                  timeout=300)


@tool('run', 'submit', 'Start the protocol for a registered task on its target (GPU / queue chosen by policy). The job '
      'runs selfcheck, stage 1, stage 2 and the convergence check; it validates automatically if converged, otherwise '
      'stops at awaiting_decision; a scientific gate stop ends at stopped_on_gate.', {
          'study': P(required=True), 'task': P(required=True)}, mutating=True)
def run_submit(study, task):
    s = Study(study)
    reg = s.load()
    rec = s.task(reg, task)
    if rec.get('status') in ACTIVE:
        raise PolicyError(f'task {task} is already {rec["status"]}')
    if not rec.get('preflight_ok'):
        raise PolicyError(f'task {task} failed preflight: {rec.get("failed_checks")}')
    return _launch(s, reg, task, 'fit')


@tool('run', 'act', 'Bounded recovery for a task that is awaiting_decision or stopped_on_gate. Actions: extend '
      '(cycles 1-3), extend_lower_damping (cycles, lam 0.05-0.15), rewind_smaller_timestep (to_iteration, dt_factor '
      '0.5-0.9), rewind_lower_damping (to_iteration, lam), restart_smaller_timestep (dt_factor 0.5-0.9: only when the '
      'run stopped on the bond gate before iteration 0 completed, also if it was escalated for that reason; restarts '
      'from the Boltzmann-inverted start, the failed attempt is archived). Budget: 3 actions and 36 extra iterations '
      'per task. The '
      'result gives the plan the job runs (a run back inside stage 1 finishes stage 1 first, then the stage-2 schedule '
      'to the target iteration) and the projected GPU-hours; --dry-run returns them without acting.', {
          'study': P(required=True), 'task': P(required=True), 'action': P(required=True, enum=ACTIONS),
          'cycles': P('integer', default=1), 'to_iteration': P('integer'), 'dt_factor': P('number'),
          'lam': P('number'), 'rationale': P(required=True, help='why this action (logged; shown in the report)'),
          'dry_run': P('boolean', default=False, help='plan, projected cost and budget check only')},
      mutating=True)
def run_act(study, task, action, cycles, rationale, dry_run=False, to_iteration=None, dt_factor=None, lam=None):
    s = Study(study)
    reg = s.load()
    rec = s.task(reg, task)
    status = rec.get('status')
    restart = action == 'restart_smaller_timestep'
    if status not in ('awaiting_decision', 'stopped_on_gate') + (('escalated',) if restart else ()):
        raise PolicyError(f'task {task} is {status}; actions only at awaiting_decision or stopped_on_gate')
    t = target(rec['target'])
    dg = remote(t, 'digest', {'task_dir': rec['task_dir'], 'last': 40}, timeout=120)
    done = dg['completed']
    if restart:                                           # bond-gate stop before iteration 0
        failed = dg.get('failed') or {}
        if (done is not None and done >= 0) or failed.get('iteration') != 0:
            raise PolicyError('restart_smaller_timestep applies only to a run stopped on the gate before iteration 0 '
                              'completed; rewind instead')
        if dt_factor is None or not 0.5 <= dt_factor <= 0.9:
            raise PolicyError('dt_factor must be in [0.5, 0.9]')
        done = -1
    proto = s.protocol()
    clen = cycle_length(proto, has_dihedrals=(dg.get('settings') or {}).get('dihedrals') not in (None, 'none'))
    params = {'action': action, 'rationale': rationale}
    if restart:
        extra = 1                                         # the failed iteration 0 is run again
        params.update(dt_factor=dt_factor, target_iters=proto['stage2']['iters'])
    elif action.startswith('extend'):
        if status == 'stopped_on_gate':
            raise PolicyError('the run stopped on a gate: its last tables cannot be continued; rewind instead')
        if not 1 <= cycles <= 3:
            raise PolicyError('cycles must be 1-3')
        extra = cycles * clen
        params['target_iters'] = done + extra
    else:
        if to_iteration is None or not 0 <= to_iteration <= done:
            raise PolicyError(f'to_iteration must be a complete iteration in [0, {done}]')
        extra = done - to_iteration + (1 if status == 'stopped_on_gate' else 0)
        params.update(to_iteration=to_iteration, target_iters=max(done, proto['stage2']['iters']))
    if action in ('extend_lower_damping', 'rewind_lower_damping'):
        if lam is None or not 0.05 <= lam <= 0.15:
            raise PolicyError('lam must be in [0.05, 0.15] for a lower-damping action')
        params['lam'] = lam
    if action == 'rewind_smaller_timestep':
        if dt_factor is None or not 0.5 <= dt_factor <= 0.9:
            raise PolicyError('dt_factor must be in [0.5, 0.9]')
        params['dt_factor'] = dt_factor
    plan = continuation_plan(proto, params, -1 if restart else to_iteration if action.startswith('rewind') else done)
    remaining = sum(st['to'] - st['from'] + 1 for st in plan)
    if (action.startswith('rewind') or restart) and dt_factor:   # a smaller timestep makes each iteration slower
        dg['cost'] = {**dg['cost'], 's_per_iteration': (dg['cost'].get('s_per_iteration') or 0) / dt_factor} \
            if dg.get('cost') and dg['cost'].get('s_per_iteration') else dg.get('cost')
    projection = project_cost(proto, dg.get('cost'), remaining, clen)
    if dry_run:
        try:
            action_budget(rec, action, extra)
            check = 'ok'
        except PolicyError as e:
            check = f'refused: {e}'
        return {'dry_run': True, 'action': params, 'extra_iters': extra, 'plan': plan, 'projection': projection,
                'budget_check': check}
    action_budget(rec, action, extra)
    if status == 'escalated':                             # reopened: the escalation asked for this decision
        s.event(reg, 'reopened', task, {'previous_outcome': rec.get('outcome'), 'by': action})
        rec['outcome'] = None
    rec.setdefault('interventions', []).append({**params, 'extra_iters': extra, 'time': now()})
    s.event(reg, 'action_taken', task, {k: v for k, v in params.items() if k != 'rationale'} | {'rationale': rationale})
    job = _launch(s, reg, task, 'continue', {'action': params})
    return {'action': params, 'extra_iters': extra, 'job': job, 'plan': plan, 'projection': projection,
            'budget_left': {'actions': 3 - len(rec['interventions']),
                            'iters': 36 - sum(i['extra_iters'] for i in rec['interventions'])}}


@tool('run', 'escalate', 'Stop the task and hand it to the user: outcome "needs expert input" with your explanation.', {
    'study': P(required=True), 'task': P(required=True), 'message': P(required=True)}, mutating=True)
def run_escalate(study, task, message):
    s = Study(study)
    reg = s.load()
    rec = s.task(reg, task)
    if rec.get('status') in ACTIVE:
        raise PolicyError(f'task {task} is {rec["status"]}; cancel it first or wait')
    rec['status'] = 'escalated'
    rec['outcome'] = {'class': 'needs expert input', 'reasons': [message]}
    s.event(reg, 'escalated', task, {'message': message})
    return {'task': task, 'status': 'escalated'}


@tool('run', 'validate', 'Held-out validation of the current tables (also after an escalation if the user asks).', {
    'study': P(required=True), 'task': P(required=True)}, mutating=True)
def run_validate(study, task):
    s = Study(study)
    reg = s.load()
    rec = s.task(reg, task)
    if rec.get('status') in ACTIVE:
        raise PolicyError(f'task {task} is {rec["status"]}')
    return _launch(s, reg, task, 'validate')


@tool('run', 'cancel', 'Cancel this study\'s job of a task (never anything else).', {
    'study': P(required=True), 'task': P(required=True)}, mutating=True)
def run_cancel(study, task):
    s = Study(study)
    reg = s.load()
    rec = s.task(reg, task)
    job = own_job(rec)
    res = remote(target(rec['target']), 'cancel', {'job': job, 'marker': f'{study}/{task}'}, timeout=120)
    rec['status'] = 'cancelled'
    rec['queue'] = None                                 # the scheduler's last estimate no longer applies
    s.event(reg, 'cancelled', task, res)
    return res
