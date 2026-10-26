"""Policies enforced inside the tools (fail closed). Each one comes from an incident of the operator trajectory or a
user instruction; none names a site."""
from __future__ import annotations

from .registry import PolicyError

# CG recovery budget per task (protocol v1 section 6)
MAX_INTERVENTIONS = 3
MAX_EXTRA_ITERS = 36
ACTIONS = ('extend', 'extend_lower_damping', 'rewind_smaller_timestep', 'rewind_lower_damping',
           'restart_smaller_timestep')


def own_job(task_rec):
    """Only jobs / processes this study launched (recorded at submission) can be cancelled."""
    job = task_rec.get('job') or {}
    if not job.get('id') and not job.get('pid'):
        raise PolicyError('this task has no job launched by the study; ascg only acts on its own jobs')
    return job


def gpu_choice(gpus, t, busy_own):
    """A GPU on a workstation target: free of foreign processes (gpu_policy avoid_foreign) and below
    tasks_per_gpu of this study's tasks. gpus: [{index, foreign: [...], own: [...]}]."""
    allowed = t.get('gpus')
    for g in gpus:
        if allowed is not None and g['index'] not in allowed:
            continue
        if t.get('gpu_policy', 'avoid_foreign') == 'avoid_foreign' and g['foreign']:
            continue
        if busy_own.get(g['index'], 0) >= int(t.get('tasks_per_gpu', 1)):
            continue
        return g['index']
    raise PolicyError(f'target {t["name"]}: no GPU free of other users\' processes '
                      f'(gpu_policy {t.get("gpu_policy")}); try another target or wait (ascg compute hosts)')


def slurm_capacity(t, n_own_active):
    lim = (t.get('submit') or {}).get('max_jobs')
    if lim is not None and n_own_active >= int(lim):
        raise PolicyError(f'target {t["name"]}: {n_own_active} of max_jobs {lim} already used by this study')


def qos_choice(qos_list, jobs, caps):
    """The QoS a new one-GPU job goes to: the first in the list with GPUs left under its per-user cap, counting the
    account's running and pending jobs in it (they come first); with none left, the one with the most room (a job
    waits in its queue either way). jobs: slurm_jobs rows (qos, gres); caps: {qos: GPUs per user or None}."""
    import re
    used = {q: 0 for q in qos_list}
    for j in jobs:
        if j.get('qos') in used and j.get('state') in ('RUNNING', 'PENDING', 'CONFIGURING'):
            m = re.search(r'gpu(?::[\w-]+)*[:=](\d+)', str(j.get('gres') or ''))
            used[j['qos']] += int(m.group(1)) if m else 1
    room = {q: (caps.get(q) - used[q]) if caps.get(q) is not None else float('inf') for q in qos_list}
    for q in qos_list:
        if room[q] > 0:
            return q, room
    return max(qos_list, key=lambda q: room[q]), room


def action_budget(task_rec, action, extra_iters):
    if action not in ACTIONS:
        raise PolicyError(f'action {action!r} not allowed; allowed: {", ".join(ACTIONS)} (or escalate)')
    used = task_rec.get('interventions', [])
    if len(used) >= MAX_INTERVENTIONS:
        raise PolicyError(f'budget: {MAX_INTERVENTIONS} interventions used; escalate instead')
    spent = sum(u.get('extra_iters', 0) for u in used)
    if spent + extra_iters > MAX_EXTRA_ITERS:
        raise PolicyError(f'budget: {spent} of {MAX_EXTRA_ITERS} extra iterations used, {extra_iters} requested')
