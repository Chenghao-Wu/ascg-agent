"""Compute targets from the user's profile: transport (ssh | local) x scheduler (slurm | none). The remote side is
`python -m ascg.remote <verb>` started inside the target's environment; arguments travel as JSON on stdin, so no user
value is ever interpolated into a shell string."""
from __future__ import annotations

import json
import shlex
import subprocess

from .config import load_profiles
from .registry import ToolError

DEFAULTS = {'transport': 'ssh', 'scheduler': 'none', 'python': 'python', 'gpu_policy': 'avoid_foreign',
            'tasks_per_gpu': 1}


def target(name):
    targets, path = load_profiles()
    if name not in targets:
        raise ToolError(f'unknown compute target {name!r}' + (f' in {path}' if path else ' (no compute profile: '
                        'ascg compute add, see the ascg-setup-compute skill)'))
    t = {**DEFAULTS, **targets[name], 'name': name}
    if 'root' not in t:
        raise ToolError(f'target {name!r}: `root` (project / package location on the target) is required')
    t.setdefault('work_root', t['root'].rstrip('/') + '/runs/ascg')
    return t


def remote(t, verb, payload=None, timeout=600):
    """Run one remote verb on target t; returns the decoded JSON reply (raises ToolError on failure)."""
    env = t.get('env', '')
    inner = (f'{env} >/dev/null 2>&1; ' if env else '') + \
        f'cd {shlex.quote(t["root"])} && {t["python"]} -m ascg.remote {shlex.quote(verb)}'
    if t['transport'] == 'local':
        cmd = ['bash', '-c', inner]
    elif t['transport'] == 'ssh':
        cmd = ['ssh', '-o', 'BatchMode=yes', t['host'], 'bash -c ' + shlex.quote(inner)]
    else:
        raise ToolError(f'target {t["name"]}: unknown transport {t["transport"]!r}')
    try:
        p = subprocess.run(cmd, input=json.dumps(payload or {}), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ToolError(f'{t["name"]}: remote verb {verb!r} timed out after {timeout} s')
    out = p.stdout.strip().splitlines()
    reply = None
    for line in reversed(out):                     # the last JSON line is the reply; anything else is host noise
        if line.startswith('{'):
            try:
                reply = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if reply is None:
        raise ToolError(f'{t["name"]}: remote verb {verb!r} failed (exit {p.returncode}): '
                        f'{(p.stderr or p.stdout).strip()[-600:]}')
    if not reply.get('ok', False):
        raise ToolError(f'{t["name"]}: {verb}: {reply.get("error")}')
    return reply['result']
