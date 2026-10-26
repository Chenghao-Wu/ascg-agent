"""Agent evaluation harness: the same tasks under Claude Code and Codex, with frozen tools, isolated runs and scoring.

    python eval/harness.py run --out ~/ascg-eval/BATCH [--tasks ID ...] [--agents claude codex] [--replicates N]
                               [--jobs J] [--claude-model M] [--codex-model M] [--codex-effort E]
    python eval/harness.py score ~/ascg-eval/BATCH

A batch freezes the tools once: the tracked files of src/ascg, src/ascg_core and plugin/ascg are copied to
BATCH/code, with the git commit and the code digest in BATCH/batch.json. Every run in the batch uses that copy;
rerunning `run` on a batch adds missing runs with the same copy (and refuses if the checkout's code changed).

Each run, BATCH/runs/<task>/<agent>_r<k>/, has its own workspace (the agent's working directory), studies root,
compute profile, task data and call log (ASCG_CALL_LOG: every ascg tool call with its result), plus the agent's
transcript and meta.json. Isolation:
- no system library unless the task names one; Claude Code without auto-memory, user settings or user MCP servers,
  reads of the source checkout and of ~/.claude/projects denied; Codex with --ignore-user-config, its shell in the
  workspace-write sandbox without network, the ascg tools over MCP only (`ascg init codex` settings);
- BATCH must lie outside the source checkout (agents would read the benchmark library and the tests) and outside
  /tmp and $TMPDIR (Codex could write the tools there).

Agents (arms): claude (main agent; subagents allowed), claude-solo (no subagents), codex.
Scoring: eval/score.py (task checks on the call log and the final answer; process metrics from the transcripts).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = Path(__file__).resolve().parent/'tasks'
FROZEN = ('src/ascg/', 'src/ascg_core/', 'plugin/ascg/')
ARMS = {'claude': {'harness': 'claude', 'subagents': True},
        'claude-solo': {'harness': 'claude', 'subagents': False},
        'codex': {'harness': 'codex'}}
CLAUDE_TOOLS = ['Bash(ascg *)', 'Read', 'Glob', 'Grep', 'Skill', 'TodoWrite']


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def load_task(tid, tasks_dir=TASKS):
    t = json.loads((Path(tasks_dir)/f'{tid}.json').read_text())
    t['sha256'] = hashlib.sha256((Path(tasks_dir)/f'{tid}.json').read_bytes()).hexdigest()[:16]
    return t


def _inside(p, roots):
    p = Path(p).expanduser().resolve()
    return next((r for r in roots if p == r or r in p.parents), None)


def freeze(out):
    """Copy the tracked tools into out/code once per batch; returns the batch record."""
    rec_file = out/'batch.json'
    files = [f for f in sh(['git', '-C', str(ROOT), 'ls-files'], check=True).stdout.splitlines()
             if f.startswith(FROZEN) and '__pycache__' not in f and (ROOT/f).is_file()]
    digest = hashlib.md5()
    for f in sorted(files):
        digest.update(f.encode() + b'\0' + (ROOT/f).read_bytes())
    digest = digest.hexdigest()[:16]
    if rec_file.exists():
        rec = json.loads(rec_file.read_text())
        if rec['tools_digest'] != digest:
            raise SystemExit(f'{out}: frozen with tools {rec["tools_digest"]}, the checkout now has {digest}; use a new '
                             'batch directory')
        return rec
    code = out/'code'
    for f in files:
        (code/f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/f, code/f)
    version = sh([sys.executable, '-c', 'import ascg; print(ascg.__version__)'],
                 env={**os.environ, 'PYTHONPATH': str(code/'src')}).stdout.strip()
    rec = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'source_commit': sh(['git', '-C', str(ROOT), 'rev-parse',
           'HEAD']).stdout.strip(), 'source_dirty': bool(sh(['git', '-C', str(ROOT), 'status', '--porcelain',
           '--untracked-files=no']).stdout.strip()), 'tools_digest': digest, 'files': len(files),
           'ascg_version': version, 'python': sys.executable,
           'claude_cli': sh(['claude', '--version']).stdout.strip(), 'codex_cli': sh(['codex', '--version']).stdout.strip()}
    rec_file.write_text(json.dumps(rec, indent=1) + '\n')
    return rec


def prepare(run, task, out):
    """Workspace, studies, data, compute profile and environment of one run."""
    from ascg import agents                              # the batch's frozen copy (main puts it first on sys.path)
    import fixtures
    ws, studies, data = run/'workspace', run/'studies', run/'data'
    for d in (ws, studies):
        d.mkdir(parents=True, exist_ok=True)
    if task.get('fixture'):
        fixtures.make(task['fixture'], data)
    compute = run/'compute.json'
    targets = {}
    if task.get('compute') == 'local':                  # a target "here": this machine, the frozen code
        targets['here'] = {'transport': 'local', 'scheduler': 'none', 'root': str(out/'code'), 'python': sys.executable,
                           'env': f'export PYTHONPATH={out/"code"/"src"}', 'work_root': str(run/'work')}
    if task.get('compute') == 'user':
        compute = Path(os.environ.get('ASCG_COMPUTE') or '~/.config/ascg/compute.yaml').expanduser().resolve()
    else:
        compute.write_text(json.dumps({'targets': targets}, indent=1) + '\n')
    lib = task.get('library', 'none')
    library = 'none' if lib == 'none' else str(Path(lib).expanduser().resolve())
    env = {k: v for k, v in os.environ.items() if not k.startswith(('ASCG_', 'CLAUDE_CODE_', 'CODEX_')) or
           k == 'CODEX_HOME'}
    env.update(ASCG_STUDIES=str(studies), ASCG_COMPUTE=str(compute), ASCG_LIBRARY=library,
               ASCG_CALL_LOG=str(run/'calls.jsonl'), ASCG_PYTHON=sys.executable)
    prompt = task['prompt'].format(data=data, run=run)
    return agents, ws, studies, compute, library, env, prompt


def claude_cmd(arm, task, run, out, env, prompt, model):
    env.update(CLAUDE_CODE_DISABLE_AUTO_MEMORY='1', ASCG_ACTOR=f'claude:{task["id"]}:{run.name}')
    tools = CLAUDE_TOOLS + (['Agent'] if ARMS[arm]['subagents'] else [])
    deny = [f'Read(/{p}/**)' for p in (ROOT, Path('~/.claude/projects').expanduser())]
    cmd = ['claude', '-p', prompt, '--plugin-dir', str(out/'code'/'plugin'/'ascg'), '--permission-mode', 'dontAsk',
           '--allowedTools', *tools, '--disallowedTools', *deny, *([] if ARMS[arm]['subagents'] else ['Agent']),
           '--setting-sources', 'project', '--strict-mcp-config', '--output-format', 'stream-json', '--verbose',
           '--max-turns', str(task.get('max_turns', 60))]
    return cmd + (['--model', model] if model else [])


def codex_cmd(agents, task, run, out, ws, studies, compute, library, prompt, model, effort):
    roots = agents.writable_roots(ws)
    agents.check_outside({'ascg code': out/'code', 'studies': studies, 'compute profile': compute, 'data': run/'data',
                          'call log': run/'calls.jsonl', 'Python': Path(sys.executable).resolve()}, roots)
    agents.write_workspace(ws, skills=out/'code'/'plugin'/'ascg'/'skills', force=True)
    s = agents.codex_settings(ws, python=sys.executable, code=out/'code'/'src', studies=studies, compute=compute,
                              library=library, actor=f'codex:{task["id"]}:{run.name}', approval='never')
    s['mcp_servers']['ascg']['env']['ASCG_CALL_LOG'] = str(run/'calls.jsonl')
    (run/'codex_settings.json').write_text(json.dumps(s, indent=1) + '\n')
    return ['codex', 'exec', '--json', '-C', str(ws), '--skip-git-repo-check', '--ignore-user-config',
            '-m', model, '-c', f'model_reasoning_effort="{effort}"', *agents.codex_flags(s), prompt]


def run_one(out, task, arm, k, opts):
    run = out/'runs'/task['id']/f'{arm}_r{k}'
    if (run/'meta.json').exists():
        return {'run': str(run), 'skipped': 'exists'}
    run.mkdir(parents=True, exist_ok=True)
    (run/'task.json').write_text(json.dumps(task, indent=1) + '\n')         # scored with the checks it ran under
    agents, ws, studies, compute, library, env, prompt = prepare(run, task, out)
    if ARMS[arm]['harness'] == 'claude':
        cmd = claude_cmd(arm, task, run, out, env, prompt, opts.claude_model)
        model = opts.claude_model or 'default'
    else:
        env['ASCG_ACTOR'] = f'codex:{task["id"]}:{run.name}'
        cmd = codex_cmd(agents, task, run, out, ws, studies, compute, library, prompt, opts.codex_model,
                        opts.codex_effort)
        model = f'{opts.codex_model} ({opts.codex_effort})'
    meta = {'task': task['id'], 'task_sha256': task['sha256'], 'held_out': task.get('held_out'), 'arm': arm,
            'replicate': k, 'model': model, 'prompt': prompt, 'start': time.strftime('%Y-%m-%d %H:%M:%S'),
            'cwd': str(ws), 'ascg_env': {x: env[x] for x in env if x.startswith('ASCG_')}}
    t0, timed_out = time.time(), False
    with open(run/'transcript.jsonl', 'w') as fo, open(run/'stderr.log', 'w') as fe:
        p = subprocess.Popen(cmd, cwd=ws, env=env, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe,
                             start_new_session=True)
        try:
            p.wait(timeout=60 * task.get('timeout_min', 30))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
    meta.update(end=time.strftime('%Y-%m-%d %H:%M:%S'), wall_s=round(time.time() - t0, 1), exit=p.returncode,
                timed_out=timed_out)
    (run/'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    return {'run': str(run), 'exit': p.returncode, 'wall_s': meta['wall_s'], 'timed_out': timed_out}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--out', required=True)
    r.add_argument('--tasks', nargs='*', help='task ids (default: all in --tasks-dir)')
    r.add_argument('--tasks-dir', default=str(TASKS))
    r.add_argument('--agents', nargs='*', default=['claude', 'codex'], choices=sorted(ARMS))
    r.add_argument('--replicates', type=int, default=1)
    r.add_argument('--jobs', type=int, default=4)
    r.add_argument('--claude-model')
    r.add_argument('--codex-model', default='gpt-5.6-sol')
    r.add_argument('--codex-effort', default='xhigh')
    s = sub.add_parser('score')
    s.add_argument('out')
    a = ap.parse_args(argv)
    if a.cmd == 'score':
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from score import score_batch
        print(json.dumps(score_batch(Path(a.out).expanduser().resolve()), indent=1))
        return 0
    out = Path(a.out).expanduser().resolve()
    import tempfile
    bad = _inside(out, [ROOT.resolve(), Path('/tmp').resolve(), Path(tempfile.gettempdir()).resolve()])
    if bad:
        raise SystemExit(f'--out {out} lies inside {bad}: choose a directory outside the checkout, /tmp and $TMPDIR')
    out.mkdir(parents=True, exist_ok=True)
    batch = freeze(out)
    sys.path[:0] = [str(out/'code'/'src'), str(Path(__file__).resolve().parent)]
    ids = a.tasks or sorted(p.stem for p in Path(a.tasks_dir).glob('*.json'))
    tasks = [load_task(i, a.tasks_dir) for i in ids]
    jobs = [(t, arm, k) for t in tasks for arm in a.agents for k in range(1, a.replicates + 1)]
    print(json.dumps({'batch': str(out), 'tools_digest': batch['tools_digest'], 'runs': len(jobs)}), flush=True)
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for res in ex.map(lambda j: run_one(out, j[0], j[1], j[2], a), jobs):
            print(json.dumps(res), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
