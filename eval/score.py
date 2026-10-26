"""Scoring of evaluation runs (eval/harness.py): task checks and process metrics per run, a summary per batch.

    python eval/score.py ~/ascg-eval/BATCH        (also: python eval/harness.py score BATCH)

Task checks (task "checks", evaluated on the run's call log, i.e. what the ascg tools actually did, and on the
agent's final answer):
- called: the tool was called (with "ok": true, successfully);
- not_called: none of the tools was called at all;
- result: a successful call of the tool returned r with `expr` true (a Python expression over r): the last one,
  or any one with "which": "any" (e.g. one of several candidate mappings an agent checked);
- final: the final answer matches `regex` (case-insensitive).
Metrics: wall time, cost (Claude Code) or tokens (Codex), ascg calls (failed, refused by policy), agent shell commands,
reads of the tool sources, subagents, final answer. Writes score.json per run and summary.json / summary.md per batch.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SAFE = {'len': len, 'sorted': sorted, 'sum': sum, 'min': min, 'max': max, 'any': any, 'all': all, 'set': set,
        'abs': abs, 'round': round}
SOURCE = re.compile(r'(/code/src/|\.py\b)')
ASCG = re.compile(r'(^|[\s;&|])ascg\s')


def _jsonl(p):
    out = []
    if Path(p).exists():
        for line in Path(p).read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def claude_metrics(events):
    m = {'shell_ascg': 0, 'shell_other': 0, 'help': 0, 'source_reads': 0, 'subagents': 0, 'skills': [],
         'denied': 0, 'final': None, 'cost_usd': None, 'turns': None, 'model': None}
    for d in events:
        if d.get('type') == 'system' and d.get('subtype') == 'init':
            m['model'] = d.get('model')
        if d.get('type') == 'assistant':
            for b in d['message'].get('content', []):
                if b.get('type') != 'tool_use':
                    continue
                i, n = b.get('input') or {}, b.get('name')
                if n == 'Bash':
                    c = i.get('command', '')
                    m['shell_ascg' if ASCG.search(c) else 'shell_other'] += 1
                    m['help'] += '--help' in c
                elif n in ('Read', 'Grep', 'Glob'):
                    m['source_reads'] += bool(SOURCE.search(str(i.get('file_path') or i.get('path') or i.get('pattern') or '')))
                elif n in ('Agent', 'Task'):
                    m['subagents'] += 1
                elif n == 'Skill':
                    m['skills'].append(i.get('skill') or i.get('name'))
        if d.get('type') == 'user':
            for b in (d.get('message') or {}).get('content', []) if isinstance((d.get('message') or {}).get('content'), list) else []:
                if b.get('type') == 'tool_result' and b.get('is_error') and 'permission' in str(b.get('content')).lower():
                    m['denied'] += 1
        if d.get('type') == 'result':
            m.update(final=d.get('result'), cost_usd=d.get('total_cost_usd'), turns=d.get('num_turns'),
                     usage=d.get('usage'))
    return m


def codex_metrics(events):
    m = {'mcp_calls': 0, 'shell': 0, 'shell_failed': 0, 'source_reads': 0, 'skills_read': 0, 'final': None,
         'tokens': {'input': 0, 'cached_input': 0, 'output': 0, 'reasoning': 0}}
    for d in events:
        it = d.get('item') or {}
        if d.get('type') == 'item.completed':
            if it.get('type') == 'mcp_tool_call':
                m['mcp_calls'] += 1
            elif it.get('type') == 'command_execution':
                c = str(it.get('command'))
                m['shell'] += 1
                m['shell_failed'] += it.get('exit_code') not in (0, None)
                m['source_reads'] += bool(SOURCE.search(c))
                m['skills_read'] += 'SKILL.md' in c
            elif it.get('type') == 'agent_message':
                m['final'] = it.get('text')
        if d.get('type') == 'turn.completed':
            u = d.get('usage') or {}
            for k, kk in (('input', 'input_tokens'), ('cached_input', 'cached_input_tokens'),
                          ('output', 'output_tokens'), ('reasoning', 'reasoning_output_tokens')):
                m['tokens'][k] += u.get(kk) or 0
    return m


def check(c, calls, final):
    tools = [c['tool']] if isinstance(c.get('tool'), str) else c.get('tool') or []
    if c['type'] == 'called':
        hits = [x for x in calls if x['tool'] in tools and (x.get('ok') or not c.get('ok'))]
        return bool(hits), f'{len(hits)} call(s)'
    if c['type'] == 'not_called':
        hits = [x['tool'] for x in calls if x['tool'] in tools]
        return not hits, ('called: ' + ', '.join(sorted(set(hits)))) if hits else 'none'
    if c['type'] == 'result':
        ok = [x for x in calls if x['tool'] in tools and x.get('ok') and x.get('result') is not None]
        if not ok:
            return False, 'no successful call'
        cands = ok if c.get('which') == 'any' else ok[-1:]
        errors = []
        for x in cands:
            try:
                if eval(c['expr'], {'__builtins__': SAFE}, {'r': x['result']}):
                    return True, f'{len(ok)} successful call(s)'
            except Exception as e:
                errors.append(f'{type(e).__name__}: {e}')
        return False, '; '.join(errors) or f'{len(cands)} call(s) checked, none matched'
    if c['type'] == 'final':
        return bool(final and re.search(c['regex'], final, re.I)), 'final answer'
    return False, f'unknown check type {c["type"]}'


def score_run(run, task):
    meta = json.loads((run/'meta.json').read_text())
    events = _jsonl(run/'transcript.jsonl')
    calls = _jsonl(run/'calls.jsonl')
    m = claude_metrics(events) if meta['arm'].startswith('claude') else codex_metrics(events)
    checks = []
    for c in task.get('checks', []):
        ok, why = check(c, calls, m.get('final'))
        checks.append({'id': c['id'], 'ok': ok, 'detail': why})
    out = {'task': meta['task'], 'arm': meta['arm'], 'replicate': meta['replicate'], 'held_out': meta.get('held_out'),
           'exit': meta['exit'], 'timed_out': meta['timed_out'], 'wall_s': meta['wall_s'],
           'checks': checks, 'passed': sum(c['ok'] for c in checks), 'n_checks': len(checks),
           'ascg_calls': len(calls), 'failed_calls': sum(1 for x in calls if not x['ok'] and x.get('kind') != 'policy'),
           'refused_calls': sum(1 for x in calls if x.get('kind') == 'policy'),
           'tools_used': sorted({x['tool'] for x in calls}), **m}
    (run/'score.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    return out


def score_batch(out):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from harness import load_task
    rows = []
    for meta in sorted(out.glob('runs/*/*/meta.json')):
        run = meta.parent
        task = (json.loads((run/'task.json').read_text()) if (run/'task.json').exists() else
                load_task(json.loads(meta.read_text())['task']))
        rows.append(score_run(run, task))
    groups = {}
    for r in rows:
        groups.setdefault((r['task'], r['arm']), []).append(r)
    summary = []
    for (t, arm), rs in sorted(groups.items()):
        n = len(rs)
        summary.append({'task': t, 'arm': arm, 'runs': n, 'held_out': rs[0]['held_out'],
                        'checks_passed': round(sum(r['passed'] / max(r['n_checks'], 1) for r in rs) / n, 3),
                        'all_checks': sum(r['passed'] == r['n_checks'] for r in rs),
                        'wall_min': round(sum(r['wall_s'] for r in rs) / n / 60, 1),
                        'cost_usd': (round(sum(r.get('cost_usd') or 0 for r in rs) / n, 2)
                                     if arm.startswith('claude') else None),
                        'tokens_k': (round(sum(r['tokens']['input'] + r['tokens']['output'] for r in rs) / n / 1000)
                                     if arm == 'codex' else None),
                        'ascg_calls': round(sum(r['ascg_calls'] for r in rs) / n, 1),
                        'failed': sum(r['failed_calls'] for r in rs), 'refused': sum(r['refused_calls'] for r in rs),
                        'source_reads': sum(r['source_reads'] for r in rs), 'timeouts': sum(r['timed_out'] for r in rs),
                        'failed_checks': sorted({c['id'] for r in rs for c in r['checks'] if not c['ok']})})
    batch = json.loads((out/'batch.json').read_text())
    (out/'summary.json').write_text(json.dumps({'batch': batch, 'summary': summary}, indent=1) + '\n')
    cols = ['task', 'arm', 'runs', 'checks_passed', 'all_checks', 'wall_min', 'cost_usd', 'tokens_k', 'ascg_calls',
            'failed', 'refused', 'source_reads', 'failed_checks']
    md = [f"# Evaluation batch {out.name}", '',
          f"Tools {batch['tools_digest']} (ascg {batch['ascg_version']}, commit {batch['source_commit'][:7]}"
          f"{', dirty' if batch['source_dirty'] else ''}); {batch['claude_cli']}; {batch['codex_cli']}.", '',
          '| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    md += ['| ' + ' | '.join('' if s[c] is None else ', '.join(s[c]) if isinstance(s[c], list) else str(s[c])
                             for c in cols) + ' |' for s in summary]
    (out/'summary.md').write_text('\n'.join(md) + '\n')
    return {'runs': len(rows), 'summary': summary}


if __name__ == '__main__':
    print(json.dumps(score_batch(Path(sys.argv[1]).expanduser().resolve()), indent=1))
