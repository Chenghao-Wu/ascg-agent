"""trace: mine agent transcripts (Claude Code JSONL) and the study call log - repeated ad-hoc work, errors, decisions,
human turns. Input for the ascg-improve skill."""
import json
import re
from collections import Counter
from pathlib import Path

from ..registry import P, tool


def mine(path):
    calls, results, users, ts = {}, {}, [], []
    for line in Path(path).read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get('timestamp'):
            ts.append(d['timestamp'])
        m = d.get('message') or {}
        c = m.get('content')
        if d.get('type') == 'assistant' and isinstance(c, list):
            for b in c:
                if b.get('type') == 'tool_use':
                    calls[b['id']] = {'name': b['name'], 'input': b.get('input', {}), 'side': d.get('isSidechain')}
        elif d.get('type') == 'user':
            if isinstance(c, list):
                for b in c:
                    if b.get('type') == 'tool_result':
                        txt = b.get('content')
                        if isinstance(txt, list):
                            txt = ' '.join(x.get('text', '') for x in txt if isinstance(x, dict))
                        results[b['tool_use_id']] = {'err': bool(b.get('is_error')), 'text': str(txt)[:300]}
            elif isinstance(c, str) and not d.get('isMeta') and not c.startswith('<'):
                users.append(c[:200])
    tools = Counter(c['name'] for c in calls.values())
    ascg = Counter()
    other_bash = Counter()
    for c in calls.values():
        if c['name'] != 'Bash':
            continue
        cmd = c['input'].get('command', '').strip()
        m = re.match(r'ascg\s+(\w+)\s+(\w+)', cmd)
        if m:
            ascg[f'{m.group(1)} {m.group(2)}'] += 1
        else:
            other_bash[cmd.split()[0] if cmd else ''] += 1
    errs = [{'tool': calls[k]['name'], 'error': r['text'][:200]} for k, r in results.items() if r['err'] and k in calls]
    skills = Counter(c['input'].get('skill') for c in calls.values() if c['name'] == 'Skill')
    agents = Counter(c['input'].get('subagent_type') for c in calls.values() if c['name'] in ('Agent', 'Task'))
    return {'tool_calls': sum(tools.values()), 'by_tool': dict(tools.most_common()), 'ascg_calls': dict(ascg.most_common()),
            'other_bash': dict(other_bash.most_common(15)), 'errors': len(errs), 'error_samples': errs[:15],
            'skills_loaded': dict(skills), 'subagents': dict(agents), 'subagent_calls': sum(1 for c in calls.values() if c['side']),
            'user_messages': len(users), 'first': ts[0] if ts else None, 'last': ts[-1] if ts else None}


@tool('trace', 'mine', 'Summarise a Claude Code transcript (JSONL): tool use, ascg calls, other shell use, errors, skills, '
      'subagents, user turns.', {'transcript': P(required=True), 'out': P(help='write the summary JSON here')})
def trace_mine(transcript, out=None):
    res = mine(transcript)
    if out:
        Path(out).write_text(json.dumps(res, indent=2) + '\n')
    return res
