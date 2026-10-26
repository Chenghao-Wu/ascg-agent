"""amend: rule changes. Agents may propose; only a person at a terminal approves (and a hook blocks the approve
command for agents). Approved amendments are the only way to re-score with changed analysis rules."""
import json
import sys
import time

from ..compute import remote, target
from ..registry import P, PolicyError, ToolError, tool
from ..study import Study, canonical_hash


@tool('amend', 'propose', 'Propose a rule change with evidence (never applied by the proposer).', {
    'study': P(required=True), 'title': P(required=True), 'rule_change': P(required=True),
    'evidence': P(required=True), 'tasks': P('array', default=[])}, mutating=True)
def amend_propose(study, title, rule_change, evidence, tasks):
    s = Study(study)
    reg = s.load()
    aid = f'A{len(s.amendments()) + 1:02d}'
    rec = {'id': aid, 'title': title, 'rule_change': rule_change, 'evidence': evidence, 'tasks': tasks,
           'status': 'pending', 'proposed': time.strftime('%Y-%m-%d %H:%M:%S')}
    (s.dir/'amendments'/f'{aid}.json').write_text(json.dumps(rec, indent=2) + '\n')
    s.event(reg, 'amendment_proposed', None, {'id': aid, 'title': title})
    return {'id': aid, 'status': 'pending', 'next': 'the user reviews it and runs `ascg amend approve` in a terminal'}


@tool('amend', 'list', 'Amendments of a study and their status.', {'study': P(required=True)})
def amend_list(study):
    return {'amendments': Study(study).amendments()}


@tool('amend', 'approve', 'HUMAN ONLY: approve a pending amendment (interactive confirmation at a terminal).', {
    'study': P(required=True), 'id': P(required=True)}, mutating=True, mcp=False)
def amend_approve(study, id):
    if not sys.stdin.isatty():
        raise PolicyError(f'amendments are approved by a person: run `ascg amend approve --study {study} --id {id}` '
                          'yourself in a separate terminal window and type the id to confirm (agent shells, '
                          'including the `!` prefix in Claude Code, are not interactive)')
    s = Study(study)
    reg = s.load()
    f = s.dir/'amendments'/f'{id}.json'
    if not f.exists():
        raise ToolError(f'no amendment {id}')
    rec = json.loads(f.read_text())
    print(json.dumps(rec, indent=2), file=sys.stderr)
    if input(f'Type {id} to approve: ').strip() != id:
        raise PolicyError('not confirmed')
    rec.update(status='approved', approved=time.strftime('%Y-%m-%d %H:%M:%S'), protocol_hash=canonical_hash(s.protocol()))
    f.write_text(json.dumps(rec, indent=2) + '\n')
    s.event(reg, 'amendment_approved', None, {'id': id})
    return rec


@tool('amend', 'rescore', 'Re-analyse a finished task\'s held-out runs under an APPROVED amendment (no new sampling); '
      'the previous outcome is kept.', {'study': P(required=True), 'task': P(required=True), 'id': P(required=True)},
      mutating=True)
def amend_rescore(study, task, id):
    s = Study(study)
    reg = s.load()
    a = {x['id']: x for x in s.amendments()}.get(id)
    if not a or a.get('status') != 'approved':
        raise PolicyError(f'amendment {id} is not approved')
    rec = s.task(reg, task)
    res = remote(target(rec['target']), 'rescore', {'ref': rec['ref'], 'task_dir': rec['task_dir'],
                                                    'reason': f'{id}: {a["title"]}'}, timeout=1800)
    rec['outcome'] = {'class': res['class'], 'reasons': res.get('reasons', []), 'amendment': id}
    s.event(reg, 'rescored', task, {'amendment': id, 'class': res['class']})
    return res
