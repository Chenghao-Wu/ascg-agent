"""report: fetch per-task reports (checksummed) and build the study benchmark locally."""
import base64
import hashlib
import json
from pathlib import Path

from ..compute import remote, target
from ..registry import P, ToolError, tool
from ..study import Study

FILES = ('report/outcome.json', 'report/user_summary.md', 'report/technical_report.md', 'report/costs.json',
         'report/structure_density.png', 'cg/ibi/final/validation.json', 'workflow/state.json',
         'workflow/stage_timings.json', 'workflow/ascg_job.json')


def _fetch_dir(t, remote_dir, local_parent):
    """A remote directory into local_parent/<its name>, checksum-verified; returns the local path."""
    import io
    import tarfile
    r = remote(t, 'fetch_dir', {'path': remote_dir}, timeout=600)
    data = base64.b64decode(r['b64'])
    if hashlib.md5(data).hexdigest() != r['md5']:
        raise ToolError(f'checksum mismatch for {remote_dir}')
    local_parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tf:
        names = tf.getnames()
        if any(n.startswith(('/', '..')) or '/../' in n for n in names):
            raise ToolError(f'unsafe path in {remote_dir}')
        try:
            tf.extractall(local_parent, filter='data')
        except TypeError:                                  # Python without extraction filters
            tf.extractall(local_parent)
    return str(local_parent/Path(remote_dir).name)


def _fetch(t, remote_path, local_path):
    r = remote(t, 'fetch', {'path': remote_path}, timeout=300)
    data = base64.b64decode(r['b64'])
    if hashlib.md5(data).hexdigest() != r['md5']:
        raise ToolError(f'checksum mismatch for {remote_path}')
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)
    return r['md5']


@tool('report', 'task', 'Fetch a task\'s reports (outcome, plain-language summary, costs, figure, validation) into the '
      'study directory, checksum-verified; --model-files also fetches the model (tables, data file and a ready fastMD '
      'NPT input per state).', {'study': P(required=True), 'task': P(required=True),
                                'model_files': P('boolean', default=False)})
def report_task(study, task, model_files=False):
    s = Study(study)
    reg = s.load()
    rec = s.task(reg, task)
    t = target(rec['target'])
    got = {}
    for f in FILES:
        try:
            got[f] = _fetch(t, f"{rec['task_dir']}/{f}", s.dir/'tasks'/task/f)
        except ToolError as e:
            got[f] = f'missing ({str(e)[:80]})'
    summ = s.dir/'tasks'/task/'report'/'user_summary.md'
    out = {'task': task, 'files': got, 'local_dir': str(s.dir/'tasks'/task)}
    if model_files:
        out['model_files'] = _fetch_dir(t, f"{rec['task_dir']}/report/model_files", s.dir/'tasks'/task/'report')
    if summ.exists():
        out['user_summary'] = summ.read_text()
    if rec.get('status') == 'escalated':
        out['escalation'] = rec.get('outcome')
    return out


@tool('report', 'study', 'Benchmark table and figure over the study\'s fetched tasks (runs `report task` first for '
      'each finished task).', {'study': P(required=True)})
def report_study(study):
    s = Study(study)
    reg = s.load()
    for k, r in reg['tasks'].items():
        if r.get('status') in ('done', 'escalated'):
            report_task(study, k)
    from ascg_core.ibi.pipeline import aggregate
    rows = aggregate(s.dir/'tasks', s.dir/'report')
    for k, r in reg['tasks'].items():                    # escalations / adopted runs without an outcome file
        if r.get('outcome') and not any(x['task'] == k for x in rows):
            rows.append({'task': k, 'class': r['outcome']['class'], 'reasons': r['outcome'].get('reasons', []),
                         'note': 'outcome recorded in the study registry (no outcome.json in the task directory)'})
    md = s.dir/'report'/'benchmark.md'                  # aggregate() labels with the pipeline's default protocol
    if md.exists():
        from ascg_core.ibi.pipeline import PROTOCOL
        md.write_text(md.read_text().replace(f'# Protocol {PROTOCOL["version"]} benchmark',
                                             f'# Protocol {s.protocol().get("version")} benchmark', 1))
    return {'rows': rows, 'table': str(md), 'figure': str(s.dir/'report'/'benchmark.png')}
