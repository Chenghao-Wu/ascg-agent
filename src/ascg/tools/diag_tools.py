"""diag / audit: deterministic diagnostics computed next to the data (the analyses of the operator trajectory)."""
from ..compute import remote, target
from ..registry import P, tool
from ..study import Study


def _rec(study, task):
    s = Study(study)
    rec = s.task(s.load(), task)
    return rec, target(rec['target'])


@tool('diag', 'digest', 'Numbers-only history of a task: recent iterations (worst RDF, pressure, NPT density), failure, '
      'rewinds, the invocations (iterations, schedule, pressure correction, timestep) and current settings, the end of '
      'stage 1, measured cost (GPU-hours spent, seconds per iteration), job status.', {'study': P(required=True), 'task': P(required=True),
                                          'last': P('integer', default=12)})
def diag_digest(study, task, last):
    rec, t = _rec(study, task)
    return remote(t, 'digest', {'task_dir': rec['task_dir'], 'last': last}, timeout=120)


@tool('diag', 'bonds', 'Frame-by-frame bond analysis of one run: longest bond per type, single-frame excursions vs '
      'bonds beyond the break length in consecutive frames, energy up the wall, AA-sampled range, table end.', {
          'study': P(required=True), 'task': P(required=True),
          'state': P(help='state label, e.g. 400 or 400_P1000 (default: the run\'s only state)'),
          'iteration': P('integer', help='iteration of a fitting run (default: the latest with this run)'),
          'iterations': P('array', help='several iterations in one call, e.g. 1,5,8,10'),
          'run': P(help='run_T<T> (default) | npt_T<T> | final/holdout_T<T>/seed_<k>')})
def diag_bonds(study, task, state=None, iteration=None, iterations=None, run=None):
    rec, t = _rec(study, task)
    return remote(t, 'diag_bonds', {'task_dir': rec['task_dir'], 'ref': rec['ref'], 'state': state,
                                    'iteration': iteration, 'run': run,
                                    'iterations': [int(i) for i in iterations] if iterations else None},
                  timeout=1800)


@tool('diag', 'pressure', 'Measured pressure change between iterations vs the first-order virial prediction of the '
      'table change (checks the pressure correction).', {
          'study': P(required=True), 'task': P(required=True),
          'iterations': P('array', help='default: the last six iterations')})
def diag_pressure(study, task, iterations=None):
    rec, t = _rec(study, task)
    return remote(t, 'diag_pressure', {'task_dir': rec['task_dir'], 'ref': rec['ref'],
                                       'iterations': [int(i) for i in iterations] if iterations else None},
                  timeout=900)


@tool('diag', 'chains', 'CG chain size (Rg) over iterations vs the AA reference.', {
    'study': P(required=True), 'task': P(required=True),
    'iterations': P('array', help='default: BI start, end of stage 1, the in-loop NPT checks and the last (at most 8)')})
def diag_chains(study, task, iterations=None):
    rec, t = _rec(study, task)
    return remote(t, 'diag_chains', {'task_dir': rec['task_dir'], 'ref': rec['ref'],
                                     'iterations': [int(i) for i in iterations] if iterations else None},
                  timeout=900)


@tool('diag', 'structure', 'Per-type NRMSE (RDF, bonds, angles, dihedrals) at the BI start, the end of stage 1 (as '
      'run) and the last iteration.', {'study': P(required=True), 'task': P(required=True)})
def diag_structure(study, task):
    rec, t = _rec(study, task)
    return remote(t, 'diag_structure', {'task_dir': rec['task_dir']}, timeout=300)


@tool('audit', 'tables', 'Bond tables vs the harmonic walls outside the AA-sampled range; table ends and extents.', {
    'study': P(required=True), 'task': P(required=True),
    'iteration': P('integer', help='default: the final tables, or the latest iteration while the fit runs')})
def audit_tables(study, task, iteration=None):
    rec, t = _rec(study, task)
    return remote(t, 'audit_tables', {'task_dir': rec['task_dir'], 'ref': rec['ref'], 'iteration': iteration},
                  timeout=300)


@tool('diag', 'joblog', 'Tail of the task\'s latest job log (or of job_n): engine output, iterations, errors.', {
    'study': P(required=True), 'task': P(required=True), 'job_n': P('integer'), 'lines': P('integer', default=60)})
def diag_joblog(study, task, lines, job_n=None):
    rec, t = _rec(study, task)
    return remote(t, 'job_log', {'task_dir': rec['task_dir'], 'job_n': job_n, 'lines': lines}, timeout=120)
