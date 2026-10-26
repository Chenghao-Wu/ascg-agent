"""Default protocol (identical to ascg_core.ibi.pipeline.PROTOCOL, protocol v1 with amendments v1.1 / v1.2 in the
code; checked by tests/ascg/test_ascg.py). For the agentic arm the scripted extension rule is replaced by agent decisions
under the budget in ascg.policy, so `extension.max` is 0 and a missed convergence check stops at awaiting_decision."""
import copy

V1 = {
    'version': 'v1',
    'model': 'tdep', 'dihedrals': 'cbt',
    'stage1': {'iters': 15, 'schedule': 'pair', 'pressure_correction': 'none'},
    'stage2': {'iters': 42, 'schedule': 'pair:3,bond:1,angle:1,dihedral:1', 'pressure_correction': 'virial'},
    'npt_every': 6,
    'extension': {'iters': 18, 'max': 1, 'density_trigger': 0.04},
    'validation': {'seeds': 3, 'eq_ps': 200.0, 'prod_ps': 1000.0},
    'gates': {'density_rel_error_max': 0.05, 'rdf_nrmse_max': 0.15},
    'chain_band': {'rg_rel_error_max': 0.10},
    'infrastructure_retries': 2,
    'max_bead_types': 16,
}


def agentic(base=V1):
    p = copy.deepcopy(base)
    p['version'] = base['version'] + '-agentic'
    p['extension'] = {**p['extension'], 'max': 0}
    return p


def cycle_length(protocol, has_dihedrals=True):
    n = 0
    for part in protocol['stage2']['schedule'].split(','):
        fam, _, k = part.partition(':')
        if fam == 'dihedral' and not has_dihedrals:
            continue
        n += int(k or 1)
    return n


def continuation_plan(protocol, action, at):
    """The stages a continuation runs; the job executes exactly this list. `at` is the iteration the run continues
    from (the rewind target, or the last completed iteration for an extension). A run back inside stage 1 finishes
    stage 1 with its own schedule first; then the stage-2 schedule runs to action['target_iters'] ('extension' for
    extend actions, 'stage2' otherwise). The runner rounds the end up to a whole schedule cycle."""
    s1, s2 = protocol['stage1'], protocol['stage2']
    plan = []
    if at < s1['iters']:
        plan.append({'stage': 'stage1', 'from': at + 1, 'to': s1['iters'], 'schedule': s1['schedule'],
                     'pressure_correction': s1['pressure_correction']})
        at = s1['iters']
    plan.append({'stage': 'extension' if action['action'].startswith('extend') else 'stage2', 'from': at + 1,
                 'to': max(int(action['target_iters']), at), 'schedule': s2['schedule'],
                 'pressure_correction': s2['pressure_correction']})
    return plan


def project_cost(protocol, cost, remaining_iters, cycle=1):
    """GPU-hours: spent so far, and the projected total after `remaining_iters` more iterations and the held-out
    validation, from the measured seconds per iteration (cost: remote digest / task_state 'cost'). The upper bound
    adds the whole-cycle rounding of the runner. None when nothing has been measured yet."""
    if not cost or not cost.get('s_per_iteration'):
        return None
    s_it, v = cost['s_per_iteration'], protocol['validation']
    fit_ps = cost.get('fit_ps_per_iteration') or 550.0
    val_s = s_it * v['seeds'] * (v['eq_ps'] + v['prod_ps']) / fit_ps
    lo = cost['spent_gpu_h'] + (max(remaining_iters, 0) * s_it + val_s) / 3600
    hi = lo + (cycle - 1) * s_it / 3600
    return {'spent_gpu_h': round(cost['spent_gpu_h'], 2), 's_per_iteration': round(s_it, 1),
            'remaining_iterations': max(remaining_iters, 0), 'validation_gpu_h_est': round(val_s / 3600, 2),
            'projected_total_gpu_h': [round(lo, 2), round(hi, 2)]}
