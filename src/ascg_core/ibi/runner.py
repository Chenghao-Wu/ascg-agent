"""Multistate IBI controller with fastMD sampling.

    python -m ascg_core.ibi.runner bi  --ref runs/cg_ref/pe --out runs/ibi/pe/cg/ibi/initial
    python -m ascg_core.ibi.runner ibi --ref runs/cg_ref/pe --task-dir runs/ibi/pe --iters 20 --gpus 0,2

States: the approved (temperature, pressure) points of the fit (--states, e.g. 400,500 or 400,400@1000; N >= 1;
default: every state of the reference; ibi.states). Run directories and files carry the state label (run_T400,
targets_T400_P1000.npz). The model depends on T only; pressure sets each state's NPT runs and pressure target.

`bi` samples the Boltzmann-inverted start at each state in NVT at that state's AA mean box and writes tables, run
directories (run_T<T>: data file, config, thermo, trajectory, fastMD log), the CG distributions and metrics.json
(per-type NRMSE, pressure, timings, provenance).

`ibi` runs the multistate IBI loop in <task-dir>/cg/ibi: iter_000 is the BI start, iter_NNN holds the tables
of iteration NNN (immutable), run_T<T> per state (NVT at the AA mean box, the states in parallel on the
given GPUs, each continuing from its last configuration), distributions and metrics.json; every
--npt-every iterations npt_T<T> measure the density at 1 atm. One family is updated per
iteration (--schedule, e.g. pair x10, then bond, angle), with an optional separately tagged pressure ramp
(--pressure-correction reith). <task-dir>/workflow/state.json records progress; a rerun resumes after the
last complete iteration.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import analysis
from . import dihedrals as dmod
from . import fastmd_io as fio
from . import states as st
from .system import load_start, load_system
from .tables import (bi_potentials, bi_tpotentials, load_potentials, load_tpotentials, save_tpotentials,
                     tpotentials_at, write_tables)
from .update import (DEFAULTS, apply_correction, apply_tcorrection, log_ratio_correction, pressure_ramp,
                     table_change_derivative, virial_amplitudes, virial_pressure_change,
                     reith_amplitude, reith_amplitudes, state_corrections, tpressure_ramp)

_pretty = lambda T: st.parse_one(T).pretty()      # '400 K' / '400 K, 1000 atm'
TEMPERATURES_K = st.BENCHMARK_TEMPERATURES_K      # older runs only; a fit's states are ibi.states.of_run(state)


def stable_dt(system, pots, dt_max=0.004, per_period=25):
    """Time step from the stiffest bond: period 2 pi sqrt(mu / k), k the table curvature at its minimum."""
    dt = dt_max
    for i, k in enumerate(system.bond_types):
        p = pots['bond'][k]
        j = int(np.argmin(p.u))
        h = p.x[1] - p.x[0]
        curv = (p.u[j + 1] - 2 * p.u[j] + p.u[j - 1]) / h**2 if 0 < j < len(p.u) - 1 else 0.0
        if curv <= 0:
            continue
        a, b = system.bonds[system.bond_type == i][0]
        mu = system.masses[a] * system.masses[b] / (system.masses[a] + system.masses[b])
        period = 2 * np.pi * np.sqrt(mu / curv)
        dt = min(dt, period / per_period)
    return float(np.floor(dt / 0.0005) * 0.0005) if dt < dt_max else dt


def fastmd_provenance(binary):
    """Binary path and md5, source commit (+ '-dirty' with local changes in the source tree)."""
    import hashlib
    src = Path(binary).resolve().parent.parent
    git = lambda *c: subprocess.run(['git', '-C', str(src), *c], capture_output=True, text=True).stdout.strip()
    try:
        commit = git('rev-parse', 'HEAD')
        if commit and git('status', '--porcelain', '--untracked-files=no'):
            commit += '-dirty'
        branch = git('rev-parse', '--abbrev-ref', 'HEAD')
    except OSError:
        commit = branch = ''
    md5 = hashlib.md5(Path(binary).read_bytes()).hexdigest() if Path(binary).exists() else ''
    return {'binary': str(Path(binary).resolve()), 'commit': commit, 'branch': branch, 'md5': md5}


def sample_state(workdir, system, entries, *, pos, box, temperature_K, dt, eq_ps, prod_ps, frame_ps, seed,
                 ensemble='nvt', pressure_atm=1.0, rc=1.5, skin=0.1, gpu=None, velocities=None):
    """One fastMD run (equilibration + production in one run; frames after eq_ps are analysed).
    Returns the run directory; the final configuration is restart_final.data there."""
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    kt = fio.kT(temperature_K)
    if velocities is None:
        velocities = fio.maxwell_velocities(system.masses, temperature_K, seed)
    system.write_data(wd/'system.data', pos, box, velocities=velocities)
    nsteps = int(round((eq_ps + prod_ps) / dt))
    every = max(1, int(round(frame_ps / dt)))
    tdamp = 100 * dt
    if ensemble == 'nvt':
        ens = ('nvt_nh', kt, kt, tdamp)
    else:
        p = pressure_atm * fio.ATM_KJ_MOL_NM3
        ens = ('npt_nh', kt, kt, tdamp, p, p, 1000 * dt)
    pair_e, bond_e, angle_e, dih_e = entries
    conf = fio.conf_text(data_file='system.data', ntypes=len(system.type_names), rc=rc, skin=skin, dt=dt,
                         nsteps=nsteps, ensemble=ens, pair_tables=pair_e, bond_tables=bond_e, angle_tables=angle_e,
                         dihedral_tables=dih_e,
                         thermo=(every, 'thermo.dat'), traj=('traj.fmd', every), restart=(0, 'restart.data'),
                         extra=('pair_tpa 0',))
    (wd/'run.conf').write_text(conf)
    t0 = time.time()
    fio.run('run.conf', wd, env={'CUDA_VISIBLE_DEVICES': str(gpu)} if gpu is not None else None)
    meta = {'temperature_K': float(temperature_K), 'ensemble': ensemble, 'dt_ps': dt, 'eq_ps': eq_ps, 'prod_ps': prod_ps,
            **({'pressure_atm': float(pressure_atm)} if ensemble != 'nvt' else {}),
            'frame_ps': frame_ps, 'seed': seed, 'wall_s': time.time() - t0, 'nsteps': nsteps}
    (wd/'run.json').write_text(json.dumps(meta, indent=2) + '\n')
    return wd


def score_state(wd, system, targets, grids, *, eq_ps, dt, bond_limits=None, pair_rc=None, bond_gate=True):
    """CG distributions, NRMSE, pressure and density of a finished run; with bond_limits ({type: break length}),
    raises analysis.BrokenBondError if a bond stretched past its break length; with pair_rc also the nonbonded
    pair-distance histograms for the virial pressure correction."""
    frames = analysis.trajectory_frames(wd/'traj.fmd', skip_ps=eq_ps, dt_ps=dt)
    cg = analysis.distributions(frames, system, grids, bond_limits=bond_limits)
    if pair_rc:
        cg['pairhist'], cg['pairhist_edges'] = analysis.pair_histograms(frames, system, pair_rc)
    analysis.save(wd/'distributions.npz', cg)
    warning = None
    if bond_limits is not None:
        try:
            analysis.check_bonds(cg, bond_limits, str(wd))
        except analysis.BrokenBondError as err:
            if bond_gate:
                raise
            warning = str(err)          # protocol v1.2: in-loop NPT checks record, the fitting and validation runs stop
    th = fio.read_thermo(wd/'thermo.dat')
    keep = th['step'] * dt >= eq_ps - 1e-9
    m = analysis.metrics(cg, targets, system)
    m['pressure_bar'] = float(th['P'][keep].mean() * fio.BAR_PER_KJ_MOL_NM3)
    m['kT_mean'] = float(th['T'][keep].mean())
    m['density_g_cm3'] = float(system.density_g_cm3(cg['volume_nm3']).mean())
    m['bond_max_nm'] = cg['bond_max_nm']
    m['n_frames'] = len(frames)
    if warning:
        m['bond_warning'] = warning
    return m


def cmd_bi(a):
    ref, out = Path(a.ref), Path(a.out)
    system = load_system(ref)
    spec = a.states or a.temperatures
    temps = st.parse(spec) if spec else st.of_reference(ref)
    pots = bi_potentials(ref, system, temps, rc=a.rc, pressure_ramp_kj=a.pressure_ramp_kj)
    entries = write_tables(out/'tables', system, pots)
    dt = a.dt or stable_dt(system, pots)
    report = {'task': system.task, 'dt_ps': dt, 'fastmd': fastmd_provenance(fio.fastmd_binary()),
              'excl13_pairs': int(len(system.excl13)), 'states': {}}
    for i, T in enumerate(temps):
        targets = np.load(ref/f'targets_T{T}.npz')
        grids = analysis.Grids(targets, system)
        pos, box = load_start(ref, T)
        wd = sample_state(out/f'run_T{T}', system, entries, pos=pos, box=box, temperature_K=T, dt=dt,
                          eq_ps=a.eq_ps, prod_ps=a.prod_ps, frame_ps=a.frame_ps, seed=a.seed + i, rc=a.rc,
                          gpu=a.gpu)
        m = score_state(wd, system, targets, grids, eq_ps=a.eq_ps, dt=dt)
        m['wall_s'] = json.loads((wd/'run.json').read_text())['wall_s']
        report['states'][str(T)] = m
        print(f"{_pretty(T)}: P {m['pressure_bar']:+.0f} bar, kT {m['kT_mean']:.4f} (target {fio.kT(T):.4f}), "
              f"RDF {'/'.join(f'{v:.3f}' for v in m['rdf'].values())}, bond "
              f"{'/'.join(f'{v:.3f}' for v in m['bond'].values())}, angle "
              f"{'/'.join(f'{v:.3f}' for v in m['angle'].values())}, {m['wall_s']:.0f} s", flush=True)
    (out/'metrics.json').write_text(json.dumps(st.jsonable(report), indent=2) + '\n')


def _gpus(spec):
    """'0,2' -> [0, 2]; 'inherit' -> [None]: keep the CUDA_VISIBLE_DEVICES the scheduler set (Slurm jobs)."""
    return [None] if spec in ('inherit', '') else [int(g) for g in spec.split(',')]


def _cycle(spec, families=('pair', 'bond', 'angle', 'dihedral')):
    """'pair:3,bond:1,angle:1,dihedral:1' -> one cycle of families; families absent from the system are dropped."""
    seq = []
    for part in spec.split(','):
        fam, _, k = part.partition(':')
        if fam not in ('pair', 'bond', 'angle', 'dihedral'):
            raise SystemExit(f'--schedule: unknown family {fam!r}')
        if fam in families:
            seq += [fam] * int(k or 1)
    if not seq:
        raise SystemExit(f'--schedule {spec}: no family of this system')
    return seq


def present_families(system, dihedrals):
    """The table families the system has (molecules of two beads have no angles, of one bead no bonds): the schedule
    skips the others instead of spending an MD iteration on an update of nothing."""
    return ('pair',) + (('bond',) if len(system.bond_types) else ()) + \
        (('angle',) if len(system.angle_types) else ()) + (('dihedral',) if dihedrals else ())


def _schedule(spec, n, families=('pair', 'bond', 'angle', 'dihedral')):
    """The family updated to make each of iterations 1..n (the cycle anchored at iteration 1)."""
    seq = _cycle(spec, families)
    return [seq[i % len(seq)] for i in range(n)]


def whole_cycles(spec, n, families=('pair', 'bond', 'angle', 'dihedral')):
    """n rounded up so the run ends on the last family of a complete cycle (the loop before
    validation ends pair, pair, pair, bond, angle, dihedral)."""
    L = len(_cycle(spec, families))
    return -(-n // L) * L


def _load_distributions(path):
    d = np.load(path)
    out = {'rdf': {}, 'bond': {}, 'angle': {}, 'dihedral': {}, 'pairhist': {}}
    for k in d.files:
        if '/' in k:
            c, name = k.split('/', 1)
            out[c][name] = d[k]
        else:
            out[k] = d[k]                   # volume_nm3, pairhist_edges
    return out


def dihedral_targets(ref, task_dir, system, temps):
    """AA dihedral targets of the task (computed once into <task>/targets/dihedrals_T*.npz)."""
    tdir = Path(task_dir)/'targets'
    tdir.mkdir(parents=True, exist_ok=True)
    files = [tdir/f'dihedrals_T{T}.npz' for T in temps]
    if not all(f.exists() and f'dihedral_w/{system.dihedral_types[0]}' in np.load(f) for f in files):
        for T, d in dmod.aa_targets(ref, system.dihedrals, system.dihedral_type, system.dihedral_types, temps).items():
            np.savez_compressed(tdir/f'dihedrals_T{T}.npz', **d)
    return {T: dict(np.load(tdir/f'dihedrals_T{T}.npz')) for T in temps}


def load_targets(ref, task_dir, system, temps):
    """Mapped-AA targets per state, plus the dihedral targets when the system has dihedrals."""
    t = {T: dict(np.load(Path(ref)/f'targets_T{T}.npz')) for T in temps}
    if system.dihedrals is not None:
        for T, d in dihedral_targets(ref, task_dir, system, temps).items():
            t[T].update(d)
    return t


def cmd_ibi(a):
    ref, task_dir = Path(a.ref), Path(a.task_dir)
    ibi_dir, wf = task_dir/'cg'/'ibi', task_dir/'workflow'
    wf.mkdir(parents=True, exist_ok=True)
    system = load_system(ref)
    if a.dihedrals is None:                     # default: CBT dihedrals with the A + B T model
        a.dihedrals = 'cbt' if a.model == 'tdep' else 'none'
    a.dihedrals = None if a.dihedrals == 'none' else a.dihedrals
    if a.dihedrals:
        if a.model != 'tdep':
            raise SystemExit('--dihedrals is implemented for --model tdep (use --dihedrals none)')
        system.add_dihedrals(a.dihedrals)
        if len(system.dihedrals):
            print(f'dihedrals ({a.dihedrals}): {len(system.dihedrals)} of types {system.dihedral_types} '
                  f'(needs fastMD with dihedral_table)', flush=True)
        else:
            print('no dihedrals in this topology (no path of three bonds)', flush=True)
            a.dihedrals = None
            system.dihedrals = system.dihedral_type = system.dihedral_types = system.dihedral_style = None
    families = present_families(system, a.dihedrals)
    if not a.no_whole_cycles:
        n = whole_cycles(a.schedule, a.iters, families)
        if n != a.iters:
            print(f'--iters {a.iters} -> {n}: the run ends on a complete cycle '
                  f"({','.join(_cycle(a.schedule, families))})", flush=True)
            a.iters = n
    state_file = wf/'state.json'
    prior = json.loads(state_file.read_text()) if state_file.exists() else None
    if a.states and a.temperatures:
        raise SystemExit('give --states or --temperatures, not both')
    spec = a.states or a.temperatures
    if prior is not None:                       # a continuation keeps its own states and weights
        temps = st.of_run(prior)
        if spec and st.parse(spec) != temps:
            raise SystemExit(f'{task_dir} fits the states {st.labels(temps)}; --states {spec} needs a new task dir')
    else:
        temps = st.parse(spec) if spec else st.of_reference(ref)
    missing = st.missing_files(ref, temps)
    if missing:
        raise SystemExit(f'{ref} has no AA reference for every state: missing {missing}')
    prior_w = ((prior or {}).get('settings') or {}).get('state_weights')
    weights = ({st.parse_one(k): v for k, v in prior_w.items()} if prior_w and not (a.weights or a.w400 is not None)
               else st.weights(temps, a.weights, a.w400))
    a.states = st.labels(temps)                         # recorded in state.json settings (ibi.states.of_run)
    a.temperatures = st.temperatures(temps)
    a.pressures_atm = {s.label: s.P for s in temps}
    a.state_weights = {st.label(T): w for T, w in weights.items()}
    targets_bar = {T: T.pressure_bar for T in temps}    # each state's pressure (NVT target, NPT checks, validation)
    kts = {T: fio.kT(T) for T in temps}
    targets = load_targets(ref, task_dir, system, temps)
    grids = {T: analysis.Grids(targets[T], system) for T in temps}
    gpus = _gpus(a.gpus)
    sched = _schedule(a.schedule, a.iters, families)
    fmd = fastmd_provenance(fio.fastmd_binary())
    aa_density = {T: float(np.mean(targets[T]['density_g_cm3'])) for T in temps}
    if state_file.exists() and json.loads(state_file.read_text()).get('failed'):
        raise SystemExit(f"{task_dir} failed earlier: {json.loads(state_file.read_text())['failed']['reason']} "
                         f"(new tables need a new task dir)")
    state = prior if prior is not None else {
        'task': system.task, 'ref': str(ref.resolve()), 'fastmd': fmd, 'settings': vars(a), 'completed': -1,
        'history': []}
    if state['settings'].get('model', 'shared') != a.model:
        raise SystemExit(f"{task_dir} holds a --model {state['settings'].get('model', 'shared')} run; use a new task dir")
    if dmod.style_of(state['settings'].get('dihedrals')) != a.dihedrals:
        raise SystemExit(f"{task_dir}: --dihedrals must match the existing run "
                         f"({dmod.style_of(state['settings'].get('dihedrals')) or 'none'}); use a new task dir")
    state.setdefault('invocations', []).append({'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                                                'start_iteration': state['completed'] + 1, 'fastmd': fmd,
                                                'settings': vars(a)})

    def run_states(it_dir, entries, starts, dt, eq_ps, ensemble='nvt', prefix='run'):
        """entries: {T: table entries of that state}."""
        def one(i_T):
            i, T = i_T
            pos, box, vel = starts[T]
            return T, sample_state(it_dir/f'{prefix}_T{T}', system, entries[T], pos=pos, box=box, temperature_K=T,
                                   dt=dt, eq_ps=eq_ps, prod_ps=a.prod_ps, frame_ps=a.frame_ps, seed=a.seed + 10 * i,
                                   ensemble=ensemble, pressure_atm=float(T.P), rc=a.rc, skin=a.skin,
                                   gpu=gpus[i % len(gpus)], velocities=vel)
        with ThreadPoolExecutor(len(temps)) as ex:
            return dict(ex.map(one, enumerate(temps)))

    tdep = a.model == 'tdep'
    pots = None
    for it in range(state['completed'] + 1, a.iters + 1):
        it_dir = ibi_dir/f'iter_{it:03d}'
        t0 = time.time()
        if it == 0:
            if tdep:
                pots = bi_tpotentials(ref, system, temps, rc=a.rc, r_switch=a.r_switch, weights=weights)
                if system.dihedrals is not None:
                    pots['dihedral'] = dmod.bi_tdihedrals(targets, system.dihedral_types, kts,
                                                          style=system.dihedral_style, weights=weights)
            else:
                pots = bi_potentials(ref, system, temps, rc=a.rc, r_switch=a.r_switch)
            starts = {T: (*load_start(ref, T), None) for T in temps}
            family, amp = 'bi', 0.0
        else:
            prev = ibi_dir/f'iter_{it - 1:03d}'
            pots = load_tpotentials(prev/'tables') if tdep else load_potentials(prev/'tables')
            cg = {T: _load_distributions(prev/f'run_T{T}'/'distributions.npz') for T in temps}
            pm = json.loads((prev/'metrics.json').read_text())
            family = sched[it - 1]
            old_pair = dict(pots['pair'])
            kw = dict(lam=a.lam, cap=a.cap, floor_rel=DEFAULTS['floor_rel'], smooth_bins=a.smooth_bins)
            g0 = grids[temps[0]]                 # the observable grids are common to the states
            if tdep and family == 'dihedral':
                centres = targets[temps[0]]['phi']
                for k in system.dihedral_types:
                    corr = {T: dmod.state_correction(
                        centres, cg[T]['dihedral'][k], targets[T][f'dihedral/{k}'], kts[T], lam=a.lam, cap=a.cap,
                        smooth_bins=a.smooth_bins,
                        gain=dmod.cbt_gain(targets[T], k) if system.dihedral_style == 'cbt' else None) for T in temps}
                    pots['dihedral'][k] = dmod.apply_tdihedral(pots['dihedral'][k], centres, corr, weights)
            elif tdep:                   # per-state corrections mapped onto A and B
                obs = {'pair': ('rdf', system.pair_keys, lambda k: g0.r),
                       'bond': ('bond', system.bond_types,
                                lambda k: 0.5 * (g0.bond_edges[k][1:] + g0.bond_edges[k][:-1])),
                       'angle': ('angle', system.angle_types, lambda k: g0.theta)}
                key, names, xfun = obs[family]
                for k in names:
                    corr = state_corrections(xfun(k), {T: cg[T][key][k] for T in temps},
                                             {T: targets[T][f'{key}/{k}'] for T in temps}, kts, **kw)
                    pots[family][k] = apply_tcorrection(pots[family][k], xfun(k), corr, r_switch=a.r_switch, rc=a.rc,
                                                        weights=weights)
            elif family == 'pair':
                for k in system.pair_keys:
                    corr, _ = log_ratio_correction(g0.r, {T: cg[T]['rdf'][k] for T in temps},
                                                   {T: targets[T][f'rdf/{k}'] for T in temps}, kts, weights, **kw)
                    pots['pair'][k] = apply_correction(pots['pair'][k], g0.r, corr, r_switch=a.r_switch, rc=a.rc)
            elif family == 'bond':
                for k in system.bond_types:
                    e = g0.bond_edges[k]
                    corr, _ = log_ratio_correction(0.5 * (e[1:] + e[:-1]), {T: cg[T]['bond'][k] for T in temps},
                                                   {T: targets[T][f'bond/{k}'] for T in temps}, kts, weights, **kw)
                    pots['bond'][k] = apply_correction(pots['bond'][k], 0.5 * (e[1:] + e[:-1]), corr)
            else:
                for k in system.angle_types:
                    corr, _ = log_ratio_correction(g0.theta, {T: cg[T]['angle'][k] for T in temps},
                                                   {T: targets[T][f'angle/{k}'] for T in temps}, kts, weights, **kw)
                    pots['angle'][k] = apply_correction(pots['angle'][k], g0.theta, corr)
            amp = 0.0
            pc = a.pressure_correction
            if pc == 'virial' and not all(cg[T]['pairhist'] for T in temps):
                print(f'iteration {it - 1} has no pair histograms: Reith step instead of the virial one', flush=True)
                pc = 'reith'
            if pc == 'virial':
                p_by_T = {T: pm['states'][str(T)]['pressure_bar'] for T in temps}
                hist = {T: cg[T]['pairhist'] for T in temps}
                edges = cg[temps[0]]['pairhist_edges']
                vol = {T: float(np.mean(cg[T]['volume_nm3'])) for T in temps}
                dp_step = {T: virial_pressure_change(hist[T], edges, vol[T], {
                    k: table_change_derivative(old_pair[k], pots['pair'][k], T if tdep else None)
                    for k in system.pair_keys}) for T in temps}
                amp, resp = virial_amplitudes(p_by_T, dp_step, hist, edges, vol, kts, r_switch=a.r_switch, rc=a.rc,
                                              scale=a.pressure_scale, cap_kt=a.pressure_cap, target_bar=targets_bar)
                if not tdep:
                    amp = sum(weights[T] * amp[T] for T in temps)
                pressure_info = {'method': 'virial', 'dp_step_pred_bar': dp_step, 'response_bar_per_kj': resp,
                                 'predicted_bar': {T: p_by_T[T] + dp_step[T] + resp[T] * (amp[T] if tdep else amp)
                                                   for T in temps}}
                for k in system.pair_keys:
                    pots['pair'][k] = (tpressure_ramp(pots['pair'][k], amp, r_switch=a.r_switch, rc=a.rc,
                                                      weights=weights) if tdep
                                       else pressure_ramp(pots['pair'][k], amp, r_switch=a.r_switch, rc=a.rc))
            if pc == 'reith':
                p_by_T = {T: pm['states'][str(T)]['pressure_bar'] for T in temps}
                if tdep:                 # one ramp per state, mapped onto A and B
                    amp = reith_amplitudes(p_by_T, kts, target_bar=targets_bar)
                    for k in system.pair_keys:
                        pots['pair'][k] = tpressure_ramp(pots['pair'][k], amp, r_switch=a.r_switch, rc=a.rc,
                                                         weights=weights)
                else:
                    amp = reith_amplitude(p_by_T, kts, weights, target_bar=targets_bar)
                    for k in system.pair_keys:
                        pots['pair'][k] = pressure_ramp(pots['pair'][k], amp, r_switch=a.r_switch, rc=a.rc)
            starts = {T: fio.read_restart(prev/f'run_T{T}'/'restart_final.data') for T in temps}
        if tdep:
            save_tpotentials(it_dir/'tables', pots)
            state_pots = {T: tpotentials_at(pots, T) for T in temps}
            entries = {T: write_tables(it_dir/f'tables_T{T}', system, state_pots[T]) for T in temps}
            dt = a.dt or min(stable_dt(system, state_pots[T]) for T in temps)
        else:
            e = write_tables(it_dir/'tables', system, pots)
            entries = {T: e for T in temps}
            state_pots = {T: pots for T in temps}
            dt = a.dt or stable_dt(system, pots)
        limits = {T: analysis.break_lengths(state_pots[T]['bond'], T) for T in temps}
        eq = a.first_eq_ps if it == 0 else a.eq_ps
        runs = run_states(it_dir, entries, starts, dt, eq)
        report = {'iteration': it, 'family_updated': family, 'pressure_ramp_kj': amp, 'dt_ps': dt,
                  'pressure_correction': pressure_info if it and pc == 'virial' else (pc if it else 'none'),
                  'bond_break_nm': {str(T): limits[T] for T in temps}, 'states': {}}

        def scored(wd, T, eq_ps, pairs=False, gate=True):
            try:
                return score_state(wd, system, targets[T], grids[T], eq_ps=eq_ps, dt=dt, bond_limits=limits[T],
                                   pair_rc=a.rc if pairs else None, bond_gate=gate)
            except analysis.BrokenBondError as err:         # deterministic gate: stop the task, say why
                state['failed'] = {'iteration': it, 'temperature_K': T, 'reason': str(err),
                                   'time': time.strftime('%Y-%m-%d %H:%M:%S')}
                state_file.write_text(json.dumps(st.jsonable(state), indent=2) + '\n')
                raise SystemExit(f'TASK FAILED (broken bonds) at iteration {it}, {_pretty(T)}: {err}')
        for T in temps:
            report['states'][str(T)] = scored(runs[T], T, eq, pairs=True)
        if a.npt_every and (it % a.npt_every == 0 or it == a.iters):
            npt_starts = {T: fio.read_restart(runs[T]/'restart_final.data') for T in temps}
            nruns = run_states(it_dir, entries, npt_starts, dt, a.npt_eq_ps, ensemble='npt', prefix='npt')
            for T in temps:
                m = scored(nruns[T], T, a.npt_eq_ps, gate=False)
                if 'bond_warning' in m:
                    print(f'iteration {it} NPT check {_pretty(T)}: {m["bond_warning"]} (recorded, not a stop: v1.2)', flush=True)
                m['density_error'] = (m['density_g_cm3'] - aa_density[T]) / aa_density[T]
                report['states'][str(T)]['npt'] = m
        report['wall_s'] = time.time() - t0
        (it_dir/'metrics.json').write_text(json.dumps(st.jsonable(report), indent=2) + '\n')
        state['completed'] = it
        state['history'].append({'iteration': it, 'family': family, 'pressure_ramp_kj': amp,
                                 **{f'rdf_{T}': report['states'][str(T)]['rdf'] for T in temps},
                                 **{f'p_bar_{T}': report['states'][str(T)]['pressure_bar'] for T in temps},
                                 **({f'density_error_{T}': report['states'][str(T)]['npt']['density_error'] for T in temps}
                                    if 'npt' in report['states'][str(temps[0])] else {}),
                                 'wall_s': report['wall_s']})
        state_file.write_text(json.dumps(st.jsonable(state), indent=2) + '\n')
        line = '  '.join(f"{_pretty(T)}: RDF {'/'.join(f'{v:.3f}' for v in report['states'][str(T)]['rdf'].values())} "
                         f"bond {'/'.join(f'{v:.3f}' for v in report['states'][str(T)]['bond'].values())} "
                         f"angle {'/'.join(f'{v:.3f}' for v in report['states'][str(T)]['angle'].values())} "
                         + (f"dih {'/'.join(f'{v:.3f}' for v in report['states'][str(T)]['dihedral'].values())} "
                            if 'dihedral' in report['states'][str(T)] else '')
                         + f"P {report['states'][str(T)]['pressure_bar']:+.0f}"
                         + (f" rho {100 * report['states'][str(T)]['npt']['density_error']:+.1f}%"
                            if 'npt' in report['states'][str(T)] else '') for T in temps)
        amp_txt = (' '.join(f'{v:+.3f}' for v in amp.values()) if isinstance(amp, dict) else f'{amp:+.3f}')
        print(f'iter {it:3d} [{family}] A {amp_txt}  {line}  ({report["wall_s"]:.0f} s)', flush=True)


def _jsd(p, q, dx):
    """Jensen-Shannon divergence (base 2, in [0, 1]) of two densities on the same grid."""
    p, q = np.clip(np.asarray(p) * dx, 0, None), np.clip(np.asarray(q) * dx, 0, None)
    if p.sum() <= 0 or q.sum() <= 0:
        return float('nan')
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    def kl(a):
        k = a > 0
        return float(np.sum(a[k] * np.log2(a[k] / m[k])))
    return 0.5 * kl(p) + 0.5 * kl(q)


def cmd_validate(a):
    """Held-out NPT runs (fresh starts from the mapped-AA configuration, own seeds, never used in
    fitting) with the final tables; density error <= 5 % and every pair RDF NRMSE <= 0.15 at both temperatures,
    plus bond / angle JSD and Rg / Ree against the AA references."""
    from ..mapping.cg_mapping import block_se, chain_size
    import shutil
    ref, task_dir = Path(a.ref), Path(a.task_dir)
    ibi_dir = task_dir/'cg'/'ibi'
    state = json.loads((task_dir/'workflow'/'state.json').read_text())
    it = a.iteration if a.iteration is not None else state['completed']
    src = ibi_dir/f'iter_{it:03d}'
    final = ibi_dir/'final'
    system = load_system(ref)
    if dmod.style_of(state['settings'].get('dihedrals')):
        system.add_dihedrals(dmod.style_of(state['settings'].get('dihedrals')))
    temps = st.of_run(state)                                       # the states the tables were fitted to
    all_targets = load_targets(ref, task_dir, system, temps)
    summary = json.loads((ref/'summary.json').read_text())
    if (src/'tables'/'potentials_tdep.json').exists():          # A + B T: each state's own tables
        tp = load_tpotentials(src/'tables')
        save_tpotentials(final/'tables', tp)
        sp = {T: tpotentials_at(tp, T) for T in temps}
        entries = {T: write_tables(final/f'tables_T{T}', system, sp[T]) for T in temps}
    else:
        shared = load_potentials(src/'tables')
        e = write_tables(final/'tables', system, shared)
        entries = {T: e for T in temps}
        sp = {T: shared for T in temps}
    limits = {T: analysis.break_lengths(sp[T]['bond'], T) for T in temps}
    dt = json.loads((src/'metrics.json').read_text())['dt_ps']
    gpus = _gpus(a.gpus)
    jobs = [(T, k) for T in temps for k in range(a.seeds)]

    def one(i_job):
        i, (T, k) = i_job
        wd = final/f'holdout_T{T}'/f'seed_{k}'
        if a.rescore and (wd/'traj.fmd').exists() and (wd/'run.json').exists():
            return (T, k), wd                     # re-analyse the existing held-out run (no new sampling)
        pos, box = load_start(ref, T)
        return (T, k), sample_state(final/f'holdout_T{T}'/f'seed_{k}', system, entries[T], pos=pos, box=box,
                                    temperature_K=T, dt=dt, eq_ps=a.eq_ps, prod_ps=a.prod_ps, frame_ps=a.frame_ps,
                                    seed=a.seed + 1000 * (k + 1) + int(round(float(T))), ensemble='npt',
                                    pressure_atm=float(T.P), rc=a.rc, skin=a.skin,
                                    gpu=gpus[i % len(gpus)])
    with ThreadPoolExecutor(min(len(jobs), len(gpus))) as ex:
        runs = dict(ex.map(one, enumerate(jobs)))
    report = {'iteration': it, 'tables': str(src/'tables'), 'seeds': a.seeds, 'eq_ps': a.eq_ps, 'prod_ps': a.prod_ps,
              'dt_ps': dt,
              'criteria': {'density_rel_error_max': 0.05, 'rdf_nrmse_max': 0.15}, 'states': st.labels(temps),
              'state_points': [s.to_dict() for s in temps],
              'states': {}}
    for T in temps:
        targets = all_targets[T]
        grids = analysis.Grids(targets, system)
        aa_rho = float(np.mean(targets['density_g_cm3']))
        aa_st = summary['states'][str(T)]
        per_seed = []
        for k in range(a.seeds):
            wd = runs[(T, k)]
            frames = analysis.trajectory_frames(wd/'traj.fmd', skip_ps=a.eq_ps, dt_ps=dt)
            cg = analysis.distributions(frames, system, grids, bond_limits=limits[T])
            analysis.save(wd/'distributions.npz', cg)
            try:
                analysis.check_bonds(cg, limits[T], str(wd))
            except analysis.BrokenBondError as err:
                report['validated'] = False
                report['failed'] = str(err)
                (final/'validation.json').write_text(json.dumps(st.jsonable(report), indent=2) + '\n')
                raise SystemExit(f'VALIDATION FAILED (broken bonds): {err}')
            rho = system.density_g_cm3(cg['volume_nm3'])
            rg, ree = zip(*[chain_size(p, system.masses, system.mol) for p, _ in frames])
            m = analysis.metrics(cg, targets, system)
            m.update(density_g_cm3=float(rho.mean()), density_se=float(block_se(rho, 10)),
                     density_error=float((rho.mean() - aa_rho) / aa_rho),
                     bond_jsd={b: _jsd(cg['bond'][b], targets[f'bond/{b}'], np.diff(grids.bond_edges[b]).mean())
                               for b in system.bond_types},
                     angle_jsd={g: _jsd(cg['angle'][g], targets[f'angle/{g}'], np.diff(grids.angle_edges).mean())
                                for g in system.angle_types},
                     **({'dihedral_jsd': {g: _jsd(cg['dihedral'][g], targets[f'dihedral/{g}'], 2 * np.pi / dmod.NBINS)
                                          for g in system.dihedral_types}} if 'dihedral' in cg else {}),
                     rg_nm=float(np.mean([r.mean() for r in rg])),
                     ree_rms_nm=float(np.mean([np.sqrt((e**2).mean()) for e in ree])))
            m['pass'] = bool(abs(m['density_error']) <= 0.05 and max(m['rdf'].values()) <= 0.15)
            per_seed.append(m)
        report['states'][str(T)] = {
            'aa': {'density_g_cm3': aa_rho, 'rg_nm': aa_st['rg_nm'], 'ree_rms_nm': aa_st['ree_rms_nm']},
            'seeds': per_seed,
            'density_error_mean': float(np.mean([m['density_error'] for m in per_seed])),
            'rdf_nrmse_worst': float(max(max(m['rdf'].values()) for m in per_seed)),
            'pass_all_seeds': all(m['pass'] for m in per_seed)}
        r = report['states'][str(T)]
        seeds_txt = ', '.join(f"{100 * m['density_error']:+.1f}" for m in per_seed)
        print(f"{_pretty(T)}: density error {100 * r['density_error_mean']:+.1f} % (seeds {seeds_txt}), worst pair RDF NRMSE "
              f"{r['rdf_nrmse_worst']:.3f}, Rg {np.mean([m['rg_nm'] for m in per_seed]):.3f} (AA {aa_st['rg_nm']:.3f}), "
              f"Ree {np.mean([m['ree_rms_nm'] for m in per_seed]):.3f} (AA {aa_st['ree_rms_nm']:.3f}), "
              f"pass {r['pass_all_seeds']}", flush=True)
    report['validated'] = all(report['states'][str(T)]['pass_all_seeds'] for T in temps)
    (final/'validation.json').write_text(json.dumps(st.jsonable(report), indent=2) + '\n')
    print('VALIDATED' if report['validated'] else 'NOT VALIDATED', flush=True)


def cmd_rewind(a):
    """Rewind a task to a complete earlier iteration (agent recovery tool): later iterations are moved to
    cg/ibi/rewound_<time>/, the history truncated, a recorded failure cleared; the next `ibi` call continues from
    there (with changed settings, so the replaced tables differ). Every rewind is recorded in state['rewinds']."""
    import shutil
    task_dir = Path(a.task_dir)
    ibi_dir, state_file = task_dir/'cg'/'ibi', task_dir/'workflow'/'state.json'
    state = json.loads(state_file.read_text())
    k = a.to_iteration
    if not 0 <= k <= state['completed'] or not (ibi_dir/f'iter_{k:03d}'/'metrics.json').exists():
        raise SystemExit(f'cannot rewind to iteration {k} (completed {state["completed"]})')
    stamp = time.strftime('%Y%m%d_%H%M%S')
    arch = ibi_dir/f'rewound_{stamp}'
    arch.mkdir()
    for d in sorted(ibi_dir.glob('iter_*')):
        if int(d.name.split('_')[1]) > k:
            shutil.move(str(d), arch/d.name)
    if (ibi_dir/'final').exists():
        shutil.move(str(ibi_dir/'final'), arch/'final')
    state.setdefault('rewinds', []).append({'time': stamp, 'from': state['completed'], 'to': k,
                                            'failed': state.pop('failed', None), 'reason': a.reason, 'archive': str(arch)})
    state['completed'] = k
    state['history'] = [e for e in state['history'] if e['iteration'] <= k]
    state_file.write_text(json.dumps(st.jsonable(state), indent=2) + '\n')
    print(f'rewound {task_dir} to iteration {k} (later iterations in {arch})', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('bi', help='sample the Boltzmann-inverted start at each state')
    b.add_argument('--ref', required=True)
    b.add_argument('--states', default=None, help='state points, e.g. 400,500 or 400@1000 (K, atm; default: all of '
                                                  'the reference)')
    b.add_argument('--temperatures', default=None, help='older form of --states (temperatures at 1 atm)')
    b.add_argument('--out', required=True)
    b.add_argument('--rc', type=float, default=1.5)
    b.add_argument('--pressure-ramp-kj', type=float, default=0.0)
    b.add_argument('--dt', type=float, default=None, help='ps (default: from the stiffest bond, <= 4 fs)')
    b.add_argument('--eq-ps', type=float, default=100.0)
    b.add_argument('--prod-ps', type=float, default=500.0)
    b.add_argument('--frame-ps', type=float, default=1.0)
    b.add_argument('--seed', type=int, default=1)
    b.add_argument('--gpu', type=int, default=None)
    c = sub.add_parser('ibi', help='multistate IBI loop')
    c.add_argument('--ref', required=True)
    c.add_argument('--task-dir', required=True, help='runs/<task_id>; writes cg/ibi/iter_NNN and workflow/state.json')
    c.add_argument('--iters', type=int, default=20)
    c.add_argument('--states', default=None,
                   help='the approved state points: T in K, optionally @P in atm (or bar with the unit), e.g. 400,500 '
                        'or 450 or 400,400@1000 (N >= 1; each needs targets_T<label>, start_T<label>, mapped_T<label> '
                        'in the reference). Default: every state of the reference; a continuation keeps its own')
    c.add_argument('--temperatures', default=None, help='older form of --states (temperatures at 1 atm)')
    c.add_argument('--weights', default=None, help="state weights, e.g. '0.3,0.7' (order of the states) or "
                                                   "'400:0.3,400_P1000:0.7'; default equal")
    c.add_argument('--dihedrals', default=None, choices=dmod.STYLES + ('none',),
                   help='tabulated CG dihedrals on every path of three bonds (fastMD dihedral_table extension): '
                        'cbt (default with --model tdep: sin^3 theta1 sin^3 theta2 U(phi)), '
                        'table (plain U(phi)) or none')
    c.add_argument('--model', choices=('shared', 'tdep'), default='tdep',
                   help='tdep (default): U(x, T) = A(x) + B(x) T; shared: one potential set '
                        'for all states')
    c.add_argument('--schedule', default='pair', help="families updated in turn, e.g. 'pair:3,bond:1,angle:1,dihedral:1'"
                   ' (dihedral is dropped without dihedrals)')
    c.add_argument('--no-whole-cycles', action='store_true',
                   help='stop at exactly --iters (default: round up so the run ends on a complete schedule cycle)')
    c.add_argument('--lam', type=float, default=DEFAULTS['lam'], help='damping lambda (protocol v1: 0.2)')
    c.add_argument('--cap', type=float, default=DEFAULTS['cap'], help='log-ratio cap c (protocol v1: 2)')
    c.add_argument('--w400', type=float, default=None, help='older two-state form of --weights (400 / 500 K only)')
    c.add_argument('--smooth-bins', type=float, default=DEFAULTS['smooth_bins'])
    c.add_argument('--pressure-correction', choices=('none', 'reith', 'virial'), default='none',
                   help='pressure (density) ramp A (1 - r/rc) S(r): reith (fixed small steps) or virial (protocol v1: '
                        'amplitude from the first-order virial response, cancelling the predicted pressure change of '
                        'the same iteration)')
    c.add_argument('--pressure-scale', type=float, default=0.8, help='virial: fraction of the deviation cancelled')
    c.add_argument('--pressure-cap', type=float, default=0.3, help='virial: max |A| per iteration in k_B T')
    c.add_argument('--npt-every', type=int, default=5, help='NPT density check every N iterations (0 = never)')
    c.add_argument('--npt-eq-ps', type=float, default=100.0, help='NPT equilibration before the density average')
    c.add_argument('--rc', type=float, default=1.5)
    c.add_argument('--r-switch', type=float, default=1.3)
    c.add_argument('--skin', type=float, default=0.2)
    c.add_argument('--dt', type=float, default=None)
    c.add_argument('--first-eq-ps', type=float, default=100.0)
    c.add_argument('--eq-ps', type=float, default=50.0)
    c.add_argument('--prod-ps', type=float, default=500.0)
    c.add_argument('--frame-ps', type=float, default=1.0)
    c.add_argument('--seed', type=int, default=1)
    c.add_argument('--gpus', default='0', help="comma-separated GPU ids (the states run in parallel), or "
                                               "'inherit' to keep the scheduler's CUDA_VISIBLE_DEVICES")
    v = sub.add_parser('validate', help='held-out NPT validation of the final tables')
    v.add_argument('--ref', required=True)
    v.add_argument('--task-dir', required=True)
    v.add_argument('--iteration', type=int, default=None, help='tables of this iteration (default: the last one)')
    v.add_argument('--seeds', type=int, default=3)
    v.add_argument('--eq-ps', type=float, default=200.0)
    v.add_argument('--prod-ps', type=float, default=1000.0)
    v.add_argument('--frame-ps', type=float, default=2.0)
    v.add_argument('--rc', type=float, default=1.5)
    v.add_argument('--skin', type=float, default=0.2)
    v.add_argument('--seed', type=int, default=7)
    v.add_argument('--gpus', default='0', help="GPU ids or 'inherit'")
    v.add_argument('--rescore', action='store_true',
                   help='re-analyse existing held-out runs (same tables, no sampling), e.g. after a gate amendment')
    w = sub.add_parser('rewind', help='rewind a task to an earlier complete iteration (recovery tool)')
    w.add_argument('--task-dir', required=True)
    w.add_argument('--to-iteration', type=int, required=True)
    w.add_argument('--reason', default='')
    a = ap.parse_args(argv)
    {'bi': cmd_bi, 'ibi': cmd_ibi, 'validate': cmd_validate, 'rewind': cmd_rewind}[a.cmd](a)


if __name__ == '__main__':
    main()
