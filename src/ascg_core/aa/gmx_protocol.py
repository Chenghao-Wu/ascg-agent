"""AA protocol v3 in GROMACS.

Per polymer (1000 CG beads, AutoPoly OPLS-AA converted by lmp2gmx):
  em_sc, relax_sc  soft-core minimization + 20 ps soft-core relaxation (lambda 0.95 -> 0, 300 K)
            in AutoPoly's dilute box: removes the near-coincident atoms of the AutoPoly build
  em        steepest descent with the full potential
  eq21      RadonPy EQ21 at the hottest approved state (500 K for the benchmark): 21 stages (hot NVT at
            max(600 K, T + 100 K), NVT at T, NPT at 1000-50000 atm, last 800 ps NPT at 1 atm), stage times as in
            RadonPy (ps); v-rescale + C-rescale
  T<label>  one branch per approved state point (T, P) (ibi.states; benchmark: T400, T500 at 1 atm; e.g. T400_P1000
            at 1000 atm) from the eq21 end state: C-rescale pre-equilibration at (T, P), then 10 ns NPT equilibration
            + 10 ns NPT production at (T, P), Nose-Hoover + Parrinello-Rahman, fresh velocities; frames every 10 ps.
            GROMACS takes bar: P_bar = P_atm x 1.01325 (both recorded)
Molecular liquids use direct equilibration instead of EQ21:
  equil     soft-core minimization + relaxation, EM, 100 ps NVT at T, 500 ps C-rescale NPT compression at
            1000 atm from AutoPoly's dilute box to liquid density, 500 ps C-rescale NPT at the state pressure
  T<label>  the same branches as the polymers from the equil end state (pre, eq, prod)
All MD: flexible bonds, 1 fs (H-bond constraints at 2 fs crashed POM in
both AutoPoly and LigParGen OPLS-AA, and at 1 fs constraints are not needed), 1.5 nm cut-offs
(1.2 nm in the high-pressure EQ21 stages), PME, no dispersion correction (as the LAMMPS lj/cut
setup it replaces). Branch thermostat/barostat applied every 3 steps (see BRANCH_COUPLING).
"""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from .eq21 import RadonPy21Config, stages

EQ21_TEMPERATURE_K = 500.0                 # the benchmark's EQ21 (its hottest state); new studies: max(approved)
BRANCH_TEMPERATURES_K = (400.0, 500.0)     # the benchmark's approved pair; write_protocol takes the approved states
BAR_PER_ATM = 1.01325
DT_PS = 0.001
DT_SOFT_PS = 0.001
DT_EQ21_PS = 0.001

COMMON = {
    'cutoff-scheme': 'Verlet', 'pbc': 'xyz',
    'coulombtype': 'PME', 'rcoulomb': 1.5, 'fourierspacing': 0.16, 'ewald-rtol': 1e-5,
    'vdwtype': 'Cut-off', 'vdw-modifier': 'Potential-shift', 'rvdw': 1.5, 'DispCorr': 'no',
    'constraints': 'none',
    'nstcalcenergy': 100, 'nstenergy': 500, 'nstlog': 5000,
}
EQ21_COUPLING = {'tcoupl': 'v-rescale', 'tc-grps': 'System', 'tau-t': 0.1,
                 'pcoupltype': 'isotropic', 'tau-p': 2.0, 'compressibility': 4.5e-5}
# EQ21 starts from a ~0.01 g/cm3 box: GROMACS's automatic Verlet buffer assumes a uniform density,
# chose nstlist 100 with rlist = rc and atoms inside the compact chains were missed (crash at 600 K).
EQ21_PAIRLIST = {'verlet-buffer-tolerance': -1, 'rlist': 1.6, 'nstlist': 10}
# GROMACS needs rlist < L/2 (LAMMPS did not); at 30000-50000 atm the smallest boxes (POM, PA, PAN)
# shrink to ~3.1 nm, so EQ21 stages 1-20 use a 1.2 nm cut-off (PME grid scaled); stage 21 (1 atm)
# and the branches, which produce the reference data, use the 1.5 nm force field.
EQ21_SHORT_CUTOFF = {'rcoulomb': 1.2, 'rvdw': 1.2, 'fourierspacing': 0.14,
                     'verlet-buffer-tolerance': -1, 'rlist': 1.3, 'nstlist': 10}
# Nose-Hoover + Parrinello-Rahman are applied every nstpcouple steps (GROMACS default 10; LAMMPS
# updates every step). The update resonates when that interval is near a multiple of half the
# period of the fastest bonded motion: flexible bonds at 1 fs, every 10 steps (C-H stretch
# ~9-11.5 fs) -> PS box 4.5 -> 247 nm in 100 ps; H-bond constraints at 2 fs, every 10 steps (C-C
# stretch, H-C-X bends ~20-25 fs) -> density 869 +- 71 kg/m3, T SD 33 K. Stable: flexible 1 fs
# every step or every 7 steps; constrained 2 fs every 1 or 3 steps.
# Every 3 steps at 1 fs = 3 fs, below the lowest X-H resonance (~4.6 fs). Coupling times are
# GROMACS production values: tau_t 1 ps (1000 steps), tau_p 5 ps (5000 steps).
# nstcalcenergy = coupling interval, so energies every 999 steps (0.999 ps), log every 9999.
BRANCH_COUPLING = {'tcoupl': 'nose-hoover', 'tc-grps': 'System', 'tau-t': 1.0,
                   'pcoupl': 'Parrinello-Rahman', 'pcoupltype': 'isotropic', 'tau-p': 5.0,
                   'compressibility': 4.5e-5, 'ref-p': BAR_PER_ATM, 'nsttcouple': 3, 'nstpcouple': 3,
                   'nstcalcenergy': 3, 'nstlog': 9999}
BRANCH_NSTENERGY = 999


def mdp_text(params: dict, comment: str) -> str:
    return f'; {comment}\n' + ''.join(f'{k:24s}= {v}\n' for k, v in params.items())


FLEXIBLE = {'constraints': 'none'}


def em_params(nsteps=50000, emtol=500.0):
    p = dict(COMMON)
    p.update({'integrator': 'steep', 'nsteps': nsteps, 'emtol': emtol, 'emstep': 0.001, **FLEXIBLE})
    return p


def soft_core_steps(*, relax_ps=20.0, lambda_start=0.95, temperature=300.0, seed=7):
    """Overlap removal before EQ21 (replaces a LAMMPS soft-potential step):
    minimization with all non-bonded interactions soft (lambda_start), then MD at 300 K while
    lambda goes to 0 (full potential) with soft-core LJ and Coulomb; the B-state topology
    comes from lmp2gmx.add_decoupled_states. AutoPoly's rigid-unit growth leaves atoms
    0.04-1 A apart that plain minimization cannot separate (infinite or trapped forces)."""
    fe = {'free-energy': 'yes', 'sc-alpha': 0.5, 'sc-power': 1, 'sc-r-power': 6, 'sc-sigma': 0.3,
          'sc-coul': 'yes', 'nstdhdl': 0, 'separate-dhdl-file': 'no', 'dhdl-derivatives': 'no',
          'calc-lambda-neighbors': 0, **EQ21_PAIRLIST}
    base = dict(COMMON)
    em = dict(base, integrator='steep', nsteps=20000, emtol=1000.0, emstep=0.001, **fe, **FLEXIBLE,
              **{'init-lambda': lambda_start})
    n = int(round(relax_ps / DT_SOFT_PS))
    md = dict(base, integrator='md', dt=DT_SOFT_PS, nsteps=n, **fe, **FLEXIBLE, **{'init-lambda': lambda_start,
              'delta-lambda': -lambda_start / n, 'tcoupl': 'v-rescale', 'tc-grps': 'System', 'tau-t': 0.1,
              'ref-t': temperature, 'pcoupl': 'no', 'gen-vel': 'yes', 'gen-temp': temperature,
              'gen-seed': seed, 'continuation': 'no', 'nstxout-compressed': 0})
    return [('em_sc', em, f'soft-core minimization, all non-bonded interactions at lambda {lambda_start}'),
            ('relax_sc', md, f'soft-core relaxation: {relax_ps:g} ps at {temperature:g} K, lambda '
                             f'{lambda_start} -> 0 (full interactions)')]


def eq21_steps(*, scale=1.0, seed=42, temperature_K=EQ21_TEMPERATURE_K):
    """[(name, params, comment)] for the 21 stages at temperature_K (hot stages at max(600, T + 100) K); `scale`
    shortens every stage (tests only)."""
    cfg = RadonPy21Config(temperature_K=float(temperature_K), high_temperature_K=max(600.0, float(temperature_K) + 100.0),
                          pressure_atm=1.0)
    out = []
    for st in stages(cfg):
        ps = st.steps * cfg.timestep_fs / 1000.0 * scale
        p = dict(COMMON, integrator='md', dt=DT_EQ21_PS, nsteps=max(10, int(round(ps / DT_EQ21_PS))), **EQ21_COUPLING)
        if st.number < 21:
            p.update(EQ21_SHORT_CUTOFF)
        p['ref-t'] = st.temperature_K
        if st.ensemble == 'npt':
            p['pcoupl'] = 'C-rescale'
            p['ref-p'] = round(st.pressure_atm * BAR_PER_ATM, 4)
        else:
            p['pcoupl'] = 'no'
            for k in ('pcoupltype', 'tau-p', 'compressibility'):
                p.pop(k)
        if st.number == 1:
            p.update({'continuation': 'no', 'gen-vel': 'yes', 'gen-temp': cfg.high_temperature_K, 'gen-seed': seed})
        else:
            p.update({'continuation': 'yes', 'gen-vel': 'no'})
        p['nstxout-compressed'] = 0
        out.append((f'stage_{st.number:02d}', p,
                    f'EQ21 block {st.block} stage {st.number}: {st.ensemble.upper()} {st.temperature_K:g} K'
                    + (f', {st.pressure_atm:g} atm' if st.pressure_atm else '') + f', {ps:g} ps'))
    return out


def branch_steps(T, *, pressure_atm=1.0, pre_ns=1.0, eq_ns=10.0, prod_ns=10.0, frame_ps=10.0, energy_ps=1.0, seed=None):
    """pre: v-rescale + C-rescale at T from the EQ21 end state (fresh velocities); then the user's
    Nose-Hoover + Parrinello-Rahman equilibration and production. Started directly from the 500 K
    state at 400 K (-1000 to -1700 bar), Parrinello-Rahman's box inertia overshot and the box ran
    away (PS 4.6 -> 119 nm in 100 ps), so it starts only after the density has relaxed at T."""
    seed = int(round(10 * T)) if seed is None else seed
    base = dict(COMMON, integrator='md', dt=DT_PS, **BRANCH_COUPLING)
    p_bar = round(float(pressure_atm) * BAR_PER_ATM, 6)
    base['ref-t'] = T
    base['ref-p'] = p_bar
    base['nstenergy'] = BRANCH_NSTENERGY
    pre = dict(COMMON, integrator='md', dt=DT_PS, nsteps=int(round(pre_ns * 1000 / DT_PS)), **EQ21_COUPLING)
    pre.update({'ref-t': T, 'pcoupl': 'C-rescale', 'ref-p': p_bar, 'continuation': 'no', 'gen-vel': 'yes',
                'gen-temp': T, 'gen-seed': seed, 'nstxout-compressed': 0,
                'nstenergy': int(round(energy_ps / DT_PS))})
    eq = dict(base, nsteps=int(round(eq_ns * 1000 / DT_PS)), continuation='yes', **{'gen-vel': 'no'},
              **{'nstxout-compressed': 0})
    prod = dict(base, nsteps=int(round(prod_ns * 1000 / DT_PS)), continuation='yes', **{'gen-vel': 'no'},
                **{'nstxout-compressed': int(round(frame_ps / DT_PS))})
    where = f'{T:g} K, {float(pressure_atm):g} atm ({p_bar:g} bar)'
    return [('pre', pre, f'{where} branch: {pre_ns:g} ns NPT pre-equilibration (v-rescale + C-rescale), '
                         f'fresh velocities seed {seed}'),
            ('eq', eq, f'{where} branch: {eq_ns:g} ns NPT equilibration (Nose-Hoover + Parrinello-Rahman)'),
            ('prod', prod, f'{where} branch: {prod_ns:g} ns NPT production, frames every {frame_ps:g} ps')]


def direct_steps(*, temperature_K, pressure_atm=1.0, nvt_ps=100.0, compress_ps=500.0, compress_atm=1000.0,
                 release_ps=500.0, seed=11):
    """Direct equilibration of a molecular liquid after the soft-core relaxation and EM: NVT at T, C-rescale NPT
    compression at compress_atm (the dilute AutoPoly box reaches liquid density in ~100 ps), C-rescale NPT at the
    state pressure. The dilute stages use the EQ21 pair-list settings (uniform-density buffer estimates fail)."""
    T = float(temperature_K)
    base = dict(COMMON, integrator='md', dt=DT_PS, **EQ21_COUPLING, **EQ21_PAIRLIST)
    base['ref-t'] = T
    nvt = dict(base, nsteps=int(round(nvt_ps / DT_PS)), pcoupl='no', continuation='no', **{'gen-vel': 'yes',
               'gen-temp': T, 'gen-seed': seed, 'nstxout-compressed': 0})
    for k in ('pcoupltype', 'tau-p', 'compressibility'):
        nvt.pop(k)
    comp = dict(base, nsteps=int(round(compress_ps / DT_PS)), pcoupl='C-rescale',
                **{'ref-p': round(compress_atm * BAR_PER_ATM, 4), 'continuation': 'yes', 'gen-vel': 'no',
                   'nstxout-compressed': 0})
    rel = dict(comp, nsteps=int(round(release_ps / DT_PS)), **{'ref-p': round(float(pressure_atm) * BAR_PER_ATM, 6)})
    return [('nvt', nvt, f'direct equilibration: {nvt_ps:g} ps NVT at {T:g} K (v-rescale), fresh velocities'),
            ('compress', comp, f'direct equilibration: {compress_ps:g} ps C-rescale NPT at {T:g} K, {compress_atm:g} atm '
                               '(dilute box -> liquid density)'),
            ('release', rel, f'direct equilibration: {release_ps:g} ps C-rescale NPT at {T:g} K, '
                             f'{float(pressure_atm):g} atm')]


def write_steps(out_dir, steps, *, topology, start_conf, start_cpt=None, step_topology=None) -> dict:
    """mdp files + steps.json; step i starts from step i-1 (.gro and, if continuing, .cpt).
    step_topology: {step name: topology} overrides (soft-core steps use the B-state topology)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries, conf, cpt = [], str(start_conf), start_cpt
    for name, params, comment in steps:
        (out/f'{name}.mdp').write_text(mdp_text(params, comment))
        cont = params.get('continuation') == 'yes'
        entries.append({'name': name, 'mdp': f'{name}.mdp', 'conf': conf, 'cpt': cpt if cont else None,
                        'nsteps': params['nsteps'], 'integrator': params['integrator'],
                        'topology': str((step_topology or {}).get(name, topology))})
        conf, cpt = str(out/f'{name}.gro'), str(out/f'{name}.cpt')
    manifest = {'topology': str(topology), 'steps': entries}
    (out/'steps.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def write_protocol(gmx_system_dir, root, *, eq21_scale=1.0, branch_kwargs=None, temperatures=BRANCH_TEMPERATURES_K,
                   eq21_temperature=None, equilibration='eq21') -> dict:
    """root/eq21 (em + 21 stages at eq21_temperature, default the hottest approved state, ending at 1 atm) and one
    branch root/T<label> (pre, eq, prod at the state's T and P) per approved state point, for one converted system."""
    from ..ibi.states import parse
    temps = parse(temperatures)
    eq21_T = float(eq21_temperature) if eq21_temperature is not None else float(max(s.T for s in temps))
    sysd, root = Path(gmx_system_dir).resolve(), Path(root).resolve()
    top = sysd/'system.top'
    sc_top = sysd/'system_sc.top'
    if not sc_top.is_file():
        from .lmp2gmx import add_decoupled_states
        add_decoupled_states(top, sc_top)
    em = em_params()
    em.update(EQ21_PAIRLIST)
    if equilibration == 'eq21':
        eq_dir, last = 'eq21', 'stage_21'
        steps = (soft_core_steps() + [('em', em, 'steepest-descent minimization, full potential')]
                 + eq21_steps(scale=eq21_scale, temperature_K=eq21_T))
    elif equilibration == 'direct':
        eq_dir, last = 'equil', 'release'
        hot = max(temps, key=lambda s: (s.T, -float(s.P)))
        steps = (soft_core_steps() + [('em', em, 'steepest-descent minimization, full potential')]
                 + direct_steps(temperature_K=eq21_T, pressure_atm=float(hot.P)))
    else:
        raise ValueError(f'unknown equilibration {equilibration!r} (eq21 | direct)')
    write_steps(root/eq_dir, steps, topology=top, start_conf=sysd/'system.gro',
                step_topology={'em_sc': sc_top, 'relax_sc': sc_top})
    for s in temps:
        write_steps(root/f'T{s.label}', branch_steps(float(s.T), pressure_atm=float(s.P), **(branch_kwargs or {})),
                    topology=top, start_conf=root/eq_dir/f'{last}.gro')
    summary = {'equilibration': equilibration, 'equilibration_dir': eq_dir,
               'eq21_temperature_K' if equilibration == 'eq21' else 'equilibration_temperature_K': eq21_T,
               'branches_K': [float(s.T) for s in temps],
               'branches': [s.to_dict() for s in temps], 'branch_dirs': [f'T{s.label}' for s in temps],
               'eq21_scale': eq21_scale, 'branch': branch_kwargs or {}}
    (root/'protocol.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def step_done(d: Path, name: str) -> bool:
    log = d/f'{name}.log'
    return (d/f'{name}.gro').is_file() and log.is_file() and 'Finished mdrun' in log.read_text(errors='replace')


def status(d) -> dict:
    d = Path(d)
    m = json.loads((d/'steps.json').read_text())
    done = [step_done(d, s['name']) for s in m['steps']]
    return {'steps': len(done), 'done': sum(done), 'next': next((s['name'] for s, x in zip(m['steps'], done) if not x), None)}


def run_steps(d, *, gmx='gmx_mpi', mdrun='mpirun -np 1 gmx_mpi mdrun', mdrun_flags='', gpu_flags='',
              maxh=None) -> dict:
    """grompp + mdrun for every unfinished step in order; continues from STEP.cpt if present.
    `mdrun_flags` apply to every step (threads), `gpu_flags` only to MD steps (EM runs on the CPU)."""
    d = Path(d).resolve()
    m = json.loads((d/'steps.json').read_text())
    for s in m['steps']:
        name = s['name']
        if step_done(d, name):
            continue
        tpr = d/f'{name}.tpr'
        if not tpr.is_file():
            cmd = [*shlex.split(gmx), 'grompp', '-f', s['mdp'], '-c', s['conf'], '-p', s.get('topology', m['topology']),
                   '-o', tpr.name, '-po', f'{name}.mdout.mdp', '-maxwarn', '2']
            if s['cpt']:
                cmd += ['-t', s['cpt']]
            with open(d/f'{name}.grompp.log', 'w') as f:
                r = subprocess.run(cmd, cwd=d, stdout=f, stderr=subprocess.STDOUT)
            if r.returncode:
                raise RuntimeError(f'grompp failed for {d}/{name} (see {name}.grompp.log)')
        cmd = [*shlex.split(mdrun), '-deffnm', name, *shlex.split(mdrun_flags)]
        if s['integrator'] != 'steep' and not s['name'].endswith('_sc'):     # free-energy steps: CPU
            cmd += shlex.split(gpu_flags)
            if (d/f'{name}.cpt').is_file():
                cmd += ['-cpi', f'{name}.cpt']
            if maxh:
                cmd += ['-maxh', str(maxh)]
        with open(d/f'{name}.mdrun.out', 'a') as f:
            r = subprocess.run(cmd, cwd=d, stdout=f, stderr=subprocess.STDOUT)
        if not step_done(d, name):
            raise RuntimeError(f'mdrun did not finish {d}/{name} (exit {r.returncode}; see {name}.log)')
        if name == 'prod':
            sel = 'Temperature\nPressure\nVolume\nDensity\n\n'
            with open(d/'prod.energy.log', 'w') as f:
                subprocess.run([*shlex.split(gmx), 'energy', '-f', 'prod.edr', '-o', 'prod_energy.xvg'],
                               cwd=d, input=sel, text=True, stdout=f, stderr=subprocess.STDOUT)
    return status(d)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Run the unfinished GROMACS steps of one protocol directory')
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--gmx', default='gmx_mpi')
    ap.add_argument('--mdrun', default='mpirun -np 1 gmx_mpi mdrun')
    ap.add_argument('--mdrun-flags', default='')
    ap.add_argument('--gpu-flags', default='')
    ap.add_argument('--maxh', type=float, default=None)
    a = ap.parse_args(argv)
    for d in a.dirs:
        print(d, json.dumps(run_steps(d, gmx=a.gmx, mdrun=a.mdrun, mdrun_flags=a.mdrun_flags,
                                      gpu_flags=a.gpu_flags, maxh=a.maxh)), flush=True)


if __name__ == '__main__':
    main()
