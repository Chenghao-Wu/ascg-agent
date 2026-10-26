"""Host side of ascg: `python -m ascg.remote <verb>` with a JSON payload on stdin; prints one JSON line
{"ok": true, "result": ...} or {"ok": false, "error": "..."}. Runs in the target's environment and calls the numerical
core (ascg_core.ibi: pipeline, runner, analysis, tables, selfcheck) unchanged.

Job verbs (`job`) run inside a batch job / background process and write workflow/ascg_job.json with their status:
running -> done (validated / needs expert input / failed under budget) | awaiting_decision (missed the convergence
check: an agent chooses a bounded action) | stopped_on_gate (scientific stop, e.g. broken bonds) | error.
"""
from __future__ import annotations

import base64
import glob
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from ascg_core.ibi import states as _states   # light: json / pathlib only

_pretty = lambda T: _states.parse_one(T).pretty()   # '400 K' / '400 K, 1000 atm'
VERBS = {}
MARK = 'ASCG_TASK'          # environment marker of processes launched by ascg: "<study>/<task>"
NOISE_REL_SE = 0.05         # targets review: block SE above 5 % of the value in the peak = statistically noisy


def verb(fn):
    VERBS[fn.__name__] = fn
    return fn


def _now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


# ------------------------------------------------------------------------------------------------ host / engine
@verb
def ping(p):
    root = Path.cwd()
    try:
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'], capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = ''
    from . import __version__
    return {'hostname': socket.gethostname(), 'python': sys.version.split()[0], 'root': str(root), 'commit': commit,
            'ascg': __version__, 'fastmd_binary': os.environ.get('FASTMD_BINARY'),
            'fastmd_present': bool(os.environ.get('FASTMD_BINARY') and Path(os.environ['FASTMD_BINARY']).exists()),
            'scheduler': {'slurm': bool(_which('sbatch')), 'pbs': bool(_which('qsub'))}}


def _which(x):
    from shutil import which
    return which(x)


def _proc_marker(pid):
    try:
        env = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
    except OSError:
        return None
    for kv in env:
        if kv.startswith(MARK.encode() + b'='):
            return kv.split(b'=', 1)[1].decode()
    return None


@verb
def gpus(p):
    """GPUs with this study's (marked) processes and foreign processes (anything else, including the same user's
    other jobs)."""
    if not _which('nvidia-smi'):
        return []
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,pci.bus_id,name,utilization.gpu,memory.used',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    out = {}
    for line in q.strip().splitlines():
        i, bus, name, util, mem = [x.strip() for x in line.split(',')]
        out[bus.upper()] = {'index': int(i), 'name': name, 'util_pct': float(util), 'mem_mib': float(mem), 'own': [],
                            'foreign': []}
    a = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_bus_id,pid,process_name', '--format=csv,noheader'],
                       capture_output=True, text=True).stdout
    for line in a.strip().splitlines():
        if not line.strip():
            continue
        bus, pid, name = [x.strip() for x in line.split(',', 2)]
        g = out.get(bus.upper())
        if g is None:
            continue
        mark = _proc_marker(pid)
        (g['own'] if mark else g['foreign']).append({'pid': int(pid), 'name': os.path.basename(name), 'task': mark})
    return sorted(out.values(), key=lambda g: g['index'])


@verb
def slurm_jobs(p):
    if not _which('squeue'):
        return []
    r = subprocess.run(['squeue', '--me', '-h', '-o', '%i|%j|%T|%r|%M|%N|%S|%q|%b'], capture_output=True,
                       text=True).stdout
    keys = ('id', 'name', 'state', 'reason', 'time', 'node', 'expected_start', 'qos', 'gres')
    return [_with_wait(dict(zip(keys, l.split('|')))) for l in r.strip().splitlines() if l]


@verb
def qos_limits(p):
    """Per-user GPU cap of each named QoS (sacctmgr MaxTRESPerUser gres/gpu), None when unlimited or unknown."""
    import re
    out = {q: None for q in p.get('qos', [])}
    if not out or not _which('sacctmgr'):
        return out
    r = subprocess.run(['sacctmgr', '-n', '-P', 'show', 'qos', ','.join(out), 'format=Name,MaxTRESPerUser'],
                       capture_output=True, text=True).stdout
    for line in r.strip().splitlines():
        name, _, tres = line.partition('|')
        m = re.search(r'gres/gpu=(\d+)', tres)
        if name in out and m:
            out[name] = int(m.group(1))
    return out


def _with_wait(j):
    """Hours until the scheduler's expected start of a pending job (None if unknown)."""
    from datetime import datetime
    try:
        start = datetime.fromisoformat(j.get('expected_start', ''))
        j['expected_wait_h'] = round(max((start - datetime.now()).total_seconds(), 0) / 3600, 1) \
            if j.get('state') == 'PENDING' else None
    except ValueError:
        j['expected_wait_h'] = None
    return j


def _queue(job_id):
    r = subprocess.run(['squeue', '-h', '-j', str(job_id), '-o', '%i|%j|%T|%r|%M|%N|%S'], capture_output=True, text=True)
    line = r.stdout.strip()
    if not line:
        return None
    j = _with_wait(dict(zip(('id', 'name', 'state', 'reason', 'time', 'node', 'expected_start'), line.split('|'))))
    return {k: j[k] for k in ('state', 'reason', 'expected_start', 'expected_wait_h')}


@verb
def selfcheck(p):
    import tempfile
    from ascg_core.ibi.selfcheck import fastmd_cbt_dihedral_check, fastmd_exclusion_energy_check
    env = {'CUDA_VISIBLE_DEVICES': str(p['gpu'])} if p.get('gpu') is not None else None
    with tempfile.TemporaryDirectory() as d:
        ex = fastmd_exclusion_energy_check(d, env=env)
    res = {'exclusions': ex}
    if p.get('dihedrals', True):
        with tempfile.TemporaryDirectory() as d:
            res['cbt_dihedrals'] = fastmd_cbt_dihedral_check(d, env=env)
    from ascg_core.ibi.selfcheck import fastmd_npt_stability_check
    with tempfile.TemporaryDirectory() as d:
        res['npt_stability'] = fastmd_npt_stability_check(d, env=env)
    res['ok'] = all(v.get('ok') for v in res.values() if isinstance(v, dict))
    return res


# ------------------------------------------------------------------------------------------------ references
@verb
def preflight(p):
    from ascg_core.ibi.pipeline import preflight as pf
    return pf(p['ref'], p.get('temperatures'))


@verb
def ref_info(p):
    ref = Path(p['ref'])
    topo = json.loads((ref/'topology.json').read_text())
    summ = json.loads((ref/'summary.json').read_text())
    mol = topo['bead_molecule']
    n_chain = len(set(mol))
    return {'task': summ.get('task', ref.name), 'n_beads': summ.get('n_beads', len(mol)), 'n_molecules': n_chain,
            'beads_per_molecule': round(len(mol) / max(n_chain, 1), 2), 'type_names': summ['type_names'],
            'bond_types': summ['bond_types'], 'angle_types': summ['angle_types'],
            'total_mass_amu': float(sum(topo['bead_masses_amu'])),
            'states': {T: {k: v for k, v in s.items() if not isinstance(v, list)} for T, s in summ['states'].items()},
            'reference_states': _states.labels(_states.of_reference(ref)),
            'reference_state_points': [s.to_dict() for s in _states.of_reference(ref)],
            'missing_files': {_states.label(T): _states.missing_files(ref, [T]) for T in _states.of_reference(ref)
                              if _states.missing_files(ref, [T])}}


def _run_temps(td=None, ref=None):
    """States of a task's IBI run (its state.json), else of the reference."""
    if td is not None:
        t = _states.of_task(td)
        if t:
            return t
    if ref is None:
        raise RuntimeError('no run state and no reference: the state points are unknown')
    return _states.of_reference(ref)


def _code_digest():
    """md5 of every source file the job runs (ascg and the numerical core), for provenance."""
    import ascg
    import ascg_core
    files = {}
    for pkg in (ascg, ascg_core):
        root = Path(pkg.__file__).parent
        for f in sorted(root.rglob('*.py')):
            files[f'{pkg.__name__}/{f.relative_to(root)}'] = hashlib.md5(f.read_bytes()).hexdigest()
    return {'digest': hashlib.md5(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16], 'files': files}


# ------------------------------------------------------------------------------------------------ deploy / doctor
DEPS = ('numpy', 'scipy', 'matplotlib', 'rdkit', 'networkx', 'MDAnalysis', 'yaml')
PINNED = ('numpy', 'scipy')            # never changed by `deps --install` (fitting numerics stay as deployed)


@verb
def manifest(p):
    """md5 of the given files under the checkout (None: missing)."""
    out = {}
    for rel in p['files']:
        f = Path(rel)
        out[rel] = hashlib.md5(f.read_bytes()).hexdigest() if f.is_file() else None
    return out


@verb
def busy(p):
    """ascg work running here: marked processes (any study) and this user's Slurm jobs named ascg-*."""
    procs = []
    for d in Path('/proc').glob('[0-9]*'):
        m = _proc_marker(d.name)
        if m:
            procs.append({'pid': int(d.name), 'task': m})
    jobs = [j for j in slurm_jobs({}) if str(j.get('name', '')).startswith('ascg-')]
    return {'processes': procs, 'slurm_jobs': jobs}


@verb
def deps(p):
    """Versions of the Python dependencies (None: missing); with install: pip-install the missing ones, numpy and scipy
    pinned at their installed versions."""
    import importlib
    def versions():
        out = {}
        for m in DEPS:
            try:
                out[m] = getattr(importlib.import_module(m), '__version__', 'present')
            except Exception:
                out[m] = None
        return out
    have = versions()
    res = {'python': sys.version.split()[0], 'executable': sys.executable, 'versions': have}
    missing = [m for m, v in have.items() if v is None]
    if p.get('install') and missing:
        import tempfile
        names = {'yaml': 'PyYAML'}
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as c:
            for m in PINNED:
                if have.get(m):
                    c.write(f'{m}=={have[m]}\n')
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-c', c.name] + [names.get(m, m) for m in missing],
                           capture_output=True, text=True)
        os.unlink(c.name)
        importlib.invalidate_caches()
        res['install'] = {'requested': missing, 'returncode': r.returncode, 'tail': (r.stdout + r.stderr)[-600:]}
        res['versions'] = versions()
        for m in PINNED:
            if have.get(m) and res['versions'].get(m) != have[m]:
                res['install']['error'] = f'{m} changed from {have[m]} to {res["versions"][m]}'
    return res


@verb
def deploy_record(p):
    """Record a deploy in the checkout (.ascg_deploy.json) and return the code digest of what is now here."""
    code = _code_digest()
    rec = {'time': _now(), 'source_commit': p.get('commit'), 'source_dirty': p.get('dirty'), 'files': p.get('n_files'),
           'code_digest': code['digest'], 'host': socket.gethostname()}
    Path('.ascg_deploy.json').write_text(json.dumps(rec, indent=1) + '\n')
    return rec


@verb
def code_digest(p):
    """Code digest of the ascg and ascg_core sources here, and the last deploy record."""
    f = Path('.ascg_deploy.json')
    return {'digest': _code_digest()['digest'], 'deploy': json.loads(f.read_text()) if f.exists() else None}


@verb
def fastmd_info(p):
    """The fastMD build the environment points at: binary, md5, source commit (-dirty with local changes)."""
    b = os.environ.get('FASTMD_BINARY')
    if not b or not Path(b).exists():
        return {'binary': b, 'present': False}
    from ascg_core.ibi.runner import fastmd_provenance
    out = {**fastmd_provenance(b), 'present': True}
    if p.get('ref'):                                    # is the source at or after this commit?
        src = Path(b).resolve().parent.parent
        r = subprocess.run(['git', '-C', str(src), 'merge-base', '--is-ancestor', p['ref'], 'HEAD'],
                           capture_output=True, text=True)
        out['includes_ref'] = {0: True, 1: False}.get(r.returncode)       # None: not a git tree / unknown commit
    return out


@verb
def ref_inspect(p):
    """Bring-your-own AA data: what the topology / trajectory / energy file hold; with a mapping, the beads it gives."""
    from ascg_core.mapping.byo import inspect
    return inspect(p['topology'], p.get('trajectory'), p.get('energy'), p.get('mapping'),
                   topology_format=p.get('topology_format'), trajectory_format=p.get('trajectory_format'),
                   timestep_fs=p.get('timestep_fs'))


@verb
def ref_build(p):
    """Bring-your-own AA data -> mapped reference at `out` (map-aa's files) + the mapping used + reference QA."""
    from ascg_core.mapping.byo import prepare_byo_reference
    out = Path(p['out'])
    if (out/'summary.json').exists() and not p.get('overwrite'):
        raise RuntimeError(f'{out} already holds a reference (summary.json); choose another --out')
    runs = {T: {k: r.get(k) for k in ('trajectory', 'energy', 'timestep_fs')} for T, r in p['runs'].items()}
    summary = prepare_byo_reference(p['task'], p['topology'], runs, p['mapping'], out,
                                    topology_format=p.get('topology_format'),
                                    trajectory_format=p.get('trajectory_format'),
                                    start_ps=float(p.get('start_ps') or 0.0), stride=int(p.get('stride') or 1),
                                    **{k: float(p[k]) for k in ('r_max_nm', 'bond_bin_nm', 'angle_bin_deg') if p.get(k)})
    (out/'mapping.json').write_text(json.dumps(p['mapping'], indent=1) + '\n')
    return {'ref': str(out), 'summary': summary, 'qa': aa_qa({'ref': str(out)})}


@verb
def aa_qa(p):
    """Checks before an AA reference is used: equilibration (density drift vs its error), sampling of chain size
    (Rg by quarters of the production), analysis range vs box, state dependence of chain size."""
    import numpy as np
    from ascg_core.mapping.cg_mapping import chain_size
    ref = Path(p['ref'])
    summ = json.loads((ref/'summary.json').read_text())
    topo = json.loads((ref/'topology.json').read_text())
    masses, mol = np.asarray(topo['bead_masses_amu'], float), np.asarray(topo['bead_molecule'])
    flags, per_T = [], {}
    for T, s in sorted(summ['states'].items()):
        d = {k: s.get(k) for k in ('density_g_cm3', 'density_se', 'density_drift_g_cm3', 'rg_nm', 'ree_rms_nm',
                                   'temperature_K_mean', 'r_max_over_half_box', 'frames')}
        f = ref/f'mapped_T{T}.npz'
        if f.exists() and mol.max() > 0:
            pos = np.load(f)['pos_nm']
            idx = np.linspace(0, len(pos) - 1, min(len(pos), 200)).astype(int)
            rg = np.array([np.sqrt(np.mean(chain_size(pos[i].astype(float), masses, mol)[0]**2)) for i in idx])
            q = np.array_split(rg, 4)
            d['rg_quarters_nm'] = [round(float(x.mean()), 4) for x in q]
            drift = (d['rg_quarters_nm'][-1] - d['rg_quarters_nm'][0]) / max(d['rg_quarters_nm'][0], 1e-9)
            d['rg_drift_rel'] = round(float(drift), 4)
            if abs(drift) > 0.03:
                flags.append({'severity': 'warn', 'state': T, 'check': 'chain relaxation',
                              'message': f'Rg drifts {100 * drift:+.1f} % from the first to the last quarter of the '
                                         'production: chain conformations may not be equilibrated'})
        if s.get('density_drift_g_cm3') is not None and s.get('density_se'):
            if abs(s['density_drift_g_cm3']) > 5 * max(s['density_se'], 1e-6) and abs(s['density_drift_g_cm3']) > 0.002:
                flags.append({'severity': 'warn', 'state': T, 'check': 'density drift',
                              'message': f'density drift {s["density_drift_g_cm3"]:+.4f} g/cm3 exceeds 5 standard errors'})
        if s.get('r_max_over_half_box') and s['r_max_over_half_box'] > 1.0:
            flags.append({'severity': 'error', 'state': T, 'check': 'analysis range',
                          'message': 'RDF range exceeds half the box'})
        sp = _states.parse_one(T)                       # the AA run's mean pressure against the state's pressure
        d['pressure_bar_target'] = round(sp.pressure_bar, 3)
        if s.get('temperature_K_mean') is not None and abs(s['temperature_K_mean'] - sp.T) > max(2.0, 0.01 * sp.T):
            flags.append({'severity': 'warn', 'state': T, 'check': 'temperature',
                          'message': f'AA mean temperature {s["temperature_K_mean"]:.1f} K is not the state temperature '
                                     f'{sp.T:g} K: the reference may not be at this state (label or thermostat)'})
        if s.get('density_g_cm3') is not None and not 0.3 <= s['density_g_cm3'] <= 3.0:
            flags.append({'severity': 'warn', 'state': T, 'check': 'density',
                          'message': f'AA density {s["density_g_cm3"]:.3g} g/cm3 is outside condensed phases (0.3-3): '
                                     'wrong units, wrong files, or not a liquid / melt'})
        if s.get('thermo', 'present') is None:
            flags.append({'severity': 'info', 'state': T, 'check': 'thermo',
                          'message': 'no energy file: the AA temperature and pressure were not checked'})
        if s.get('pressure_bar_mean') is not None:
            dev, se = s['pressure_bar_mean'] - sp.pressure_bar, max(s.get('pressure_bar_se') or 0.0, 1e-9)
            d['pressure_bar_mean'] = round(s['pressure_bar_mean'], 2)
            if abs(dev) > max(3 * se, 50.0 + 0.02 * sp.pressure_bar):
                flags.append({'severity': 'warn', 'state': T, 'check': 'pressure',
                              'message': f'AA mean pressure {s["pressure_bar_mean"]:+.0f} bar (SE {se:.0f}) is not the '
                                         f'state pressure {sp.pressure_bar:.0f} bar: the reference may not be at this '
                                         'state (label or barostat)'})
        per_T[T] = d
    for P in sorted({_states.parse_one(T).P for T in per_T}):    # chain size across temperatures at one pressure
        Ts = _states.labels(x for x in _states.parse(list(per_T)) if x.P == P)
        if len(Ts) >= 2 and all(per_T[T].get('rg_nm') for T in Ts):
            lo, hi = per_T[Ts[0]]['rg_nm'], per_T[Ts[-1]]['rg_nm']
            rel = (lo - hi) / hi
            if abs(rel) > 0.10:
                flags.append({'severity': 'warn', 'state': 'all', 'check': 'state dependence of chain size',
                              'message': f'Rg changes {100 * rel:+.0f} % between {_pretty(Ts[-1])} and {_pretty(Ts[0])}; '
                                         'most melts change by a few %. Check whether the lower state is below Tm or Tg, '
                                         'or not equilibrated.'})
    return {'states': per_T, 'flags': flags, 'ok': not any(f['severity'] == 'error' for f in flags)}


@verb
def aa_audit(p):
    """OPLS typing / charge audit of a built AA system directory (AutoPoly / moltemplate): system.data with
    system.in.charges, and system.neutral.data when present (ascg_core.aa.opls_typing_audit)."""
    from ascg_core.aa.opls_typing_audit import audit
    d = Path(p['system_dir'])
    ch = d/'system.in.charges'
    out = {'system_dir': str(d)}
    for name in ('system.data', 'system.neutral.data'):
        if (d/name).exists():
            r = audit(d/name, ch if ch.exists() else None, float(p.get('tolerance', 1e-4)))
            out[name] = {'verdict': r['verdict'], 'atoms': r['atoms'], 'molecules': r['molecules'],
                         'templates': [{k: t[k] for k in ('molecules', 'atoms', 'net_charge_e')} for t in r['templates']],
                         'findings': r['findings'],
                         'type_tables': r['templates'] if p.get('tables') else None}
    if len(out) == 1:
        raise RuntimeError(f'no system.data in {d}')
    return out


@verb
def map_review(p):
    """Mapping audit on the mapped reference: masses, types, connectivity, exclusions, dihedrals."""
    import numpy as np
    from ascg_core.ibi.system import load_system
    from ascg_core.ibi.dihedrals import enumerate_dihedrals
    ref = Path(p['ref'])
    s = load_system(ref)
    topo = json.loads((ref/'topology.json').read_text())
    issues = []
    if np.any(s.masses <= 0):
        issues.append('non-positive bead masses')
    nb = np.bincount(s.mol)
    excl = {tuple(sorted(map(int, e))) for e in topo['exclusions']}
    bonded = {tuple(sorted(map(int, b))) for b in s.bonds}
    one3 = {tuple(sorted((int(a), int(c)))) for a, _, c in s.angles}
    if not bonded <= excl:
        issues.append('bonds missing from the exclusions')
    if not one3 <= excl:
        issues.append('1-3 pairs missing from the exclusions')
    quads, dt, names = enumerate_dihedrals(s)
    return {'n_beads': s.n, 'types': {n: int((s.types == i).sum()) for i, n in enumerate(s.type_names)},
            'type_masses_amu': {n: round(float(m), 3) for n, m in zip(s.type_names, s.type_masses)},
            'molecules': int(len(nb)), 'beads_per_molecule': sorted(set(int(x) for x in nb)),
            'bond_types': s.bond_types, 'angle_types': s.angle_types, 'n_bonds': int(len(s.bonds)),
            'n_angles': int(len(s.angles)), 'n_excluded_13': int(len(s.excl13)), 'n_dihedrals': int(len(quads)),
            'dihedral_types': names, 'issues': issues, 'ok': not issues,
            'reference_task': json.loads((ref/'summary.json').read_text()).get('task')
            if (ref/'summary.json').exists() else None}


@verb
def targets_review(p):
    """Target distributions: statistical noise from the block standard errors (median SE / value over the bins above
    20 % of the peak) and resolution (bins across the half-maximum width) of bond / angle histograms; RDF noise."""
    import numpy as np
    ref = Path(p['ref'])
    summ = json.loads((ref/'summary.json').read_text())
    out, warnings, notes = {}, [], []
    for T in _states.labels(_states.parse(list(summ['states']))):
        t = np.load(ref/f'targets_T{T}.npz')
        r = {}
        for k in t.files:
            if '/' not in k or k.startswith(('bond_edges', 'rdf_se', 'bond_se', 'angle_se')):
                continue
            fam, name = k.split('/', 1)
            y = np.asarray(t[k], float)
            if y.ndim != 1 or len(y) < 5 or fam not in ('rdf', 'bond', 'angle'):
                continue
            sek = f'{fam}_se/{name}'
            core = y >= 0.2 * y.max()
            rel = float(np.median(np.asarray(t[sek], float)[core] / y[core])) if sek in t.files and core.any() else None
            r[k] = {'bins': int(len(y)), 'rel_se': None if rel is None else round(rel, 4)}
            if fam in ('bond', 'angle'):
                r[k]['bins_fwhm'] = fwhm = int((y >= 0.5 * y.max()).sum())
                if fwhm < 5:
                    notes.append(f'{_pretty(T)} {k}: the peak spans {fwhm} bins (smooth, under-resolved); finer bins at '
                                 'mapping (map-aa --bond-bin-nm / --angle-bin-deg) would resolve its shape. The fit can '
                                 'still match the histogram at this resolution')
            if rel is not None and rel > NOISE_REL_SE:
                warnings.append(f'{_pretty(T)} {k}: statistically noisy (block SE {100 * rel:.0f} % of the value in the '
                                'peak); longer AA sampling or wider bins (re-mapping) would reduce it')
        out[T] = r
        st = summ['states'][T]
        if st.get('r_max_over_half_box', 0) > 1:
            warnings.append(f'{_pretty(T)}: RDF range beyond half the box')
    dih = _dihedral_review(ref, _states.parse(list(summ['states'])))
    for T, d in dih.items():
        out[T].update(d['targets'])
        warnings += d['warnings']
    return {'targets': out, 'warnings': warnings, 'notes': notes}


def _dihedral_review(ref, temps):
    """AA dihedral distributions (the runner's own target function): roughness, empty bins, CBT prefactor."""
    import numpy as np
    from ascg_core.ibi.dihedrals import aa_targets, enumerate_dihedrals
    from ascg_core.ibi.system import load_system
    quads, dtype, names = enumerate_dihedrals(load_system(ref))
    if not len(quads):
        return {}
    out = {}
    for T, d in aa_targets(ref, quads, dtype, names, list(temps)).items():
        r, w = {}, []
        for k in names:
            y = np.asarray(d[f'dihedral/{k}'], float)
            rough = float(np.sqrt(np.mean(np.diff(y, 2)**2)) / max(np.sqrt(np.mean(y**2)), 1e-12))
            empty = float(np.mean(y <= 0))
            r[f'dihedral/{k}'] = {'bins': int(len(y)), 'roughness': round(rough, 4), 'empty_bin_fraction':
                                  round(empty, 3), 'mean_cbt_prefactor': round(float(d[f'dihedral_w/{k}']), 3)}
            if rough > 0.15:
                w.append(f'{_pretty(T)} dihedral/{k}: noisy histogram (roughness {rough:.2f})')
            if empty > 0.2:
                w.append(f'{_pretty(T)} dihedral/{k}: {100 * empty:.0f} % of the bins never sampled (table extrapolated there)')
        out[str(T)] = {'targets': r, 'warnings': w}
    return out


@verb
def md5_tree(p):
    root = Path(p['path'])
    h = {}
    for f in sorted(root.rglob('*')):
        if f.is_file():
            h[str(f.relative_to(root))] = hashlib.md5(f.read_bytes()).hexdigest()
    return {'files': len(h), 'digest': hashlib.md5(json.dumps(h, sort_keys=True).encode()).hexdigest()}


# ------------------------------------------------------------------------------------------------ jobs
def _job_file(task_dir):
    return Path(task_dir)/'workflow'/'ascg_job.json'


def _set_job(task_dir, **kw):
    f = _job_file(task_dir)
    f.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(f.read_text()) if f.exists() else {}
    cur.update(kw, updated=_now())
    f.write_text(json.dumps(cur, indent=2) + '\n')


def job_script(p, td, n, spec, python=None):
    """Text of the launch script (snapshot). Slurm directives come directly after the shebang."""
    env_line = p.get('env', '')
    lines = ['#!/bin/bash']                                       # #SBATCH directives must follow the shebang
    if p.get('scheduler') == 'slurm':
        s = p.get('submit') or {}
        lines += [f'#SBATCH --job-name=ascg-{p["task"]}', f'#SBATCH --output={td}/workflow/ascg_job_{n:02d}.out',
                  '#SBATCH --nodes=1', '#SBATCH --ntasks=1', f'#SBATCH --cpus-per-task={s.get("cpus", 4)}',
                  f'#SBATCH --mem={s.get("mem", "16G")}', f'#SBATCH --time={s.get("time", "08:00:00")}',
                  f'#SBATCH --gres={s.get("gres", "gpu:1")}']
        if s.get('qos'):
            lines.append(f'#SBATCH --qos={s["qos"]}')
        if s.get('partition'):
            part = s['partition'] if isinstance(s['partition'], str) else ','.join(s['partition'])
            lines.append(f'#SBATCH --partition={part}')
        if s.get('account'):
            lines.append(f'#SBATCH --account={s["account"]}')
    lines.append('set -o pipefail')
    if env_line:
        lines.append(f'{env_line} >/dev/null 2>&1')
    lines += [f'export {MARK}={p["study"]}/{p["task"]}', f'cd {Path.cwd()}']
    if p.get('gpu') is not None:
        lines.append(f'export CUDA_VISIBLE_DEVICES={p["gpu"]}')
    lines.append(f'{(python or sys.executable)} -m ascg.remote job < {spec}')
    return '\n'.join(lines) + '\n'


@verb
def launch(p):
    """Write a job spec + launch script snapshot into the task dir and start it (slurm: sbatch; none: a new session led by the job script, which carries the ASCG_TASK marker)."""
    td = Path(p['task_dir'])
    (td/'workflow').mkdir(parents=True, exist_ok=True)
    n = len(list((td/'workflow').glob('ascg_job_*.json'))) + 1
    spec = td/'workflow'/f'ascg_job_{n:02d}.json'
    spec.write_text(json.dumps(p, indent=2) + '\n')
    script = td/'workflow'/f'ascg_launch_{n:02d}.sh'           # snapshot: never edited after launch
    script.write_text(job_script(p, td, n, spec))
    script.chmod(0o755)
    _set_job(td, status='submitted', mode=p['mode'], job_n=n, message='')
    if p.get('scheduler') == 'slurm':
        r = subprocess.run(['sbatch', '--parsable', str(script)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'sbatch failed: {r.stderr.strip()}')
        directives = [l.split(None, 1)[1] for l in script.read_text().splitlines() if l.startswith('#SBATCH ')]
        return {'id': r.stdout.strip().split(';')[0], 'script': str(script), 'job_n': n, 'sbatch': directives}
    log = td/'workflow'/f'ascg_job_{n:02d}.out'
    env = {**os.environ, MARK: f'{p["study"]}/{p["task"]}'}       # the session leader itself carries the marker
    proc = subprocess.Popen(['bash', str(script)], stdout=open(log, 'w'), stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    return {'pid': proc.pid, 'pgid': proc.pid, 'script': str(script), 'job_n': n, 'log': str(log)}


@verb
def job(p):
    """Runs inside the batch job. Modes: fit | continue | validate."""
    from ascg_core.ibi import pipeline as P
    td, ref = Path(p['task_dir']), Path(p['ref'])
    proto = p['protocol']
    log = P._Log(td)
    code = _code_digest()
    jf = _job_file(td)
    n = (json.loads(jf.read_text()).get('job_n') if jf.exists() else None) or 0
    (td/'workflow'/f'ascg_code_{int(n):02d}.json').write_text(json.dumps(code, indent=1) + '\n')
    _set_job(td, status='running', mode=p['mode'], phase='fitting' if p['mode'] != 'validate' else 'validating',
             message='', started=_now(), host=socket.gethostname(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
             code_digest=code['digest'])
    retries = proto.get('infrastructure_retries', 2)
    try:
        _check_gates(proto)
        states = p.get('states')                      # the task's approved state points (aa import --states)
        pf = P.preflight(ref, states)
        dih = proto['dihedrals'] if pf.get('n_dihedrals') else 'none'
        model = proto['model']
        sf = td/'workflow'/'state.json'
        if p['mode'] != 'fit' and sf.exists():          # continue an existing run with its own model / dihedrals
            from ascg_core.ibi.dihedrals import style_of
            prev = json.loads(sf.read_text())['settings']
            dih, model = style_of(prev.get('dihedrals')) or 'none', prev.get('model', model)
        common = ['--ref', str(ref), '--task-dir', str(td), '--model', model, '--dihedrals', dih,
                  '--npt-every', str(proto['npt_every']), '--gpus', 'inherit']
        if p['mode'] == 'fit' and pf.get('states'):
            common += ['--states', ','.join(pf['states'])]
        s1, s2 = proto['stage1'], proto['stage2']
        st2 = ['--schedule', s2['schedule'], '--pressure-correction', s2['pressure_correction']]
        if p['mode'] == 'fit':
            if not pf['ok']:
                _set_job(td, status='done', outcome='unsupported',
                         message='; '.join(f"{c['name']}: {c['detail']}" for c in pf['checks'] if not c['ok']))
                P.write_reports(ref, td, {'task': p['task'], 'class': 'unsupported', 'reasons': []})
                return {'status': 'unsupported'}
            from ascg_core.ibi.selfcheck import main as selfcheck_main
            if selfcheck_main([] if dih != 'none' else ['--no-dihedrals']) != 0:
                raise RuntimeError('fastMD selfcheck failed on this host')
            P._call(log, 'stage1', ['ibi', *common, '--iters', str(s1['iters']), '--schedule', s1['schedule'],
                                    '--pressure-correction', s1['pressure_correction']], retries)
            P._call(log, 'stage2', ['ibi', *common, '--iters', str(s2['iters']), *st2], retries)
        elif p['mode'] == 'continue':
            a = p['action']
            st = json.loads((td/'workflow'/'state.json').read_text())
            extra = []
            if a['action'] == 'restart_smaller_timestep':      # gate stop before iteration 0 completed
                ibi = td/'cg'/'ibi'
                dts = [json.loads(f.read_text())['dt_ps'] for f in sorted(ibi.glob('iter_000/*_T*/run.json'))]
                if not dts:
                    raise RuntimeError('no run of the failed iteration 0 records its timestep')
                stamp = time.strftime('%Y%m%d_%H%M%S')
                ibi.rename(td/'cg'/f'ibi_restart_{stamp}')                 # the failed attempt is kept, not deleted
                (td/'workflow'/'state.json').rename(td/'workflow'/f'state_restart_{stamp}.json')
                log.event('restart', archived=f'cg/ibi_restart_{stamp}', dt_ps=min(dts),
                          dt_factor=a['dt_factor'], reason=a.get('rationale', '')[:300])
                extra += ['--dt', f'{min(dts) * float(a["dt_factor"]):.6f}']
            elif a['action'].startswith('rewind'):
                P._call(log, 'rewind', ['rewind', '--task-dir', str(td), '--to-iteration', str(a['to_iteration']),
                                        '--reason', a.get('rationale', '')[:300]], 0)
            if a.get('dt_factor') and a['action'] != 'restart_smaller_timestep':
                k = a.get('to_iteration', st['completed'])
                dt = json.loads((td/'cg'/'ibi'/f'iter_{int(k):03d}'/'metrics.json').read_text())['dt_ps']
                extra += ['--dt', f'{dt * float(a["dt_factor"]):.6f}']
            if a.get('lam'):
                extra += ['--lam', str(a['lam'])]
            from .protocols import continuation_plan
            at = (-1 if a['action'] == 'restart_smaller_timestep' else
                  int(a['to_iteration']) if a['action'].startswith('rewind') else int(st['completed']))
            for step in continuation_plan(proto, a, at):     # back inside stage 1: stage 1 is finished first
                P._call(log, step['stage'], ['ibi', *common, '--iters', str(step['to']), '--schedule', step['schedule'],
                                             '--pressure-correction', step['pressure_correction'], *extra], retries)
        if p['mode'] in ('fit', 'continue'):
            ok, why = P.converged(td, proto['extension']['density_trigger'], proto['gates']['rdf_nrmse_max'])
            log.event('convergence check', ok=ok, reason=why)
            if ok:
                _set_job(td, phase='validating', message=f'converged ({why}); held-out validation running')
            if not ok:
                _set_job(td, status='awaiting_decision', message=why, finished=_now())
                return {'status': 'awaiting_decision', 'reason': why}
        v = proto['validation']
        try:
            P._call(log, 'validate', ['validate', '--ref', str(ref), '--task-dir', str(td), '--seeds', str(v['seeds']),
                                      '--eq-ps', str(v['eq_ps']), '--prod-ps', str(v['prod_ps']), '--gpus', 'inherit'],
                    retries)
        except P.ScientificFailure:
            pass                                    # recorded in validation.json; classify() reports it
        vf = td/'cg'/'ibi'/'final'/'validation.json'
        cls, reasons = P.classify(json.loads(vf.read_text()) if vf.exists() else None)
        state = json.loads((td/'workflow'/'state.json').read_text())
        outcome = {'task': p['task'], 'ref': str(ref), 'task_dir': str(td), 'class': cls, 'reasons': reasons,
                   'iterations': state['completed'], 'arm': 'agentic (ascg)',
                   'states': _states.labels(_states.of_run(state))}
        P.finalize(ref, td, outcome, log)
        _annotate_outcome(td, proto)
        _set_job(td, status='done', outcome=cls, message='; '.join(reasons), finished=_now())
        return {'status': 'done', 'class': cls}
    except P.ScientificFailure as e:
        _set_job(td, status='stopped_on_gate', message=str(e), finished=_now())
        return {'status': 'stopped_on_gate', 'reason': str(e)}
    except Exception as e:
        _set_job(td, status='error', message=f'{type(e).__name__}: {e}', trace=traceback.format_exc(limit=4),
                 finished=_now())
        raise


def _alive(job_rec):
    if job_rec.get('pid'):
        try:
            os.kill(int(job_rec['pid']), 0)
            return True
        except OSError:
            return False
    if job_rec.get('id') and _which('squeue'):
        r = subprocess.run(['squeue', '-h', '-j', str(job_rec['id']), '-o', '%T'], capture_output=True, text=True)
        return bool(r.stdout.strip())
    return None


@verb
def task_state(p):
    """Compact state of several task dirs (one ssh call per host)."""
    out = {}
    for t in p['tasks']:
        td = Path(t['task_dir'])
        d = {'job': json.loads(_job_file(td).read_text()) if _job_file(td).exists() else None}
        sf = td/'workflow'/'state.json'
        if sf.exists():
            s = json.loads(sf.read_text())
            h = s['history'][-1] if s['history'] else None
            d['completed'] = s['completed']
            d['failed'] = s.get('failed')
            d['cost'] = _cost(td, s)
            Ts = [_states.label(T) for T in _states.of_run(s)]
            d['states'] = Ts
            if h:
                d['last'] = {'iteration': h['iteration'], 'family': h['family'],
                             'worst_rdf': {T: round(max(h[f'rdf_{T}'].values()), 4) for T in Ts if f'rdf_{T}' in h},
                             'pressure_bar': {T: round(h[f'p_bar_{T}']) for T in Ts if f'p_bar_{T}' in h}}
                npt = [e for e in s['history'] if f'density_error_{Ts[0]}' in e]
                if npt:
                    d['last_npt'] = {'iteration': npt[-1]['iteration'],
                                     **{T: round(npt[-1][f'density_error_{T}'], 4) for T in Ts}}
        of = td/'report'/'outcome.json'
        if of.exists():
            o = json.loads(of.read_text())
            d['outcome'] = {'class': o['class'], 'reasons': o.get('reasons', [])}
        d['alive'] = _alive(t.get('job') or {})
        jid = (t.get('job') or {}).get('id')
        if jid and d['alive'] and (d['job'] or {}).get('status') == 'submitted' and _which('squeue'):
            d['queue'] = _queue(jid)
        out[t['key']] = d
    return out


@verb
def cancel(p):
    j = p['job']
    if j.get('id'):
        r = subprocess.run(['scancel', str(j['id'])], capture_output=True, text=True)
        return {'cancelled': j['id'], 'rc': r.returncode}
    pid = int(j['pid'])
    mark = _proc_marker(pid)
    if mark != p['marker']:
        raise RuntimeError(f'process {pid} is not this task ({mark!r} != {p["marker"]!r}); not touching it')
    os.killpg(os.getpgid(pid), 15)
    return {'killed_process_group_of': pid}


def _annotate_outcome(td, study_protocol):
    """outcome.json: the study's protocol (not the default one) and the settings each invocation actually used."""
    of = Path(td)/'report'/'outcome.json'
    if not of.exists():
        return
    o = json.loads(of.read_text())
    st = json.loads((Path(td)/'workflow'/'state.json').read_text())
    o['protocol'] = study_protocol
    o['run_history'] = [{k: inv['settings'].get(k) for k in ('iters', 'schedule', 'pressure_correction', 'dihedrals',
                                                            'model', 'lam', 'dt', 'states', 'temperatures',
                                                            'state_weights')}
                        | {'start_iteration': inv.get('start_iteration')}
                        for inv in st.get('invocations', [])]
    o['first_settings'] = {k: st['settings'].get(k) for k in ('model', 'dihedrals', 'pressure_correction')}
    o['code_digests'] = {f.stem.split('_')[-1]: json.loads(f.read_text())['digest']
                         for f in sorted((Path(td)/'workflow').glob('ascg_code_*.json'))}
    o['reference_task'] = st.get('task')          # the reference's own name (the study task key may differ)
    of.write_text(json.dumps(o, indent=2, default=str) + '\n')
    tr = Path(td)/'report'/'technical_report.md'    # written by the pipeline with its default protocol: replace
    if tr.exists():
        import re
        txt = tr.read_text()
        txt = re.sub(r'\(protocol [^)]*\)', f"(protocol {study_protocol.get('version')})", txt, count=1)
        txt = re.sub(r'## Protocol\n```json\n.*?\n```', lambda _: '## Protocol (the study\'s locked protocol)\n```json\n'
                     + json.dumps(study_protocol, indent=2) + '\n```', txt, count=1, flags=re.S)
        tr.write_text(txt)
    rd = Path(td)/'report'/'model_files'/'README.md'
    if rd.exists():
        rd.write_text(rd.read_text().replace(f"(protocol {_pipeline_version()},",
                                             f"(protocol {study_protocol.get('version')},", 1))


def _pipeline_version():
    from ascg_core.ibi.pipeline import PROTOCOL
    return PROTOCOL['version']


def _check_gates(proto):
    """The pipeline classifies with its own v1 gates (pipeline.classify): a study protocol whose gates, chain band
    or validation settings differ cannot be honoured by this engine version - refuse rather than mislabel."""
    from ascg_core.ibi.pipeline import PROTOCOL
    diff = [k for k in ('gates', 'chain_band') if proto.get(k) != PROTOCOL.get(k)]
    if diff:
        raise RuntimeError(f'study protocol differs from the engine\'s classifier in {diff}; unsupported by this '
                           'engine version (classify uses pipeline.PROTOCOL)')


@verb
def classify(p):
    """Outcome class of a task directory from its held-out validation (for adopted runs without an outcome)."""
    from ascg_core.ibi import pipeline as P
    vf = Path(p['task_dir'])/'cg'/'ibi'/'final'/'validation.json'
    if not vf.exists():
        return {'class': None, 'reasons': ['no held-out validation in this directory']}
    cls, reasons = P.classify(json.loads(vf.read_text()))
    return {'class': cls, 'reasons': reasons, 'from': str(vf)}


@verb
def job_log(p):
    """Tail of the task's latest (or given) job log."""
    td = Path(p['task_dir'])
    n = p.get('job_n') or (json.loads(_job_file(td).read_text()).get('job_n') if _job_file(td).exists() else None)
    if n is None:
        raise RuntimeError('no ascg job recorded in this task directory')
    f = td/'workflow'/f'ascg_job_{int(n):02d}.out'
    lines = f.read_text(errors='replace').splitlines() if f.exists() else []
    k = int(p.get('lines', 60))
    return {'job_n': int(n), 'log': str(f), 'lines_total': len(lines), 'tail': lines[-k:]}


@verb
def digest(p):
    """Numbers-only history of a task for diagnosis."""
    td = Path(p['task_dir'])
    job = json.loads(_job_file(td).read_text()) if _job_file(td).exists() else None
    if not (td/'workflow'/'state.json').exists():
        return {'completed': None, 'note': 'no IBI state yet: the job is queued, starting, or failed before iteration 0 '
                                           '(see diag joblog)', 'job': job}
    s = json.loads((td/'workflow'/'state.json').read_text())
    h = s['history']
    cur = ((s.get('invocations') or [{}])[-1].get('settings')) or s['settings']   # the latest invocation's settings
    Ts = [_states.label(T) for T in _states.of_run(s)]
    rows = [{'iteration': e['iteration'], 'family': e['family'],
             'worst_rdf': {T: round(max(e[f'rdf_{T}'].values()), 4) for T in Ts},
             'pressure_bar': {T: round(e[f'p_bar_{T}']) for T in Ts},
             **({'npt_density_error': {T: round(e[f'density_error_{T}'], 4) for T in Ts}}
                if f'density_error_{Ts[0]}' in e else {})} for e in h[-int(p.get('last', 12)):]]
    return {'completed': s['completed'], 'failed': s.get('failed'), 'rewinds': s.get('rewinds', []),
            'states': Ts,
            'recent': rows, 'npt_history': [{'iteration': e['iteration'], **{T: round(e[f'density_error_{T}'], 4)
                                                                             for T in Ts}}
                                            for e in h if f'density_error_{Ts[0]}' in e],
            'settings': {k: cur.get(k) for k in ('model', 'dihedrals', 'lam', 'dt', 'schedule', 'pressure_correction',
                                                   'temperatures', 'state_weights')},
            'invocations': _invocations(td, s), 'stage1_end': _stage1_end(td, s), 'cost': _cost(td, s),
            'job': job}


@verb
def rewind(p):
    from ascg_core.ibi import runner
    runner.main(['rewind', '--task-dir', p['task_dir'], '--to-iteration', str(p['iteration']), '--reason',
                 p.get('reason', '')])
    return {'rewound_to': p['iteration']}


@verb
def rescore(p):
    from ascg_core.ibi import pipeline as P
    return P.rescore(p['ref'], p['task_dir'], reason=p.get('reason', 'amendment'))


# ------------------------------------------------------------------------------------------------ run history
def _invocations(td, s):
    """The fitting invocations of a run (each continuation adds one): iterations from / to, schedule, pressure
    correction, damping and the timestep actually used (dt_ps of its first iteration)."""
    inv = s.get('invocations') or [{'start_iteration': 0, 'settings': s.get('settings', {})}]
    out = []
    for i, v in enumerate(inv):
        st = v.get('settings') or {}
        a = v.get('start_iteration', 0)
        b = inv[i + 1].get('start_iteration', 0) - 1 if i + 1 < len(inv) else s.get('completed')
        mf = Path(td)/'cg'/'ibi'/f'iter_{max(int(a), 1):03d}'/'metrics.json'
        dt = json.loads(mf.read_text()).get('dt_ps') if mf.exists() else st.get('dt')
        out.append({'from': a, 'to': b, 'schedule': st.get('schedule'), 'pressure_correction':
                    st.get('pressure_correction'), 'lam': st.get('lam'), 'dt_ps': dt, 'time': v.get('time')})
    return out


def _stage1_end(td, s):
    """Last iteration of stage 1: the end of the leading invocations that share the first one's schedule and pressure
    correction (a rewind into stage 1 may add one)."""
    inv = _invocations(td, s)
    end = inv[0]['to']
    for v in inv[1:]:
        if (v['schedule'], v['pressure_correction']) != (inv[0]['schedule'], inv[0]['pressure_correction']):
            break
        end = v['to']
    return end


def _cost(td, s=None):
    """Measured cost: GPU-hours spent (finished stages plus the running one), mean wall seconds of the recent
    iterations (NPT checks included), simulated ps per iteration and state."""
    import statistics
    wf = Path(td)/'workflow'
    tf = wf/'stage_timings.json'
    t = json.loads(tf.read_text()) if tf.exists() else {}
    spent = sum(t.get(k, 0.0) for k in ('stage1', 'stage2', 'extension', 'validate'))
    pv, job = wf/'provenance.jsonl', json.loads(_job_file(td).read_text()) if _job_file(td).exists() else {}
    if pv.exists() and job.get('status') == 'running':
        ev = [json.loads(l) for l in pv.read_text().splitlines() if '"stage start"' in l or '"stage done"' in l]
        if ev and ev[-1]['event'] == 'stage start':
            spent += max(time.time() - time.mktime(time.strptime(ev[-1]['time'], '%Y-%m-%d %H:%M:%S')), 0.0)
    s = s or (json.loads((wf/'state.json').read_text()) if (wf/'state.json').exists() else {})
    walls = [e['wall_s'] for e in s.get('history', [])[-12:] if e.get('wall_s')]
    st = ((s.get('invocations') or [{}])[-1].get('settings')) or s.get('settings') or {}
    return {'spent_gpu_h': spent / 3600, 's_per_iteration': statistics.mean(walls) if walls else None,
            'fit_ps_per_iteration': float(st.get('eq_ps', 50.0)) + float(st.get('prod_ps', 500.0))}


# ------------------------------------------------------------------------------------------------ diagnostics
def _system(ref, td):
    from ascg_core.ibi.system import load_system
    from ascg_core.ibi.dihedrals import style_of
    s = load_system(ref)
    st = json.loads((Path(td)/'workflow'/'state.json').read_text())
    style = style_of(st['settings'].get('dihedrals'))
    if style:
        s.add_dihedrals(style)
    return s


@verb
def diag_bonds(p):
    """Frame-by-frame bond lengths of one run against the run's bond tables: longest per frame, persistence beyond
    the break length, energy up the wall, AA-sampled range. run: run_T<T> | npt_T<T> | final/holdout_T<T>/seed_<k>."""
    import numpy as np
    from ascg_core.ibi import analysis
    from ascg_core.ibi.tables import load_tpotentials, tpotentials_at
    from ascg_core.mapping.cg_mapping import bond_lengths
    td, ref = Path(p['task_dir']), Path(p['ref'])
    if p.get('state'):
        T = _states.parse_one(p['state'])
    else:                                               # default: the run's only state
        Ts = list(_run_temps(td, ref))
        if len(Ts) != 1:
            raise RuntimeError(f'give --state: one of {_states.labels(Ts)}')
        T = Ts[0]
    s = _system(ref, td)
    if p.get('iterations'):
        return {'state': _states.label(T), 'runs': {int(i): _bonds_one(td, ref, T, s, int(i), p.get('run'))
                                                     for i in p['iterations']}}
    return _bonds_one(td, ref, T, s, p.get('iteration'), p.get('run'))


def _bonds_one(td, ref, T, s, iteration, run_name):
    import numpy as np
    from ascg_core.ibi import analysis
    from ascg_core.ibi.tables import load_tpotentials, tpotentials_at
    from ascg_core.mapping.cg_mapping import bond_lengths
    p = {'iteration': iteration, 'run': run_name}
    if (p.get('run') or '').startswith('final'):
        tab_dir, run = td/'cg'/'ibi'/'final'/'tables', td/'cg'/'ibi'/p['run']
    else:
        name = p.get('run') or f'run_T{T}'
        if p.get('iteration') is None:                  # default: the latest iteration that has this run
            have = sorted(d for d in (td/'cg'/'ibi').glob('iter_[0-9][0-9][0-9]') if (d/name/'run.json').exists())
            if not have:
                raise RuntimeError(f'no iteration has a {name} run yet')
            it = int(have[-1].name.split('_')[1])
        else:
            it = int(p['iteration'])
        tab_dir, run = td/'cg'/'ibi'/f'iter_{it:03d}'/'tables', td/'cg'/'ibi'/f'iter_{it:03d}'/name
    sp = tpotentials_at(load_tpotentials(tab_dir), T)
    lim = analysis.break_lengths(sp['bond'], T)
    meta = json.loads((run/'run.json').read_text())
    fr = analysis.trajectory_frames(run/'traj.fmd', skip_ps=0, dt_ps=meta['dt_ps'])
    bl = np.array([bond_lengths(q, s.bonds) for q, _ in fr])
    t = np.load(ref/f'targets_T{T}.npz')
    out = {'run': str(run.relative_to(td)), 'ensemble': meta.get('ensemble'), 'frames': len(fr), 'dt_ps': meta['dt_ps'],
           'volume_first_last_nm3': [round(float(np.prod(np.asarray(fr[0][1])[:3])), 1),
                                     round(float(np.prod(np.asarray(fr[-1][1])[:3])), 1)], 'types': {}}
    for i, k in enumerate(s.bond_types):
        x = bl[:, s.bond_type == i]
        over = x > lim[k]
        persist = int((over[1:] & over[:-1]).sum())
        pot = sp['bond'][k]
        j = int(np.argmin(pot.u))
        e = t[f'bond_edges/{k}']
        c, P = 0.5 * (e[1:] + e[:-1]), t[f'bond/{k}']
        on = P > 1e-3 * P.max()
        out['types'][k] = {'break_length_nm': round(lim[k], 4), 'longest_nm': round(float(x.max()), 4),
                           'longest_energy_above_min_kj': round(float(np.interp(x.max(), pot.x, pot.u) - pot.u[j]), 1),
                           'bonds_ever_beyond': int(over.any(0).sum()),
                           'single_frame_excursions': int(((over.sum(0) == 1)).sum()),
                           'consecutive_frame_pairs_beyond': persist,
                           'aa_sampled_nm': [round(float(c[on].min()), 4), round(float(c[on].max()), 4)],
                           'table_end_nm': round(float(pot.x[-1]), 4),
                           'table_end_energy_kj': round(float(pot.u[-1] - pot.u[j]), 1),
                           'table_end_force_kj_nm': round(float(pot.force()[-1]), 1)}
    return out


@verb
def diag_pressure(p):
    """Measured pressure change between consecutive iterations vs the first-order virial prediction of the table change."""
    import numpy as np
    from ascg_core.ibi import analysis
    from ascg_core.ibi.tables import load_tpotentials
    from ascg_core.ibi.update import table_change_derivative, virial_pressure_change
    td, ref = Path(p['task_dir']), Path(p['ref'])
    s = _system(ref, td)
    rows = []
    done = json.loads((td/'workflow'/'state.json').read_text())['completed']
    for it in p.get('iterations') or range(max(done - 5, 1), done + 1):     # default: the last six iterations
        prev, cur = td/'cg'/'ibi'/f'iter_{it - 1:03d}', td/'cg'/'ibi'/f'iter_{it:03d}'
        pm, cm = json.loads((prev/'metrics.json').read_text()), json.loads((cur/'metrics.json').read_text())
        old, new = load_tpotentials(prev/'tables')['pair'], load_tpotentials(cur/'tables')['pair']
        r = {'iteration': it, 'family': cm['family_updated']}
        for T in _run_temps(td, ref):
            meta = json.loads((prev/f'run_T{T}'/'run.json').read_text())
            fr = analysis.trajectory_frames(prev/f'run_T{T}'/'traj.fmd', skip_ps=meta['eq_ps'], dt_ps=meta['dt_ps'])
            h, e = analysis.pair_histograms(fr, s, 1.5)
            V = float(np.mean([np.prod(np.asarray(b)[:3]) for _, b in fr]))
            pred = virial_pressure_change(h, e, V, {k: table_change_derivative(old[k], new[k], T) for k in s.pair_keys})
            r[str(T)] = {'predicted_bar': round(pred), 'measured_bar': round(cm['states'][str(T)]['pressure_bar'] -
                                                                            pm['states'][str(T)]['pressure_bar']),
                         'p_bar': round(cm['states'][str(T)]['pressure_bar'])}
        rows.append(r)
    return rows


@verb
def diag_chains(p):
    """Rg of the CG chains over iterations (and validation) against the AA reference."""
    import numpy as np
    from ascg_core.ibi import analysis
    from ascg_core.mapping.cg_mapping import chain_size
    td, ref = Path(p['task_dir']), Path(p['ref'])
    s = _system(ref, td)
    summ = json.loads((ref/'summary.json').read_text())
    rows = []
    its = p.get('iterations')
    if not its:                     # default: BI start, end of stage 1, in-loop NPT checks, last (at most 8)
        st = json.loads((td/'workflow'/'state.json').read_text())
        its = sorted({0, _stage1_end(td, st), st['completed']} |
                     {e['iteration'] for e in st['history'] if any(k.startswith('density_error') for k in e)})
        if len(its) > 8:
            its = sorted({its[0], its[-1]} | set(its[1:-1][::-(-(len(its) - 2) // 6)]))
    for it in its:
        r = {'iteration': it}
        for T in _run_temps(td, ref):
            run = td/'cg'/'ibi'/f'iter_{it:03d}'/f'run_T{T}'
            if not (run/'traj.fmd').exists():
                continue
            meta = json.loads((run/'run.json').read_text())
            fr = analysis.trajectory_frames(run/'traj.fmd', skip_ps=meta['eq_ps'], dt_ps=meta['dt_ps'])
            rg = np.mean([np.sqrt(np.mean(chain_size(q, s.masses, s.mol)[0]**2)) for q, _ in fr[::5]])
            r[str(T)] = round(float(rg), 4)
        rows.append(r)
    return {'aa_rg_nm': {T: summ['states'][T]['rg_nm'] for T in summ['states']}, 'cg_rg_rms_nm': rows}


@verb
def audit_tables(p):
    """Bond tables outside the AA-sampled range against the harmonic walls 1/2 k (b - b0)^2 (k = kT / var), table ends,
    and per-family table extents (one iteration or the final tables)."""
    import numpy as np
    from ascg_core.ibi import fastmd_io as fio
    from ascg_core.ibi.tables import BOND_BLEND_NM, load_tpotentials
    td, ref = Path(p['task_dir']), Path(p['ref'])
    if p.get('iteration') is not None:
        tab = td/'cg'/'ibi'/f'iter_{int(p["iteration"]):03d}'/'tables'
    elif (td/'cg'/'ibi'/'final'/'tables'/'potentials_tdep.json').exists():
        tab = td/'cg'/'ibi'/'final'/'tables'
    else:                                               # no final tables yet (fit running): the latest iteration
        have = sorted(d for d in (td/'cg'/'ibi').glob('iter_[0-9][0-9][0-9]') if (d/'tables').is_dir())
        if not have:
            raise RuntimeError('no tables yet')
        tab = have[-1]/'tables'
    tp = load_tpotentials(tab)
    out = {'tables': str(tab.relative_to(td)), 'bond': {}, 'extent': {}}
    worst = 0.0
    for T in _run_temps(td, ref):
        t = np.load(ref/f'targets_T{T}.npz')
        for k, q in tp['bond'].items():
            e = t[f'bond_edges/{k}']
            c, P = 0.5 * (e[1:] + e[:-1]), np.clip(t[f'bond/{k}'], 0, None)
            on = P > 1e-3 * P.max()
            lo, hi = c[on].min(), c[on].max()
            w = P / P.sum()
            b0 = (w * c).sum()
            kk = fio.kT(T) / (w * (c - b0)**2).sum()
            u = q.a + q.b * float(T)
            devs = []
            for sel, xr in ((q.x >= hi + BOND_BLEND_NM, hi + BOND_BLEND_NM), (q.x <= lo - BOND_BLEND_NM, lo - BOND_BLEND_NM)):
                if sel.sum() >= 3:
                    ur = np.interp(xr, q.x, u)
                    devs.append(float(np.abs(u[sel] - (0.5 * kk * ((q.x[sel] - b0)**2 - (xr - b0)**2) + ur)).max()))
            dev = max(devs) if devs else None
            worst = max(worst, dev or 0.0)
            out['bond'][f'{T}/{k}'] = {'k_kj_nm2': round(float(kk), 1), 'b0_nm': round(float(b0), 4),
                                       'sampled_nm': [round(float(lo), 4), round(float(hi), 4)],
                                       'max_dev_from_harmonic_kj': None if dev is None else round(dev, 4),
                                       'table_end_nm': round(float(q.x[-1]), 4),
                                       'end_energy_above_min_kj': round(float(u[-1] - u.min()), 1)}
    for fam, d in tp.items():
        out['extent'][fam] = {k: [round(float(q.x[0]), 4), round(float(q.x[-1]), 4)] for k, q in d.items()}
    out['harmonic_walls_ok'] = worst < 0.1
    out['worst_dev_kj'] = round(worst, 4)
    return out


@verb
def diag_structure(p):
    """Per-type NRMSE at the BI start, the end of stage 1 and the last iteration (from metrics.json)."""
    td = Path(p['task_dir'])
    s = json.loads((td/'workflow'/'state.json').read_text())
    s1 = _stage1_end(td, s)
    its = sorted({0, s1, s['completed']} & {e['iteration'] for e in s['history']})
    out = {}
    for it in its:
        m = json.loads((td/'cg'/'ibi'/f'iter_{it:03d}'/'metrics.json').read_text())
        out[it] = {T: {fam: {k: round(v, 4) for k, v in m['states'][T][fam].items()}
                       for fam in ('rdf', 'bond', 'angle', 'dihedral') if fam in m['states'][T]}
                   for T in m['states']}
    return {'bi_start': 0, 'stage1_end': s1, 'last': s['completed'], 'by_iteration': out}


@verb
def fetch(p):
    """Small files (reports, figures) as base64 with md5, from inside a task dir."""
    f = Path(p['path'])
    data = f.read_bytes()
    if len(data) > int(p.get('max_bytes', 20_000_000)):
        raise RuntimeError(f'{f} is larger than max_bytes')
    return {'b64': base64.b64encode(data).decode(), 'md5': hashlib.md5(data).hexdigest(), 'bytes': len(data)}


@verb
def fetch_dir(p):
    """A directory (a task's model files) as a gzipped tar, base64 with md5."""
    import io
    import tarfile
    d = Path(p['path'])
    if not d.is_dir():
        raise RuntimeError(f'{d} is not a directory')
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as t:
        t.add(d, arcname=d.name)
    data = buf.getvalue()
    if len(data) > int(p.get('max_bytes', 100_000_000)):
        raise RuntimeError(f'{d} is larger than max_bytes when packed')
    return {'b64': base64.b64encode(data).decode(), 'md5': hashlib.md5(data).hexdigest(), 'bytes': len(data)}


@verb
def list_files(p):
    return sorted(str(Path(x)) for x in glob.glob(p['pattern']))


# ------------------------------------------------------------------------------------------------ AA execution
# One AA campaign per study system: <dir> = <aa root>/<study>/<key> on the target, with inputs/ (the study's derived
# recipe and mapping library), build/ (AutoPoly), protocol/ (GROMACS), cg_ref/ (map-aa) and jobs/ (scripts, logs and
# one status file per job). Jobs carry the ASCG_TASK marker "<study>/aa:<key>".
def _aa_paths(p):
    d = Path(p['dir'])
    return d, d/'jobs', d/'build'/p['key']


@verb
def aa_stage(p):
    """Write the campaign's input files (recipes.json, library.json, system.json) and return their md5."""
    d = Path(p['dir'])
    (d/'inputs').mkdir(parents=True, exist_ok=True)
    (d/'jobs').mkdir(exist_ok=True)
    out = {}
    for name, text in p['files'].items():
        if '/' in name or name.startswith('.'):
            raise RuntimeError(f'bad input file name {name!r}')
        (d/'inputs'/name).write_text(text)
        out[name] = hashlib.md5(text.encode()).hexdigest()
    return {'dir': str(d), 'md5': out}


def _aa_commands(p, d, key):
    py, stage = p.get('python', 'python'), p['stage']
    lib = '--mappings inputs/library.json' if (d/'inputs'/'library.json').exists() else ''
    if stage == 'build':
        cmds = ['mkdir -p build',
                f'{py} -m ascg_core build-opls --recipes inputs/recipes.json {lib} --task-key {key} --out build '
                f'--seed 42 > build/{key}.build.json',
                f'{py} -m ascg_core neutralize-aa --recipes inputs/recipes.json --system-dir build/{key} --task-key {key} '
                f'> build/{key}.neutral.json',
                f'{py} -m ascg_core audit-opls --data build/{key}/system.neutral.data > build/{key}.audit.json || true']
        if lib:
            cmds.append(f'{py} -m ascg_core cg-topology --recipes inputs/recipes.json {lib} --system-dir build/{key} '
                        f'--task-key {key} > build/{key}.cg.json')
        return cmds
    if stage == 'md':
        flag = p.get('require_cpu_flag')
        cmds = [f'grep -q {shlex.quote(flag)} /proc/cpuinfo || {{ echo "$(hostname) lacks {flag}"; exit 2; }}'] if flag else []
        gpu = '-nb gpu -pme gpu -bonded gpu -update auto' if p.get('mode') == 'gpu' else ''
        cmds.append(f'{py} -m ascg_core.aa.gmx_protocol ' + ' '.join(shlex.quote(f'protocol/{x}') for x in p['dirs'])
                    + f' --gmx {shlex.quote(p.get("gmx", "gmx"))} --mdrun {shlex.quote(p.get("mdrun", "gmx mdrun"))}'
                    + f' --mdrun-flags {shlex.quote("-ntomp " + str(p.get("cpus", 16)))} --gpu-flags {shlex.quote(gpu)}'
                    + (f' --maxh {float(p["maxh"])}' if p.get('maxh') else ''))
        cmds.append('for x in ' + ' '.join(f'protocol/{x}' for x in p['dirs'])
                    + '; do grep -h "^Performance" $x/*.log 2>/dev/null | tail -1 | sed "s#^#$x #"; done')
        return cmds
    if stage == 'map':
        runs = ' '.join(f'--run {lab}=protocol/T{lab}' for lab in p['labels'])
        return ['mkdir -p cg_ref',
                f'{py} -m ascg_core map-aa --system-dir build/{key} --task-key {key} --recipes inputs/recipes.json '
                f'{lib} {runs} --out cg_ref/{key} > cg_ref/{key}.summary.json']
    raise RuntimeError(f'unknown AA stage {stage!r}')


@verb
def aa_launch(p):
    """Write and start one AA job (stage build | md | map): Slurm (sbatch, optionally after another job) or a new
    session on a workstation. The script records its status in jobs/<name>.json."""
    d, jobs, _ = _aa_paths(p)
    jobs.mkdir(parents=True, exist_ok=True)
    key, stage = p['key'], p['stage']
    n = len(list(jobs.glob(f'{stage}_*.sh'))) + 1
    name = f'{stage}_{n:02d}' + (f'_{p["label"]}' if p.get('label') else '')
    s = p.get('submit') or {}
    lines = ['#!/bin/bash']
    if p.get('scheduler') == 'slurm':
        lines += [f'#SBATCH --job-name=ascg-aa-{key}-{stage}', f'#SBATCH --output={jobs}/{name}.out', '#SBATCH --nodes=1',
                  '#SBATCH --ntasks=1', f'#SBATCH --cpus-per-task={p.get("cpus", s.get("cpus", 2))}',
                  f'#SBATCH --mem={s.get("mem", "16G")}', f'#SBATCH --time={s.get("time", "08:00:00")}']
        for k in ('qos', 'partition', 'account', 'gres'):
            if s.get(k):
                v = s[k] if isinstance(s[k], str) else ','.join(s[k])
                lines.append(f'#SBATCH --{k}={v}')
    status = f'{jobs}/{name}.json'
    lines += [f'{p["env"]} >/dev/null 2>&1' if p.get('env') else ':',
              f'{p["aa_env"]} >/dev/null 2>&1' if p.get('aa_env') and stage == 'md' else ':',
              f'export {MARK}={p["study"]}/aa:{key}', f'cd {shlex.quote(str(d))}',
              f'echo "{{\\"status\\": \\"running\\", \\"host\\": \\"$(hostname)\\", \\"start\\": \\"$(date "+%F %T")\\"}}" > {status}',
              f'trap \'rc=$?; s=done; [ $rc = 0 ] || s=failed; echo "{{\\"status\\": \\"$s\\", \\"exit\\": $rc, '
              f'\\"end\\": \\"$(date "+%F %T")\\"}}" > {status}\' EXIT',
              'set -eo pipefail', *_aa_commands(p, d, key)]
    script = jobs/f'{name}.sh'
    script.write_text('\n'.join(lines) + '\n')
    script.chmod(0o755)
    rec = {'name': name, 'stage': stage, 'script': str(script), 'status_file': status, 'log': f'{jobs}/{name}.out',
           'submitted': _now(), **({'dirs': p['dirs']} if p.get('dirs') else {}),
           **({'after': p['after']} if p.get('after') else {})}
    if p.get('scheduler') == 'slurm':
        cmd = ['sbatch', '--parsable'] + ([f'--dependency=afterok:{p["after"]}'] if p.get('after') else []) + [str(script)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'sbatch failed: {r.stderr.strip()}')
        return {**rec, 'id': r.stdout.strip().split(';')[0]}
    env = {**os.environ, MARK: f'{p["study"]}/aa:{key}'}
    if p.get('gpu') is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(p['gpu'])
    proc = subprocess.Popen(['bash', str(script)], stdout=open(rec['log'], 'w'), stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    return {**rec, 'pid': proc.pid, 'pgid': proc.pid}


@verb
def aa_prepare(p):
    """GROMACS conversion and protocol (prepare-gmx-protocol) for the built system; quick, runs where called."""
    d, _, sysdir = _aa_paths(p)
    if not (sysdir/'system.neutral.data').exists():
        raise RuntimeError(f'no built system in {sysdir} (aa build first)')
    args = [p.get('python', sys.executable), '-m', 'ascg_core', 'prepare-gmx-protocol', '--system-dir', str(sysdir),
            '--out', str(d/'protocol'), '--states', p['states'], '--equilibration', p['equilibration'],
            '--eq-ns', str(p['eq_ns']), '--prod-ns', str(p['prod_ns']), '--pre-ns', str(p['pre_ns'])]
    if p.get('ff_override'):
        args += ['--ff-override', p['ff_override']]
    r = subprocess.run(args, capture_output=True, text=True, cwd=str(d))
    if r.returncode != 0:
        raise RuntimeError(f'prepare-gmx-protocol failed: {(r.stderr or r.stdout)[-800:]}')
    return json.loads((d/'protocol'/'protocol.json').read_text())


def _aa_job_state(j):
    out = {k: j.get(k) for k in ('name', 'stage', 'id', 'pid', 'dirs')}
    sf = Path(j['status_file'])
    st = json.loads(sf.read_text()) if sf.exists() else {'status': 'submitted'}
    out.update(st)
    alive = _alive(j)
    if st['status'] in ('submitted', 'running') and alive is False:
        out['status'] = 'lost'                          # ended without writing its status (killed, time limit)
    if out['status'] == 'submitted' and j.get('id') and alive:
        out['queue'] = _queue(j['id'])
    if out['status'] in ('failed', 'lost') and Path(j['log']).exists():
        out['log_tail'] = Path(j['log']).read_text(errors='replace').splitlines()[-12:]
    return out


@verb
def aa_state(p):
    """State of one AA campaign: inputs, build (atoms, charge, audit), protocol progress per directory (steps done,
    next step, last ns/day), mapped reference, and the jobs."""
    from ascg_core.aa import gmx_protocol as G
    d, _, sysdir = _aa_paths(p)
    key = p['key']
    out = {'dir': str(d), 'jobs': [_aa_job_state(j) for j in p.get('jobs', [])]}
    b = d/'build'
    if (b/f'{key}.audit.json').exists():
        try:
            a = json.loads((b/f'{key}.audit.json').read_text())
            nz = json.loads((b/f'{key}.neutral.json').read_text())
            out['build'] = {'atoms': a.get('atoms'), 'molecules': a.get('molecules'), 'types': a.get('types'),
                            'net_charge_e': a.get('net_charge_e'), 'charged_molecules': a.get('charged_molecules'),
                            'molecules_shifted': nz.get('molecules_shifted'),
                            'charge_fix_opls_types': nz.get('adjust_opls_types'),
                            'cg_topology': (b/f'{key}.cg.json').exists()}
            cache = b/f'{key}.typing_audit.json'                 # the OPLS typing audit, once per build
            if not cache.exists() or cache.stat().st_mtime < (sysdir/'system.neutral.data').stat().st_mtime:
                cache.write_text(json.dumps(aa_audit({'system_dir': str(sysdir)})) + '\n')
            out['build']['audit'] = json.loads(cache.read_text())
        except (OSError, ValueError, RuntimeError) as e:
            out['build'] = {'error': f'{type(e).__name__}: {e}'}
    pj = d/'protocol'/'protocol.json'
    if pj.exists():
        proto = json.loads(pj.read_text())
        dirs = [proto['equilibration_dir'], *proto['branch_dirs']]
        prog = {}
        for x in dirs:
            xd = d/'protocol'/x
            if (xd/'steps.json').exists():
                s = G.status(xd)
                perf = None
                for log in sorted(xd.glob('*.log'), key=lambda f: f.stat().st_mtime):
                    for line in log.read_text(errors='replace').splitlines():
                        if line.startswith('Performance:'):
                            perf = float(line.split()[1])
                prog[x] = {**s, 'ns_per_day': perf}
        out['protocol'] = {'equilibration': proto['equilibration'], 'branches': proto['branch_dirs'], 'progress': prog}
    ref = d/'cg_ref'/key
    if (ref/'summary.json').exists():
        out['reference'] = str(ref)
    return out


@verb
def aa_cancel(p):
    """Cancel this campaign's own jobs (Slurm ids, or process groups carrying its marker)."""
    done = []
    for j in p['jobs']:
        if j.get('id'):
            subprocess.run(['scancel', str(j['id'])], capture_output=True, text=True)
            done.append(j['id'])
        elif j.get('pid') and _alive(j):
            pid = int(j['pid'])
            if _proc_marker(pid) != p['marker']:
                raise RuntimeError(f'process {pid} is not this campaign; not touching it')
            os.killpg(os.getpgid(pid), 15)
            done.append(pid)
    return {'cancelled': done}


@verb
def aa_check(p):
    """What the AA path needs on this host: AutoPoly (build), MDAnalysis (mapping), GROMACS (after the AA env line)."""
    out = {}
    for mod in ('AutoPoly', 'MDAnalysis', 'rdkit'):
        r = subprocess.run([sys.executable, '-c', f'import {mod}'], capture_output=True, text=True)
        out[mod] = r.returncode == 0
    gmx = p.get('gmx', 'gmx')
    line = f'{p.get("env") or ":"} >/dev/null 2>&1; {p.get("aa_env") or ":"} >/dev/null 2>&1; ' \
           f'command -v {shlex.quote(gmx.split()[0])}'                  # as in the job scripts: one shell
    r = subprocess.run(['bash', '-c', line], capture_output=True, text=True)
    out['gmx'] = r.stdout.strip() or None
    return out


# ------------------------------------------------------------------------------------------------ entry
def main():
    v = sys.argv[1] if len(sys.argv) > 1 else ''
    try:
        payload = json.loads(sys.stdin.read() or '{}')
        if v not in VERBS:
            raise RuntimeError(f'unknown verb {v!r}')
        res = VERBS[v](payload)
        print(json.dumps({'ok': True, 'result': res}, default=str))
        return 0
    except Exception as e:
        print(json.dumps({'ok': False, 'error': f'{type(e).__name__}: {e}'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
