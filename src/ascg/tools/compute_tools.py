"""compute: user-defined targets (no site is built in)."""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from ..compute import DEFAULTS, remote, target
from ..config import load_profiles, profile_path, save_profiles
from ..registry import P, PolicyError, ToolError, tool


@tool('compute', 'list', 'Compute targets of the user profile (file, transport, scheduler, GPU policy).')
def compute_list():
    targets, path = load_profiles()
    return {'profile': str(path) if path else None,
            'targets': {k: {kk: v.get(kk) for kk in ('transport', 'host', 'scheduler', 'root', 'work_root', 'gpus',
                                                    'gpu_policy', 'submit')} for k, v in targets.items()}}


@tool('compute', 'add', 'Add or update a compute target in the user profile. USER ACTION: its env line runs as a '
      'shell command on every call, so agents give the user the exact command to run (not served over MCP).', {
    'name': P(required=True, help='target name'),
    'transport': P(enum=('ssh', 'local'), default='ssh'), 'host': P(help='ssh alias (transport ssh)'),
    'scheduler': P(enum=('slurm', 'none'), default='none'),
    'root': P(help='directory on the target that contains the ascg / ascg_core sources (transport local: '
                   'default this checkout)'),
    'work_root': P(help='where task directories go (default <root>/runs/ascg)'),
    'env': P(help='shell line run before python, e.g. "source /path/env.sh"'),
    'python': P(default='python'), 'gpus': P('array', help='workstation GPU indices usable, e.g. 0,1'),
    'gpu_policy': P(enum=('avoid_foreign', 'shared'), default='avoid_foreign'),
    'qos': P(help='one QoS, or several comma-separated: each job goes to the first with GPUs left under its '
                  'per-user cap'), 'partition': P(), 'gres': P(default='gpu:1'), 'max_jobs': P('integer'),
    'account': P(), 'cpus': P('integer', help='Slurm CPUs per job (default 4)'),
    'mem': P(help='Slurm memory per job (default 16G)'), 'time': P(help='Slurm time limit per job (default 08:00:00)'),
    'aa': P('object', help='AA settings (aa build / submit / map): {"env": GROMACS environment line, "gmx": "gmx_mpi", '
                           '"mdrun": "mpirun -np 1 gmx_mpi mdrun", "mode": "cpu" | "gpu", "require_cpu_flag": "avx512f", '
                           '"max_core_h": 2000, "build" / "md" / "gpu" / "map": {"qos", "partition", "account", "cpus", '
                           '"mem", "time"}}')},
      mutating=True, mcp=False)
def compute_add(name, transport, scheduler, python, gpu_policy, gres, root=None, host=None, work_root=None, env=None,
                gpus=None, qos=None, partition=None, max_jobs=None, account=None, cpus=None, mem=None, time=None,
                aa=None):
    if transport == 'ssh' and not host:
        raise ToolError('transport ssh needs --host (an ssh alias that works non-interactively)')
    if root is None:
        if transport != 'local':
            raise ToolError('--root: the directory on the target that contains the ascg sources')
        root = str(CHECKOUT)
        if python == 'python':
            python = sys.executable
    targets, path = load_profiles()
    t = {'transport': transport, 'scheduler': scheduler, 'root': root, 'python': python, 'gpu_policy': gpu_policy}
    for k, v in (('host', host), ('work_root', work_root), ('env', env)):
        if v:
            t[k] = v
    if gpus:
        t['gpus'] = [int(g) for g in gpus]
    if aa:
        t['aa'] = json.loads(aa) if isinstance(aa, str) else aa
    if scheduler == 'slurm':
        if qos and ',' in qos:
            qos = [q.strip() for q in qos.split(',') if q.strip()]
        t['submit'] = {k: v for k, v in (('qos', qos), ('partition', partition), ('gres', gres),
                                         ('max_jobs', max_jobs), ('account', account), ('cpus', cpus), ('mem', mem),
                                         ('time', time)) if v is not None}
    targets[name] = t
    return {'saved': str(save_profiles(targets, path)), 'target': {name: t}}


@tool('compute', 'test', 'Connect to a target and report Python, ascg, fastMD binary, scheduler and GPUs.', {
    'target': P(required=True)})
def compute_test(target_name=None, **kw):
    t = target(kw.get('target') or target_name)
    ping = remote(t, 'ping', timeout=120)
    gp = remote(t, 'gpus', timeout=120)
    return {'target': t['name'], 'ping': ping, 'gpus': gp,
            'note': None if ping.get('fastmd_present') else 'FASTMD_BINARY not found: see the ascg-setup-engine skill'}


@tool('compute', 'hosts', 'GPU occupancy (this study\'s vs other processes) and scheduler jobs on every target: ascg '
      'jobs in full (state, expected start and wait), the account\'s other jobs as counts by state.', {
          'targets': P('array', help='default: all'), 'all_jobs': P('boolean', default=False,
                                                                  help='list the other jobs too')})
def compute_hosts(all_jobs=False, targets=None):
    allt, _ = load_profiles()
    out = {}
    for name in (targets or list(allt)):
        t = target(name)
        try:
            d = {'gpus': remote(t, 'gpus', timeout=120)}
            if t['scheduler'] == 'slurm':
                jobs = remote(t, 'slurm_jobs', timeout=120)
                d['ascg_jobs'] = [j for j in jobs if str(j.get('name', '')).startswith('ascg-')]
                other = [j for j in jobs if not str(j.get('name', '')).startswith('ascg-')]
                by_state = {}
                for j in other:
                    by_state[j.get('state')] = by_state.get(j.get('state'), 0) + 1
                d['other_jobs'] = other if all_jobs else {'count': len(other), 'by_state': by_state}
            out[name] = d
        except ToolError as e:
            out[name] = {'error': str(e)}
    return out


# ------------------------------------------------------------------------------------------------ deploy / doctor
CHECKOUT = Path(__file__).resolve().parents[3]          # the checkout this ascg runs from
DEPLOY = ('src/', 'cluster/', 'benchmarks/', 'patches/', 'pyproject.toml')
SITE_FILE = 'cluster/site.local.sh'                     # untracked site settings, copied when present


def _deploy_files():
    def git(*a):
        return subprocess.run(['git', '-C', str(CHECKOUT), *a], capture_output=True, text=True, check=True).stdout
    try:
        names = set(git('ls-files').splitlines()) | set(git('ls-files', '--others', '--exclude-standard').splitlines())
    except (subprocess.CalledProcessError, OSError):
        raise ToolError(f'{CHECKOUT} is not a git checkout: deploy copies its tracked files')
    files = sorted(f for f in names if f.startswith(DEPLOY) and '__pycache__' not in f and (CHECKOUT/f).is_file())
    if (CHECKOUT/SITE_FILE).is_file():
        files.append(SITE_FILE)
    commit = git('rev-parse', '--short', 'HEAD').strip()
    dirty = bool(git('status', '--porcelain', '--untracked-files=no').strip())
    return files, commit, dirty


def _local_digest():
    from ..remote import _code_digest
    return _code_digest()['digest']


@tool('compute', 'deploy', 'Copy this checkout\'s code to a target and verify it (md5 of every file): src, cluster, '
      'benchmarks, patches, pyproject (+ the untracked cluster/site.local.sh). Refuses while ascg jobs run there '
      '(--force overrides); never deletes files on the target. --install-deps pip-installs missing Python dependencies '
      'into the target\'s Python, with numpy and scipy pinned at their installed versions. Afterwards jobs on the target '
      'run exactly this code (run submit refuses a target whose code differs). USER ACTION: agents give the user the '
      'command to run (not served over MCP).', {
          'target': P(required=True), 'install_deps': P('boolean', default=False),
          'force': P('boolean', default=False, help='deploy even while ascg jobs run there')}, mutating=True, mcp=False)
def compute_deploy(install_deps, force, **kw):
    import hashlib
    import tempfile
    t = target(kw['target'])
    files, commit, dirty = _deploy_files()
    try:
        b = remote(t, 'busy', timeout=120)
    except ToolError as e:
        b = {'unknown': str(e)[:200]}                   # older code on the target: no busy verb yet
    if (b.get('processes') or b.get('slurm_jobs')) and not force:
        raise PolicyError(f'{t["name"]}: ascg work is running ({len(b.get("processes", []))} processes, '
                          f'{len(b.get("slurm_jobs", []))} Slurm jobs); deploy after it ends, or --force')
    same = t['transport'] == 'local' and Path(t['root']).resolve() == CHECKOUT
    if not same:
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as fl:
            fl.write('\n'.join(files) + '\n')
        try:
            if t['transport'] == 'ssh':
                subprocess.run(['ssh', '-o', 'BatchMode=yes', t['host'], f'mkdir -p {shlex.quote(t["root"])}'],
                               check=True, capture_output=True, timeout=120)
                r = subprocess.run(['rsync', '-a', f'--files-from={fl.name}', f'{CHECKOUT}/', f'{t["host"]}:{t["root"]}/'],
                                   capture_output=True, text=True, timeout=1800)
            else:
                import shutil
                for f in files:
                    dst = Path(t['root'])/f
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(CHECKOUT/f, dst)
                r = None
        finally:
            os.unlink(fl.name)
        if r is not None and r.returncode != 0:
            raise ToolError(f'rsync to {t["name"]} failed: {r.stderr[-400:]}')
    local = {f: hashlib.md5((CHECKOUT/f).read_bytes()).hexdigest() for f in files}
    there = remote(t, 'manifest', {'files': files}, timeout=600)
    bad = sorted(f for f in files if there.get(f) != local[f])
    if bad:
        raise ToolError(f'{t["name"]}: {len(bad)} files differ after the copy, e.g. {bad[:5]}')
    dep = remote(t, 'deps', {'install': install_deps}, timeout=1800)
    rec = remote(t, 'deploy_record', {'commit': commit, 'dirty': dirty, 'n_files': len(files)}, timeout=120)
    missing = [m for m, v in dep['versions'].items() if v is None]
    return {'target': t['name'], 'files': len(files), 'copied': not same, 'verified': True, 'source_commit': commit,
            'source_dirty': dirty, 'code_digest': rec['code_digest'], 'matches_local': rec['code_digest'] == _local_digest(),
            'python': dep['executable'], 'dependencies': dep['versions'], 'install': dep.get('install'),
            **({'missing': missing, 'note': 'missing dependencies: rerun with --install-deps (MDAnalysis is needed only '
                                            'for bring-your-own references)'} if missing else {})}


@tool('compute', 'doctor', 'Check a target end to end: connection, Python dependencies, code (same as this checkout?), '
      'fastMD build (present, source commit), the engine gate on a free GPU (energies vs numpy, CBT dihedrals, NPT '
      'stability of stiff small molecules; workstations and local targets - Slurm jobs run it at their start), GPUs and '
      'scheduler. Each problem comes with its fix.', {'target': P(required=True),
                                                       'engine': P('boolean', default=True, help='run the engine gate')})
def compute_doctor(engine, **kw):
    from .engine_tools import FASTMD_REF
    t = target(kw['target'])
    checks = []
    def add(name, ok, detail, fix=None, required=True):
        checks.append({'check': name, 'ok': ok, 'detail': detail, **({'fix': fix} if not ok and fix else {}),
                       **({} if required else {'required': False})})
    try:
        ping = remote(t, 'ping', timeout=120)
        add('connection', True, f"{ping['hostname']}, Python {ping['python']}, root {ping['root']}")
    except ToolError as e:
        add('connection', False, str(e)[:300], 'check the ssh alias / root / env line of the target (compute list)')
        return {'target': t['name'], 'ok': False, 'checks': checks}
    try:
        dep = remote(t, 'deps', timeout=300)
        miss = [m for m, v in dep['versions'].items() if v is None and m != 'MDAnalysis']
        add('dependencies', not miss, {k: v for k, v in dep['versions'].items()},
            f'ascg compute deploy --target {t["name"]} --install-deps')
        if dep['versions'].get('MDAnalysis') is None:
            add('MDAnalysis', False, 'missing: bring-your-own references cannot be read here',
                f'ascg compute deploy --target {t["name"]} --install-deps', required=False)
    except ToolError as e:
        add('dependencies', False, f'cannot run the deps check (older ascg code there?): {str(e)[:200]}',
            f'ascg compute deploy --target {t["name"]}')
    try:
        cd = remote(t, 'code_digest', timeout=120)
        local = _local_digest()
        add('code', cd['digest'] == local, {'target': cd['digest'], 'this checkout': local, 'last deploy': cd['deploy']},
            f'ascg compute deploy --target {t["name"]}')
    except ToolError as e:
        add('code', False, f'cannot read the code digest: {str(e)[:200]}', f'ascg compute deploy --target {t["name"]}')
    try:
        fm = remote(t, 'fastmd_info', {'ref': FASTMD_REF}, timeout=120)
        if not fm.get('present'):
            add('fastMD', False, f"FASTMD_BINARY {fm.get('binary')!r} not found", f'build fastMD (ascg engine setup '
                f'--target {t["name"]}) and export FASTMD_BINARY=<build>/fastmd in the target\'s env line (ascg compute '
                'add ... --env)')
        else:
            commit = str(fm.get('commit', ''))
            current = fm.get('includes_ref')
            if current is None:                         # older code on the target, or the commit is unknown there
                current = commit.startswith(FASTMD_REF)
            add('fastMD', True, {k: fm.get(k) for k in ('binary', 'commit', 'branch', 'md5')})
            add('fastMD version', bool(current), f"source {commit or 'unknown'}; ascg expects fastMD main at {FASTMD_REF} "
                'or later' + (' (local changes in the source tree)' if commit.endswith('-dirty') else ''),
                f'rebuild from fastMD {FASTMD_REF} (ascg engine setup --target {t["name"]})', required=False)
    except ToolError as e:
        add('fastMD', False, str(e)[:200], f'ascg compute deploy --target {t["name"]}')
    try:
        gp = remote(t, 'gpus', timeout=120)
        usable = [g for g in gp if t.get('gpus') is None or g['index'] in t['gpus']]
        free = [g['index'] for g in usable if not g['foreign'] and not g['own']]
        add('GPUs', bool(usable) or t['scheduler'] == 'slurm',
            f'{len(gp)} GPU(s) visible here, free of other processes: {free}' if gp else
            'no GPU on this node' + (' (login node: jobs get GPUs)' if t['scheduler'] == 'slurm' else ''),
            'CG fits need an NVIDIA GPU: use a workstation or cluster target with one (and check --gpus)')
    except ToolError as e:
        add('GPUs', False, str(e)[:200]); free = []
    if engine:
        if t['scheduler'] == 'slurm':
            add('engine gate', True, 'runs at the start of every job on its GPU (workflow/ascg_job_*.out)',
                required=False)
        elif not free:
            add('engine gate', False, 'no GPU free of other processes now', 'retry when a GPU is free', required=False)
        else:
            try:
                sc = remote(t, 'selfcheck', {'gpu': free[0], 'dihedrals': True}, timeout=900)
                add('engine gate', sc['ok'], {k: v for k, v in sc.items() if k != 'ok'},
                    'rebuild fastMD (ascg engine setup); an NPT failure means a build without fastMD PR #4')
            except ToolError as e:
                add('engine gate', False, str(e)[:300], 'rebuild fastMD (ascg engine setup)')
    if t['scheduler'] == 'slurm':
        try:
            jobs = remote(t, 'slurm_jobs', timeout=120)
            add('scheduler', True, f'{len(jobs)} job(s) of this account queued or running')
        except ToolError as e:
            add('scheduler', False, str(e)[:200])
    if t.get('aa'):                                     # AA builds and runs (aa build / submit / map)
        try:
            a = remote(t, 'aa_check', {'env': t.get('env', ''), 'aa_env': t['aa'].get('env', ''),
                                       'gmx': t['aa'].get('gmx', 'gmx')}, timeout=180)
            add('AA build (AutoPoly)', a['AutoPoly'], 'AutoPoly v2.0 importable' if a['AutoPoly'] else 'not importable',
                'install AutoPoly v2.0 (with patches/, cluster/aa/apply_autopoly_patch.sh) in the target\'s Python',
                required=False)
            add('AA mapping (MDAnalysis)', a['MDAnalysis'], 'MDAnalysis importable' if a['MDAnalysis'] else 'missing',
                f'ascg compute deploy --target {t["name"]} --install-deps', required=False)
            add('GROMACS', bool(a['gmx']), a['gmx'] or f"{t['aa'].get('gmx', 'gmx')} not found after the AA env line",
                'set aa.env (e.g. "module load gromacs/...") and aa.gmx in the target', required=False)
        except ToolError as e:
            add('AA', False, str(e)[:200], f'ascg compute deploy --target {t["name"]}', required=False)
    problems = [c for c in checks if not c['ok'] and c.get('required', True)]
    return {'target': t['name'], 'ok': not problems, 'checks': checks,
            'problems': [f"{c['check']}: {c.get('fix', '')}" for c in problems]}
