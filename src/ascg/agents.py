"""Agent setup: `ascg init codex` (a person runs it, never an agent).

    ascg init codex --workspace ~/cg-work [--studies DIR] [--profile ascg] [--force] [--dry-run]

Writes a Codex workspace and a Codex profile:
- <workspace>/.agents/skills/: the ascg skills (the same files as the Claude Code plugin);
- <workspace>/AGENTS.md: how the skills map onto the MCP tools;
- <workspace>/.codex/rules/ascg.rules: shell commands that may never leave the sandbox (second layer);
- $CODEX_HOME/<profile>.config.toml: the ascg MCP server, the sandbox and the trusted workspace. The user's own
  config.toml is not touched; `codex -p <profile>` layers the profile on top of it.

Containment: the agent's shell runs in Codex's workspace-write sandbox without network access. It can write only the
workspace, /tmp and $TMPDIR. The ascg tools are served by the MCP server, which Codex starts outside the sandbox, so
they are the only route to studies and compute targets. Everything the server runs or reads as configuration (Python,
the ascg code, the studies, the compute profile, the system library) must therefore lie outside the writable roots:
init refuses otherwise. Settings that would let an agent run its own shell lines outside the sandbox (`compute add`,
`compute deploy`) and human approvals (`amend approve`) are not served.

The evaluation harness (eval/) builds its Codex runs from the same settings (codex_settings / codex_flags).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .registry import ToolError

CHECKOUT = Path(__file__).resolve().parents[2]          # src/ascg/agents.py -> the checkout (when run from one)
MARK = 'written by ascg init codex'
PASS_ENV = ['PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'SSH_AUTH_SOCK', 'FASTMD_BINARY']   # ssh; local fastMD

RULES = '''# {mark} (ascg {version}). Shell commands that may never leave the sandbox: studies and compute targets are
# reached through the ascg MCP tools only.
prefix_rule(pattern = ["ascg"], decision = "forbidden",
            justification = "use the ascg MCP tools (<group>_<action>); amendments, compute and agent setup are the user's")
prefix_rule(pattern = [["ssh", "scp", "sftp", "rsync"]], decision = "forbidden",
            justification = "remote access goes through the ascg tools only")
prefix_rule(pattern = [["sbatch", "scancel", "srun", "salloc", "squeue", "qsub", "qdel", "bkill"]], decision = "forbidden",
            justification = "jobs are submitted, watched and cancelled with the ascg run / study / compute tools")
prefix_rule(pattern = [["kill", "pkill", "killall"]], decision = "forbidden",
            justification = "processes are stopped with the ascg run_cancel tool only")
prefix_rule(pattern = ["git", "push"], decision = "forbidden", justification = "no publishing from agent sessions")
'''

AGENTS_MD = '''<!-- {mark} (ascg {version}); `ascg init codex` rewrites this file -->
# ascg agent (Codex)

You coarse-grain molecular systems with the ascg tools, following the ascg skills in .agents/skills (start with
ascg-study).
- The ascg tools are the MCP tools of the server `ascg`, named <group>_<action> (study_status, aa_import, run_submit,
  diag_digest, ...), with the parameters of `ascg <group> <action> --param value` (dashes become underscores; arrays
  as JSON lists). Wherever a skill says to run `ascg <group> <action> ...`, call that MCP tool instead.
- Your shell runs in a sandbox without network access and cannot write the study files: ssh, cluster and scheduler
  commands are blocked. Everything that touches studies or compute targets goes through the MCP tools.
- Some steps belong to the user and are not served as tools: `ascg compute add`, `ascg compute deploy` (compute
  setup) and `ascg amend approve` (protocol amendments). Give the user the exact command to run in their own terminal,
  then continue with the tools.
- study_watch blocks until an event or its timeout (default 540 s); call it again to keep waiting.
'''


def _version():
    from . import __version__
    return __version__


def writable_roots(workspace):
    """Directories the agent's shell can write in Codex's workspace-write sandbox."""
    roots = [Path(workspace), Path('/tmp'), Path(tempfile.gettempdir())]
    if os.environ.get('TMPDIR'):
        roots.append(Path(os.environ['TMPDIR']))
    out = []
    for r in roots:
        for p in (r.expanduser().absolute(), r.expanduser().resolve()):
            if p not in out:
                out.append(p)
    return out


def check_outside(paths, roots):
    """Refuse when a path the MCP server runs or trusts lies inside a root the agent can write."""
    bad = []
    for what, p in paths.items():
        if not p:
            continue
        for q in {Path(p).expanduser().absolute(), Path(p).expanduser().resolve()}:
            hit = next((r for r in roots if q == r or r in q.parents), None)
            if hit:
                bad.append(f'{what} {p} (inside {hit})')
                break
    if bad:
        raise ToolError('the agent could change these, so they must lie outside its writable roots (the workspace, '
                        '/tmp and $TMPDIR): ' + '; '.join(bad))


def code_dir():
    """The directory that holds the ascg / ascg_core packages this ascg runs from."""
    return Path(__file__).resolve().parents[1]


def skills_dir():
    for c in (os.environ.get('ASCG_PLUGIN'), CHECKOUT/'plugin'/'ascg'):
        if c and (Path(c)/'skills').is_dir():
            return Path(c)/'skills'
    raise ToolError('the ascg skills were not found: run this from a checkout of ascg-agent, or set ASCG_PLUGIN to its '
                    'plugin/ascg directory')


def _defaults(studies=None, compute=None, library=None):
    from .config import library_path, profile_path
    studies = Path(studies or os.environ.get('ASCG_STUDIES') or '~/ascg-studies').expanduser().resolve()
    compute = Path(compute or os.environ.get('ASCG_COMPUTE') or profile_path() or
                   '~/.config/ascg/compute.yaml').expanduser().resolve()
    if library is None:
        lib = library_path()
        library = str(lib) if lib else 'none'
    elif library != 'none':
        library = str(Path(library).expanduser().resolve())
    return studies, compute, library


def codex_settings(workspace, *, python=None, code=None, studies, compute, library, actor='codex',
                   approval='on-request'):
    """Codex configuration (a nested dict) for an ascg workspace: sandbox, trusted workspace, the ascg MCP server."""
    ws = str(Path(workspace).expanduser().resolve())
    return {
        'sandbox_mode': 'workspace-write',
        'approval_policy': approval,
        'sandbox_workspace_write': {'network_access': False},
        'projects': {ws: {'trust_level': 'trusted'}},
        'mcp_servers': {'ascg': {
            'command': str(python or sys.executable), 'args': ['-m', 'ascg.mcp'],
            'env': {'PYTHONPATH': str(code or code_dir()), 'ASCG_STUDIES': str(studies), 'ASCG_COMPUTE': str(compute),
                    'ASCG_LIBRARY': str(library), 'ASCG_ACTOR': actor},
            'env_vars': PASS_ENV, 'startup_timeout_sec': 60, 'tool_timeout_sec': 900,
            'default_tools_approval_mode': 'approve'}}}


def _key(k):
    return k if re.fullmatch(r'[A-Za-z0-9_-]+', k) else json.dumps(k)


def _value(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(_value(x) for x in v) + ']'
    if isinstance(v, dict):
        return '{' + ', '.join(f'{_key(k)} = {_value(x)}' for k, x in v.items()) + '}'
    return json.dumps(str(v))                           # TOML basic string (JSON escapes are valid TOML)


def to_toml(d, header=''):
    """Minimal TOML writer for codex_settings (tables, strings, numbers, booleans, lists)."""
    scalars = {k: v for k, v in d.items() if not isinstance(v, dict)}
    out = ''
    if scalars:
        out += (f'\n[{header}]\n' if header else '') + ''.join(f'{_key(k)} = {_value(v)}\n' for k, v in scalars.items())
    for k, v in d.items():
        if isinstance(v, dict):
            out += to_toml(v, f'{header}.{_key(k)}' if header else _key(k))
    return out


def codex_flags(settings, prefix=''):
    """The same settings as `-c key=value` arguments (for `codex exec --ignore-user-config`)."""
    out = []
    for k, v in settings.items():
        key = f'{prefix}.{_key(k)}' if prefix else _key(k)
        if isinstance(v, dict) and v:
            out += codex_flags(v, key)
        else:
            out += ['-c', f'{key}={_value(v)}']
    return out


def write_workspace(workspace, *, skills=None, force=False):
    """Skills, AGENTS.md and rules in a Codex workspace; returns the files written."""
    ws = Path(workspace).expanduser().resolve()
    version = _version()
    agents_md = ws/'AGENTS.md'
    if agents_md.exists() and MARK not in agents_md.read_text()[:300] and not force:
        raise ToolError(f'{agents_md} exists and was not written by ascg: move it, or rerun with --force')
    src = Path(skills or skills_dir())
    dst = ws/'.agents'/'skills'
    dst.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in src.iterdir() if (p/'SKILL.md').is_file())
    for n in names:                                     # only ascg's skills are replaced; others stay
        if (dst/n).exists():
            shutil.rmtree(dst/n)
        shutil.copytree(src/n, dst/n, ignore=shutil.ignore_patterns('__pycache__'))
    (dst/'.ascg').write_text(json.dumps({'version': version, 'skills': names, 'written': time.strftime('%Y-%m-%d %H:%M:%S'),
                                         'source': str(src)}, indent=1) + '\n')
    agents_md.write_text(AGENTS_MD.format(mark=MARK, version=version))
    rules = ws/'.codex'/'rules'/'ascg.rules'
    rules.parent.mkdir(parents=True, exist_ok=True)
    rules.write_text(RULES.format(mark=MARK, version=version))
    return [str(dst), str(agents_md), str(rules)]


def init_codex(workspace, *, studies=None, profile='ascg', codex_home=None, force=False, dry_run=False):
    ws = Path(workspace).expanduser().resolve()
    studies, compute, library = _defaults(studies)
    home = Path(codex_home or os.environ.get('CODEX_HOME') or '~/.codex').expanduser().resolve()
    prof = home/f'{profile}.config.toml'
    roots = writable_roots(ws)
    check_outside({'Python': Path(sys.executable).resolve(), 'ascg code': code_dir(), 'studies': studies,
                   'compute profile': compute, 'system library': None if library == 'none' else library,
                   'Codex profile': prof}, roots)
    if ws in (code_dir(), CHECKOUT) or code_dir() in ws.parents:
        raise ToolError(f'choose a workspace outside the ascg code ({code_dir()})')
    settings = codex_settings(ws, studies=studies, compute=compute, library=library)
    text = (f'# {MARK} (ascg {_version()}, {time.strftime("%Y-%m-%d %H:%M:%S")}). Start Codex with\n'
            f'#   codex -p {profile} -C {ws}\n# Rerun `ascg init codex` to update it; do not edit by hand.\n\n'
            + to_toml(settings))
    if prof.exists() and MARK not in prof.read_text()[:300] and not force:
        raise ToolError(f'{prof} exists and was not written by ascg: choose another --profile, or --force')
    deps = subprocess.run([sys.executable, '-c', 'import numpy, scipy, rdkit, networkx'], capture_output=True)
    out = {'workspace': str(ws), 'profile': str(prof), 'python': sys.executable, 'ascg_code': str(code_dir()),
           'studies': str(studies), 'compute_profile': str(compute) + ('' if compute.exists() else ' (not created yet)'),
           'system_library': library, 'version': _version(),
           'start': f'codex -p {profile} -C {ws}',
           'headless': f'codex exec -p {profile} -C {ws} "<your request>"',
           'not_served_to_the_agent': ['compute add', 'compute deploy', 'amend approve'],
           **({} if deps.returncode == 0 else {'warning': f'{sys.executable} lacks numpy / scipy / rdkit / networkx: '
                                                          'set ASCG_PYTHON to a Python that has them and rerun'})}
    if dry_run:
        return {**out, 'dry_run': True, 'profile_text': text}
    ws.mkdir(parents=True, exist_ok=True)
    studies.mkdir(parents=True, exist_ok=True)
    out['files'] = write_workspace(ws, force=force)
    prof.parent.mkdir(parents=True, exist_ok=True)
    prof.write_text(text)
    out['files'].append(str(prof))
    return out


USAGE = ('usage: ascg init codex --workspace DIR [--studies DIR] [--profile NAME] [--codex-home DIR] [--force] '
         '[--dry-run]\n'
         '  Sets up a Codex workspace for ascg (skills, AGENTS.md, rules) and a Codex profile with the ascg MCP server;\n'
         '  then start Codex with: codex -p NAME -C DIR. Run it yourself: agents are not allowed to.')


def main(argv):
    if not argv or argv[0] in ('-h', '--help', 'help') or argv[0] != 'codex':
        print(USAGE)
        return 0 if argv and argv[0] in ('-h', '--help', 'help') else 1
    from .cli import _parse
    try:
        rest, pos = argv[1:], []
        while rest and not rest[0].startswith('--'):       # `ascg init codex DIR`
            pos.append(rest.pop(0))
        args = _parse(rest)
        if pos:
            args.setdefault('workspace', ' '.join(pos))
        if args.pop('help', False):
            print(USAGE)
            return 0
        if not args.get('workspace'):
            raise ToolError('--workspace DIR: the directory Codex works in (created if missing)')
        unknown = set(args) - {'workspace', 'studies', 'profile', 'codex_home', 'force', 'dry_run'}
        if unknown:
            raise ToolError(f'unknown option(s): {sorted(unknown)}')
        res = init_codex(args['workspace'], studies=args.get('studies'), profile=args.get('profile') or 'ascg',
                         codex_home=args.get('codex_home'), force=bool(args.get('force')),
                         dry_run=bool(args.get('dry_run')))
        print(json.dumps({'ok': True, 'tool': 'init codex', 'result': res}, indent=2))
        return 0
    except ToolError as e:
        print(json.dumps({'ok': False, 'tool': 'init codex', 'error': str(e), 'usage': USAGE}, indent=2))
        return 1
