"""ascg command line: `ascg <group> <action> [--param value ...] [--json '{...}']` -> one JSON document on stdout.
Exit status: 0 ok, 1 tool error, 2 refused by policy. `ascg tools` lists every tool with its parameters."""
from __future__ import annotations

import json
import os
import sys
import time

from . import tools  # noqa: F401  (registers the tools)
from .registry import TOOLS, PolicyError, ToolError, primary, schema, usage, validate
from .study import Study

USAGE = ('usage: ascg <group> <action> [--param value ...] [--json JSON] | ascg tools [--schema]\n'
         '       ascg init codex --workspace DIR      (set up Codex for ascg; run it yourself)\n'
         '       (other first words are passed to the project CLI: ascg build-opls ..., ascg map-aa ...)')


def _legacy_commands():
    from ascg_core.cli import COMMANDS                 # light: argparse / json only
    return set(COMMANDS)


def _parse(argv):
    args, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith('--'):
            raise ToolError(f'unexpected argument {a!r}')
        k = a[2:].replace('-', '_')
        if '=' in k:
            k, v = k.split('=', 1)
            i += 1
        elif i + 1 < len(argv) and not argv[i + 1].startswith('--'):
            j = i + 1                                  # several values -> one space-joined value (array params)
            while j + 1 < len(argv) and not argv[j + 1].startswith('--'):
                j += 1
            v = argv[i + 1] if j == i + 1 else ' '.join(argv[i + 1:j + 1])
            i = j + 1
        else:
            v, i = True, i + 1
            if k.startswith('no_'):                     # --no-engine: the boolean parameter engine = false
                k, v = k[3:], False
        if k == 'json':
            args.update(json.loads(v))
        else:
            args[k] = v
    return args


def _log(t, args, ok, result=None, error=None, kind=None, t0=None):
    """Every call of a study tool goes to the study's calls.jsonl; with ASCG_CALL_LOG set (evaluation runs) every call,
    with its result, also goes to that file."""
    rec = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'tool': t.name, 'args': args, 'ok': ok,
           'seconds': round(time.time() - t0, 2) if t0 else None, 'actor': os.environ.get('ASCG_ACTOR', 'cli'),
           'mutating': t.mutating}
    if error:
        rec.update(error=str(error)[:500], kind=kind)
    if os.environ.get('ASCG_CALL_LOG'):
        try:
            res = json.dumps(result, default=str)
            full = {**rec, 'result': json.loads(res) if len(res) < 200_000 else None, 'result_chars': len(res)}
            with open(os.environ['ASCG_CALL_LOG'], 'a') as f:
                f.write(json.dumps(full, default=str) + '\n')
        except Exception:
            pass
    if not args.get('study'):
        return
    try:
        Study(args['study']).log_call(rec)
    except Exception:
        pass


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help', 'help'):
        print(USAGE)
        return 0
    if argv[0] == 'init':                                   # agent setup: a person's command, not a tool
        from .agents import main as init
        return init(argv[1:])
    if argv[0] == 'tools':
        full = '--schema' in argv
        out = {f'{g} {a}': ({'summary': t.summary, 'mutating': t.mutating, 'parameters': schema(t)} if full
                            else t.summary) for (g, a), t in sorted(TOOLS.items())}
        print(json.dumps(out, indent=2))
        return 0
    groups = sorted({g for g, _ in TOOLS})
    if argv[0] not in groups and argv[0] in _legacy_commands():   # project CLI: build-opls, map-aa, ...
        from ascg_core.cli import main as legacy
        return legacy(argv) or 0
    if len(argv) < 2 or (argv[0], argv[1]) not in TOOLS:
        print(json.dumps({'ok': False, 'error': f'unknown tool {" ".join(argv[:2])!r}; groups: {groups}; '
                          'run `ascg tools`'}))
        return 1
    t = TOOLS[(argv[0], argv[1])]
    t0 = time.time()
    phase = 'arguments'
    try:
        rest = argv[2:]
        pos = []
        while rest and not rest[0].startswith('--'):          # a positional value fills the only required parameter
            pos.append(rest.pop(0))
        raw = _parse(rest)
        if raw.pop('help', False):
            print(json.dumps({'tool': t.name, 'summary': t.summary, 'usage': usage(t), 'parameters': schema(t)},
                             indent=2))
            return 0
        if pos:
            k = primary(t)
            if k is None or k in raw:
                raise ToolError(f'unexpected argument {pos[0]!r}')
            raw[k] = ' '.join(pos)
        args = validate(t, raw)
        phase = 'run'
        res = t.fn(**args)
        _log(t, args, True, result=res, t0=t0)
        print(json.dumps({'ok': True, 'tool': t.name, 'result': res}, indent=2, default=str))
        return 0
    except PolicyError as e:
        _log(t, locals().get('args') or locals().get('raw') or {}, False, error=e, kind='policy', t0=t0)
        print(json.dumps({'ok': False, 'tool': t.name, 'refused_by_policy': True, 'error': str(e)}, indent=2))
        return 2
    except (ToolError, ValueError, KeyError, FileNotFoundError) as e:
        _log(t, locals().get('args') or locals().get('raw') or {}, False, error=e, kind='error', t0=t0)
        out = {'ok': False, 'tool': t.name, 'error': f'{type(e).__name__}: {e}'}
        if phase == 'arguments':
            out['usage'] = usage(t)
        print(json.dumps(out, indent=2))
        return 1
