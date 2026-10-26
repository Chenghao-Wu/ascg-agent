"""MCP server of the ascg tools (stdio, JSON-RPC 2.0, one JSON message per line), generated from the tool registry.

    python -m ascg.mcp            (PYTHONPATH must include this repository's src)

For agent harnesses whose shell runs in a sandbox without network access (Codex): the harness starts this server
outside the sandbox, so the ascg tools reach the compute targets while the agent's own commands cannot. Tool names
are <group>_<action> (e.g. study_status); arguments are the CLI parameters with underscores. Results are the same JSON
as the CLI's. Tools registered with mcp=False are not served: `amend approve` (a person at a terminal), `compute add`
and `compute deploy` (they change the user's compute setup and run its shell lines outside the sandbox; the user runs
them).
"""
from __future__ import annotations

import json
import sys
import time

from . import tools  # noqa: F401  (registers the tools)
from .cli import _log
from .registry import TOOLS, PolicyError, ToolError, schema, usage, validate

PROTOCOL_VERSION = '2025-06-18'


def _tools():
    return {f'{g}_{a}'.replace('-', '_'): t for (g, a), t in sorted(TOOLS.items()) if t.mcp}


def _call(t, arguments):
    t0 = time.time()
    try:
        args = validate(t, {k.replace('-', '_'): v for k, v in (arguments or {}).items()})
    except (ToolError, ValueError) as e:
        return {'ok': False, 'tool': t.name, 'error': f'{type(e).__name__}: {e}', 'usage': usage(t)}, True
    try:
        res = t.fn(**args)
        _log(t, args, True, result=res, t0=t0)
        return {'ok': True, 'tool': t.name, 'result': res}, False
    except PolicyError as e:
        _log(t, args, False, error=e, kind='policy', t0=t0)
        return {'ok': False, 'tool': t.name, 'refused_by_policy': True, 'error': str(e)}, True
    except (ToolError, ValueError, KeyError, FileNotFoundError) as e:
        _log(t, args, False, error=e, kind='error', t0=t0)
        return {'ok': False, 'tool': t.name, 'error': f'{type(e).__name__}: {e}'}, True


def handle(msg):
    """One JSON-RPC message -> the response dict (None for notifications)."""
    method, mid = msg.get('method'), msg.get('id')
    if mid is None:                                     # notifications (initialized, cancelled): no reply
        return None
    if method == 'initialize':
        result = {'protocolVersion': (msg.get('params') or {}).get('protocolVersion') or PROTOCOL_VERSION,
                  'capabilities': {'tools': {'listChanged': False}},
                  'serverInfo': {'name': 'ascg', 'version': __import__('ascg').__version__},
                  'instructions': 'The ascg tools for coarse-graining studies: the same tools and parameters as the '
                                  '`ascg <group> <action> --param value` CLI the ascg skills describe, named '
                                  '<group>_<action>. Use them for everything the skills do with `ascg`.'}
    elif method == 'ping':
        result = {}
    elif method == 'tools/list':
        result = {'tools': [{'name': n, 'description': t.summary, 'inputSchema': schema(t),
                             'annotations': {'readOnlyHint': not t.mutating}} for n, t in _tools().items()]}
    elif method == 'tools/call':
        p = msg.get('params') or {}
        t = _tools().get(str(p.get('name') or ''))
        if t is None:
            return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32602, 'message': f'unknown tool {p.get("name")!r}'}}
        out, is_error = _call(t, p.get('arguments'))
        result = {'content': [{'type': 'text', 'text': json.dumps(out, indent=1, default=str)}], 'isError': is_error}
    else:
        return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32601, 'message': f'method {method!r} not supported'}}
    return {'jsonrpc': '2.0', 'id': mid, 'result': result}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            out = {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'parse error'}}
        else:
            try:
                out = handle(msg)
            except Exception as e:                      # never kill the server on one bad call
                out = {'jsonrpc': '2.0', 'id': msg.get('id'),
                       'error': {'code': -32603, 'message': f'{type(e).__name__}: {e}'}}
        if out is not None:
            sys.stdout.write(json.dumps(out, default=str) + '\n')
            sys.stdout.flush()


if __name__ == '__main__':
    main()
