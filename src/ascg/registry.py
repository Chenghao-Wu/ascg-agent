"""Tool registry: every tool is a function with a JSON-Schema-like parameter spec, a group, and a mutating flag.
The CLI, the call log and (if ever needed) an MCP adapter are generated from this one registry."""
from __future__ import annotations

from dataclasses import dataclass, field

TOOLS: dict = {}


class PolicyError(RuntimeError):
    """A tool refused an action by policy (fail closed). Exit status 2."""


class ToolError(RuntimeError):
    """A tool could not complete (bad input, missing data, remote failure). Exit status 1."""


@dataclass
class Tool:
    group: str
    action: str
    summary: str
    params: dict = field(default_factory=dict)      # name -> {'type', 'required', 'default', 'help', 'enum'}
    mutating: bool = False
    fn: object = None
    mcp: bool = True                                # served over MCP (False: changes the user's setup or needs a person)

    @property
    def name(self):
        return f'{self.group} {self.action}'


def tool(group, action, summary, params=None, mutating=False, mcp=True):
    def deco(fn):
        TOOLS[(group, action)] = Tool(group, action, summary, params or {}, mutating, fn, mcp)
        return fn
    return deco


def P(type_='string', required=False, default=None, help='', enum=None):
    """Parameter spec helper."""
    d = {'type': type_, 'required': required, 'help': help}
    if default is not None:
        d['default'] = default
    if enum:
        d['enum'] = list(enum)
    return d


def schema(t: Tool):
    """JSON Schema of a tool's parameters (for documentation / an MCP adapter)."""
    props, req = {}, []
    for k, s in t.params.items():
        props[k] = {kk: vv for kk, vv in s.items() if kk in ('type', 'default', 'enum')}
        props[k]['description'] = s.get('help', '')
        if s.get('required'):
            req.append(k)
    return {'type': 'object', 'properties': props, 'required': req}


def usage(t: Tool):
    """One-line usage of a tool, shown with parameter errors (saves a --help round trip)."""
    req = [f'--{k} <{s.get("type", "string")}>' for k, s in t.params.items() if s.get('required')]
    opt = [f'[--{k}{"" if s.get("type") == "boolean" else " <" + s.get("type", "string") + ">"}]'
           for k, s in t.params.items() if not s.get('required')]
    return ' '.join(['ascg', t.group, t.action, *req, *opt])


def primary(t: Tool):
    """The parameter a positional value fills: the tool's only required parameter (e.g. catalogue search acetone)."""
    req = [k for k, s in t.params.items() if s.get('required')]
    return req[0] if len(req) == 1 else None


def _alias(t: Tool, args: dict):
    """--target for --targets, --iteration for --iterations and the reverse, when only one of them exists."""
    out = {}
    for k, v in args.items():
        if k not in t.params:
            for alt in (k + 's', k[:-1] if k.endswith('s') else None):
                if alt and alt in t.params and alt not in args:
                    k = alt
                    break
        out[k] = v
    return out


def _as_list(v):
    """An array parameter from JSON ("[35, 40]"), "35,40", "35 40", or tokens that split such a value ('[35', '40]')."""
    import json
    if isinstance(v, (list, tuple)):
        v = ' '.join(str(x) for x in v)
    v = str(v).strip()
    if v.startswith('['):
        try:
            out = json.loads(v)
            if isinstance(out, list):
                return out
        except ValueError:
            pass
    return [x for x in v.strip('[]').replace(',', ' ').replace('[', ' ').replace(']', ' ').split() if x]


def validate(t: Tool, args: dict):
    args = _alias(t, args)
    out = {}
    for k, s in t.params.items():
        if k in args and args[k] is not None:
            v = args[k]
            typ = s.get('type', 'string')
            try:
                if typ == 'integer':
                    v = int(v)
                elif typ == 'number':
                    v = float(v)
                elif typ == 'boolean':
                    v = v if isinstance(v, bool) else str(v).lower() in ('1', 'true', 'yes', 'on')
                elif typ == 'array':
                    v = _as_list(v)
            except ValueError:
                raise ToolError(f'parameter --{k}: expected {typ}, got {v!r}')
            if s.get('enum') and v not in s['enum']:
                raise ToolError(f'parameter --{k}: must be one of {s["enum"]}, got {v!r}')
            out[k] = v
        elif s.get('required'):
            raise ToolError(f'missing required parameter --{k}')
        elif 'default' in s:
            out[k] = s['default']
    unknown = set(args) - set(t.params)
    if unknown:
        raise ToolError(f'unknown parameter(s): {", ".join(sorted(unknown))}')
    return out
