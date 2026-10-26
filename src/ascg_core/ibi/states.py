"""State points of a fit (design Revision 6, state-point contract): user-approved (temperature, pressure) pairs.

A state is (T in K, P in atm; default 1 atm). A fit uses N >= 1 states, and every one needs its own mapped AA reference
at that (T, P): targets_T<label>.npz, start_T<label>.npz, mapped_T<label>.npz and an entry in summary.json 'states'.

Labels: a 1 atm state is labelled by its temperature alone ('400', '298.15'), as in every reference made so far, so
'T400' files and '400' keys are unchanged. Other pressures add the pressure in atm: '400_P1000', '450_P0.5'.

Spec strings (CLI, ascg): '400,500' (1 atm), '400@1000' (atm), '400@1013.25bar', or the labels themselves
('400_P1000', 'T400_P1000').

Model (ibi.tables.solve_ab): the CG potential depends on temperature only, U(x, T) = A(x) + B(x) T. Pressure enters
the sampling: the NPT checks and held-out validation run at each state's pressure, and the pressure correction drives
each state's NVT pressure to its own target. States that share a temperature share one table set, fitted as their
weighted compromise. One distinct temperature gives B = 0 (one table set). Two distinct temperatures over two states
make A and B exact. Otherwise A and B are a weighted least-squares fit, and a state may keep a residual that the
per-state gates report. Weights default to equal; they also weight the shared-model update (sum_s w_s ...).
"""
from __future__ import annotations

import json
import numbers
import re
from functools import total_ordering
from pathlib import Path

BAR_PER_ATM = 1.01325
T_MIN_K, T_MAX_K = 1.0, 5000.0
P_MIN_ATM, P_MAX_ATM = 1e-6, 1e5


def norm(x):
    """400.0 -> 400; 298.150 -> 298.15."""
    t = round(float(x), 3)
    return int(t) if t == int(t) else t


@total_ordering
class State:
    """One state point: T (K), P (atm). Hashes and compares like its temperature at 1 atm, so 400 and State(400) are
    the same dict key; str() / format() give the label."""
    __slots__ = ('T', 'P')

    def __init__(self, T, P=1.0):
        self.T, self.P = norm(T), norm(P)
        if not T_MIN_K <= self.T <= T_MAX_K:
            raise ValueError(f'non-physical temperature {T} K')
        if not P_MIN_ATM <= self.P <= P_MAX_ATM:
            raise ValueError(f'pressure {P} atm outside ({P_MIN_ATM}, {P_MAX_ATM}]')

    @property
    def at_1atm(self):
        return abs(float(self.P) - 1.0) < 1e-9

    @property
    def label(self):
        return str(self.T) if self.at_1atm else f'{self.T}_P{self.P}'

    @property
    def pressure_bar(self):
        return float(self.P) * BAR_PER_ATM

    def pretty(self):
        return f'{self.T} K' if self.at_1atm else f'{self.T} K, {self.P} atm'

    def key(self):
        return (float(self.T), float(self.P))

    def __float__(self):
        return float(self.T)

    def __str__(self):
        return self.label

    def __format__(self, spec):
        return format(self.label, spec)

    def __repr__(self):
        return f'State({self.T}, {self.P})'

    def __hash__(self):
        return hash(self.T) if self.at_1atm else hash((self.T, self.P))

    def __eq__(self, other):
        o = _coerce(other)
        return NotImplemented if o is None else self.key() == o.key()

    def __lt__(self, other):
        o = _coerce(other)
        return NotImplemented if o is None else self.key() < o.key()

    def to_dict(self):
        return {'label': self.label, 'temperature_K': self.T, 'pressure_atm': self.P,
                'pressure_bar': round(self.pressure_bar, 6)}


def _coerce(x):
    if isinstance(x, State):
        return x
    if isinstance(x, numbers.Real) and not isinstance(x, bool):
        try:
            return State(x)
        except ValueError:
            return None
    return None


_ITEM = re.compile(r'^T?(?P<T>[0-9.]+)\s*K?(?:(?:@|_P)\s*(?P<P>[0-9.eE+-]+)\s*(?P<u>atm|bar)?)?$', re.I)


def parse_one(x):
    """A State from a State, a number (K, 1 atm), {'temperature_K', 'pressure_atm'} or a spec string."""
    if isinstance(x, State):
        return x
    if isinstance(x, dict):
        return State(x.get('temperature_K', x.get('T')), x.get('pressure_atm', x.get('P', 1.0)))
    if isinstance(x, numbers.Real) and not isinstance(x, bool):
        return State(x)
    m = _ITEM.match(str(x).strip())
    if not m:
        raise ValueError(f'cannot read state point {x!r} (e.g. 400, 400@1000, 400@1013.25bar, 400_P1000)')
    P = float(m['P']) if m['P'] else 1.0
    if (m['u'] or 'atm').lower() == 'bar':
        P /= BAR_PER_ATM
    return State(float(m['T']), P)


def parse(spec):
    """'400,500' | '400@1000 500' | [400, '450_P2'] -> ascending tuple of States; rejects empty, duplicate or
    non-physical states."""
    if isinstance(spec, str):
        items = [x for x in spec.replace(',', ' ').split() if x]
    elif isinstance(spec, (State, numbers.Real, dict)):
        items = [spec]
    else:
        items = list(spec)
    if not items:
        raise ValueError('no state points given')
    states = [parse_one(x) for x in items]
    if len(set(states)) != len(states):
        raise ValueError(f'duplicate state points in {[str(s) for s in states]}')
    return tuple(sorted(states))


BENCHMARK_STATES = parse('400,500')      # the benchmark polymers' approved pair (1 atm); the states of runs made
BENCHMARK_TEMPERATURES_K = (400, 500)    # before states were recorded. Never a default for new studies.


def label(x):
    """File / key label of a state (or of a temperature at 1 atm): 'T' + label is the reference file suffix."""
    return parse_one(x).label


def labels(states):
    return [s.label for s in states]


def temperatures(states):
    return sorted({s.T for s in states})


def of_reference(ref_dir):
    """The states a mapped reference holds (summary.json 'states' labels)."""
    return parse(list(json.loads((Path(ref_dir)/'summary.json').read_text())['states']))


def of_run(state):
    """The states of an IBI run (workflow/state.json settings 'states'; 'temperatures' for earlier runs made at
    1 atm; the benchmark pair for runs made before either was recorded)."""
    s = state.get('settings') or {}
    if s.get('states'):
        return parse(s['states'])
    if s.get('temperatures'):
        return parse(s['temperatures'])
    return BENCHMARK_STATES


def of_task(task_dir):
    f = Path(task_dir)/'workflow'/'state.json'
    return of_run(json.loads(f.read_text())) if f.exists() else None


def missing_files(ref_dir, states):
    """Reference files absent for the given states."""
    ref = Path(ref_dir)
    return [f'{p}_T{label(s)}.npz' for s in states for p in ('targets', 'start', 'mapped')
            if not (ref/f'{p}_T{label(s)}.npz').exists()]


def weights(states, spec=None, w400=None):
    """{State: w} summing to 1. spec: None (equal), 'w1,w2,...' in the order of the states, or 'label:w,...'
    ('400:1,400_P1000:2'); w400: the older two-state option (only for the 400 / 500 K pair at 1 atm)."""
    states = tuple(parse(states))
    if w400 is not None:
        if states != BENCHMARK_STATES:
            raise ValueError('--w400 applies to the 400 / 500 K pair only; use --weights')
        if spec:
            raise ValueError('give --weights or --w400, not both')
        w = {states[0]: float(w400), states[1]: 1.0 - float(w400)}
    elif not spec:
        w = {s: 1.0 / len(states) for s in states}
    else:
        parts = [x for x in str(spec).replace(' ', ',').split(',') if x]
        if all(':' in x for x in parts):
            w = {parse_one(k): float(v) for k, v in (x.rsplit(':', 1) for x in parts)}
            if set(w) != set(states):
                raise ValueError(f'--weights states {labels(sorted(w))} differ from the fit states {labels(states)}')
        else:
            if len(parts) != len(states):
                raise ValueError(f'--weights has {len(parts)} values for {len(states)} states')
            w = dict(zip(states, (float(x) for x in parts)))
    if any(v < 0 for v in w.values()) or sum(w.values()) <= 0:
        raise ValueError(f'state weights must be >= 0 with a positive sum: { {str(k): v for k, v in w.items()} }')
    tot = sum(w.values())
    return {s: w[s] / tot for s in states}


def describe(states):
    """'400 K and 500 K' / '450 K' / '400 K; 400 K, 1000 atm and 500 K'."""
    t = [parse_one(s).pretty() for s in states]
    if len(t) == 1:
        return t[0]
    sep = '; ' if any(',' in x for x in t) else ', '
    return sep.join(t[:-1]) + ' and ' + t[-1]


def jsonable(obj):
    """obj with State keys / values turned into labels (json cannot serialise them)."""
    if isinstance(obj, dict):
        return {(str(k) if isinstance(k, State) else k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return str(obj) if isinstance(obj, State) else obj
