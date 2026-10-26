"""Study registry: the single source of truth for one study (tasks, where they run, job ids / PIDs, state cache,
outcomes, protocol lock, amendments, events, call log). Plain JSON files under <studies_root>/<study>/."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .config import studies_root
from .registry import ToolError


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


class Study:
    def __init__(self, name, root=None):
        if not name or '/' in name or name.startswith('.'):
            raise ToolError(f'invalid study name {name!r}')
        self.name = name
        self.dir = Path(root or studies_root())/name
        self.reg_file = self.dir/'registry.json'

    # -- persistence
    def exists(self):
        return self.reg_file.exists()

    def load(self):
        if not self.exists():
            raise ToolError(f'study {self.name!r} not found under {self.dir.parent} (ascg study init)')
        return json.loads(self.reg_file.read_text())

    def save(self, reg):
        tmp = self.reg_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(reg, indent=2) + '\n')
        os.replace(tmp, self.reg_file)

    def init(self, protocol, description=''):
        if self.exists():
            raise ToolError(f'study {self.name!r} exists')
        self.dir.mkdir(parents=True)
        (self.dir/'amendments').mkdir()
        (self.dir/'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        reg = {'study': self.name, 'created': now(), 'description': description,
               'protocol': {'file': 'protocol.json', 'hash': canonical_hash(protocol), 'version': protocol.get('version')},
               'tasks': {}, 'event_seq': 0}
        self.save(reg)
        self.event(reg, 'study_created', None, {'protocol_hash': reg['protocol']['hash']})
        return reg

    def protocol(self):
        return json.loads((self.dir/'protocol.json').read_text())

    def check_protocol_lock(self, reg):
        h = canonical_hash(self.protocol())
        if h != reg['protocol']['hash']:
            approved = [a for a in self.amendments() if a.get('status') == 'approved' and a.get('protocol_hash') == h]
            if not approved:
                raise ToolError(f'protocol.json changed (hash {h} != locked {reg["protocol"]["hash"]}) without an '
                                'approved amendment; rule changes go through `ascg amend propose` + human approval')
        return h

    # -- events and calls
    def event(self, reg, kind, task, data=None):
        reg['event_seq'] = reg.get('event_seq', 0) + 1
        ev = {'seq': reg['event_seq'], 'time': now(), 'event': kind, 'task': task, **(data or {})}
        with (self.dir/'events.jsonl').open('a') as f:
            f.write(json.dumps(ev) + '\n')
        self.save(reg)
        return ev

    def events(self, since=0):
        f = self.dir/'events.jsonl'
        if not f.exists():
            return []
        return [e for e in (json.loads(x) for x in f.read_text().splitlines() if x.strip()) if e['seq'] > since]

    def log_call(self, rec):
        if self.dir.exists():
            with (self.dir/'calls.jsonl').open('a') as f:
                f.write(json.dumps(rec, default=str) + '\n')

    # -- mappings (proposed before any reference exists; approved by the user)
    def mapping_file(self, key):
        return self.dir/'mappings'/f'{key}.json'

    def mapping(self, key):
        f = self.mapping_file(key)
        return json.loads(f.read_text()) if f.exists() else None

    # -- systems (one SMILES per species: the AA build and, with bead numbers, the mapping; approved by the user)
    def system_file(self, key):
        return self.dir/'systems'/f'{key}.json'

    def system(self, key):
        f = self.system_file(key)
        return json.loads(f.read_text()) if f.exists() else None

    # -- amendments
    def amendments(self):
        d = self.dir/'amendments'
        return [json.loads(p.read_text()) for p in sorted(d.glob('*.json'))] if d.exists() else []

    def task(self, reg, key):
        if key not in reg['tasks']:
            raise ToolError(f'task {key!r} not in study {self.name!r} (have: {", ".join(reg["tasks"]) or "none"})')
        return reg['tasks'][key]
