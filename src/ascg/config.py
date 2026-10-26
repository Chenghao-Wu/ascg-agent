"""Locations and the user's compute profiles. Nothing site-specific lives in the package: targets come from
ASCG_COMPUTE, ./ascg.compute.yaml|json or ~/.config/ascg/compute.yaml|json."""
from __future__ import annotations

import json
import os
from pathlib import Path


def studies_root():
    return Path(os.environ.get('ASCG_STUDIES', Path.cwd()/'ascg-studies')).expanduser().resolve()


def _load(path):
    text = Path(path).read_text()
    if str(path).endswith(('.yaml', '.yml')):
        try:
            import yaml
        except ImportError as e:            # PyYAML is optional; JSON profiles need nothing
            raise RuntimeError(f'{path}: install PyYAML or use a .json profile') from e
        return yaml.safe_load(text) or {}
    return json.loads(text)


def profile_path():
    if os.environ.get('ASCG_COMPUTE'):                  # explicit: used even before it exists, never falls through
        return Path(os.environ['ASCG_COMPUTE']).expanduser()
    cands = [Path.cwd()/f'ascg.compute.{e}' for e in ('yaml', 'yml', 'json')]
    cands += [Path('~/.config/ascg').expanduser()/f'compute.{e}' for e in ('yaml', 'yml', 'json')]
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


def load_profiles():
    p = profile_path()
    if p is None or not p.exists():
        return {}, None
    data = _load(p)
    return dict(data.get('targets', {})), p


def save_profiles(targets, path=None):
    path = Path(path or profile_path() or Path('~/.config/ascg/compute.yaml').expanduser())
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {'targets': targets}
    if str(path).endswith(('.yaml', '.yml')):
        import yaml
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    else:
        path.write_text(json.dumps(data, indent=2) + '\n')
    return path


def library_path():
    """The user's system library (schema ascg-library/1): ASCG_LIBRARY, ./ascg.library.json or
    ~/.config/ascg/library.json. None when not configured, or with ASCG_LIBRARY=none (e.g. for evaluation runs):
    ascg has no built-in systems."""
    if os.environ.get('ASCG_LIBRARY', '').strip().lower() == 'none':
        return None
    if os.environ.get('ASCG_LIBRARY'):                  # explicit: never falls through to other files
        p = Path(os.environ['ASCG_LIBRARY']).expanduser()
        return p.resolve() if p.is_file() else None
    cands = [Path.cwd()/'ascg.library.json', Path.home()/'.config'/'ascg'/'library.json']
    for c in cands:
        if c and Path(c).expanduser().is_file():
            return Path(c).expanduser().resolve()
    return None


def load_library():
    """{'path', 'name', 'systems': {key: entry}} or None. Entry fields: key, name, system_class, aliases, resolution,
    mapping_status, mapping (with 'format'), recipe {'file', 'task'} and force_field_override (paths relative to the
    library file)."""
    import json
    p = library_path()
    if p is None:
        return None
    d = json.loads(p.read_text())
    if d.get('schema') != 'ascg-library/1':
        raise ValueError(f'{p}: not an ascg-library/1 file')
    return {'path': p, 'name': d.get('name', p.name), 'systems': {s['key']: s for s in d['systems']}}
