"""catalogue: search the user's system library (ascg.config.load_library; schema ascg-library/1). ascg has no built-in
systems: without a configured library every system is specified by the user."""
import re

from ..config import load_library
from ..registry import P, tool


def _norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()


def catalogue():
    """{key: entry} of the configured library, or {} when none is configured."""
    lib = load_library()
    return lib['systems'] if lib else {}


@tool('catalogue', 'search', 'Find systems of the user\'s system library (ASCG_LIBRARY or ./ascg.library.json) by '
      'name, key or alias. Empty when no library is configured or the system is not in it: then the user specifies '
      'the system and its mapping.', {'query': P(required=True)})
def catalogue_search(query):
    lib = load_library()
    if lib is None:
        return {'query': query, 'library': None, 'matches': [],
                'note': 'no system library configured (ASCG_LIBRARY): specify the system with the user '
                        '(ascg-specify-system)'}
    q = f' {_norm(query)} '
    hits = []
    for k, s in lib['systems'].items():
        names = {_norm(k), _norm(s.get('name', '')), _norm(str(s.get('name', '')).replace('Poly(', 'poly'))}
        names |= {_norm(a) for a in s.get('aliases', [])}
        if any(n and f' {n} ' in q for n in names):
            hits.append(k)
    hits.sort(key=lambda k: -len(str(lib['systems'][k].get('name', ''))))
    return {'query': query, 'library': lib['name'],
            'matches': [{f: lib['systems'][k].get(f) for f in ('key', 'id', 'name', 'system_class',
                                                                 'resolution', 'mapping_status')}
                        | {'has_mapping': bool(lib['systems'][k].get('mapping')),
                           'has_recipe': bool(lib['systems'][k].get('recipe'))} for k in hits],
            'note': None if hits else 'not in the library: specify the system and its mapping with the user'}
