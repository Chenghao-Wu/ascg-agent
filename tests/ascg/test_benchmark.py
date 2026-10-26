"""The benchmark is one set of 40 systems: the system library and the recipes agree, every mapping applies to its
recipe, and the README lists every system."""
import json
from collections import Counter
from pathlib import Path

import pytest

B = Path(__file__).resolve().parents[2]/'benchmarks'
LIB = json.loads((B/'ascg_library.json').read_text())
REC = json.loads((B/'aa_recipes.json').read_text())
SYSTEMS = LIB['systems']


def test_one_benchmark_of_40_systems():
    assert LIB['schema'] == 'ascg-library/1' and len(SYSTEMS) == 40
    assert len({s['key'] for s in SYSTEMS}) == 40 and len({s['id'] for s in SYSTEMS}) == 40
    assert Counter(s['system_class'] for s in SYSTEMS) == {'polymer': 33, 'liquid': 6, 'liquid_mixture': 1}
    assert [s['id'] for s in SYSTEMS] == [f'B{i:02d}' for i in range(1, 41)]
    assert {s['mapping_status'] for s in SYSTEMS} == {'benchmark_v1'} and not any('origin' in s for s in SYSTEMS)
    assert all('chains' in t for t in REC['tasks'].values() if t['class'] == 'polymer')
    assert sorted(s['recipe']['task'] for s in SYSTEMS) == sorted(REC['tasks'])
    assert all(s['recipe']['file'] == 'aa_recipes.json' for s in SYSTEMS)
    for s in SYSTEMS:
        assert REC['tasks'][s['recipe']['task']]['class'] == s['system_class'], s['key']
    used = {tok.split('*')[0] for t in REC['tasks'].values() for tok in str(t.get('sequence', '')).split()} | \
           {c for t in REC['tasks'].values() for c in t.get('random', {})}
    assert used == set(REC['units'])
    assert set(REC['states']) == set(REC['equilibration']) == {'polymer', 'liquid', 'liquid_mixture'}


@pytest.mark.parametrize('key', [s['key'] for s in SYSTEMS])
def test_every_mapping_applies_to_its_recipe(key):
    pytest.importorskip('rdkit')
    from ascg_core.mapping.formats import library_inputs
    from ascg_core.mapping.spec import INDICES, check
    _, maps = library_inputs(key, LIB, REC)
    assert maps['tasks'][key]['units']
    s = next(x for x in SYSTEMS if x['key'] == key)
    if s['mapping'].get('format') == INDICES and not s['mapping'].get('components'):    # complete, connected partitions
        rep = check(s['mapping'], s['system_class'])
        assert rep['ok'] and not any('not bonded' in w for w in rep['warnings']), rep


def test_readme_lists_every_system():
    text = (B/'README.md').read_text()
    assert all(f"| `{s['key']}` |" in text for s in SYSTEMS)
