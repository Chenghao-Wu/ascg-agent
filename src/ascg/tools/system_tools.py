"""system: a system described by one SMILES per species (plus size and chain-end choices): the AA build recipe and,
with bead numbers in the SMILES, the CG mapping, derived together, saved in the study and approved by the user."""
import json
import os

from ..registry import P, PolicyError, ToolError, tool
from ..study import Study, now
from .reference_tools import SYSTEM_CLASSES, _spec_arg, map_approve, map_save

SPEC_HELP = ('the system: {"system_class": "polymer" | "liquid" | "liquid_mixture", "species": [{"name", "smiles", '
             '"bead_types"; liquids: "count" or "fraction"}], copolymers: one species per monomer and "sequence" '
             '("A*5 B*5", "A B A B ...") or "random" ({"A": 0.5, "B": 0.5}), polymers: "repeat_units" (10), '
             '"tacticity" (atactic), '
             '"end_groups" ("H", "CH3" or {"head", "tail"}) or "first" / "last" end units, "chains" or '
             '"target_cg_beads" (1000); liquids: "molecules" (1000). The SMILES is a repeat unit with two * (head '
             'first) or a whole molecule; with bead numbers (annotated SMILES, as for map check) the mapping comes '
             'with it')


def _derive(spec, key):
    try:
        from ascg_core.aa.system_spec import derive
    except ImportError as e:
        raise ToolError(f'system check needs RDKit in the Python that runs ascg: {e}')
    return derive(spec, key)


def _summary(d):
    return {k: d.get(k) for k in ('ok', 'errors', 'warnings', 'spec', 'species', 'size')} | {
        'mapping': d.get('mapping'), 'bead_types': (d.get('mapping_check') or {}).get('bead_types'),
        'beads': [[{k: b.get(k) for k in ('site_id', 'bead_type', 'heavy_atoms', 'mass_amu')} for b in sp['beads']]
                  for sp in (d.get('mapping_check') or {}).get('species', [])] or None,
        'bonded_types': d.get('bonded_types'),
        'recipe_task': (d.get('recipe') or {}).get('tasks'), 'units': (d.get('recipe') or {}).get('units')}


@tool('system', 'check', 'Check a system described by SMILES before anything is built (read-only): every species '
      'parses, repeat units have two attachment points, chain ends, sizes (chains or molecules, atoms, mass, CG beads, '
      'box at 1 g/cm3), the AutoPoly units the AA build will use and, with bead numbers in the SMILES, the mapping '
      'check (masses, bonded types). Errors block; warnings go to the user.', {
          'spec': P('object', required=True, help=SPEC_HELP), 'key': P(default='system', help='system / task key')})
def system_check(spec, key='system'):
    return _summary(_derive(_spec_arg(spec), key))


@tool('system', 'save', 'Record a proposed system in a study (it must pass system check): the spec, the derived AA '
      'build recipe and, with bead numbers in the SMILES, the mapping (saved as with map save). It stays "proposed" '
      'until the user approves it (system approve). A new proposal replaces an earlier one (kept in the history).', {
          'study': P(required=True), 'key': P(required=True, help='system key: the task key the reference will use'),
          'spec': P('object', required=True, help=SPEC_HELP),
          'rationale': P(help='the user\'s request and the choices made for them (chain length, ends, size)')},
      mutating=True)
def system_save(study, key, spec, rationale=None):
    s = Study(study)
    reg = s.load()
    spec = _spec_arg(spec)
    d = _derive(spec, key)
    if not d['ok']:
        raise PolicyError('system check fails; fix the system first: ' + '; '.join(d['errors']))
    old = s.system(key)
    history = (old or {}).get('history', [])
    if old:
        history.append({k: old.get(k) for k in ('status', 'spec', 'proposed', 'approved', 'approved_by')})
    out_dir = s.system_file(key).with_suffix('')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'recipes.json').write_text(json.dumps(d['recipe'], indent=1) + '\n')
    files = {'recipes': str(out_dir/'recipes.json')}
    if d['library']:
        (out_dir/'library.json').write_text(json.dumps(d['library'], indent=1) + '\n')
        files['library'] = str(out_dir/'library.json')
    rec = {'key': key, 'system_class': spec['system_class'], 'status': 'proposed', 'spec': spec,
           'derived': {k: d[k] for k in ('spec', 'species', 'size', 'warnings')}, 'files': files,
           'mapping_from_system': d['mapping'] is not None, 'rationale': rationale, 'proposed': now(),
           'history': history}
    s.system_file(key).write_text(json.dumps(rec, indent=1) + '\n')
    s.event(reg, 'system_proposed', key, {'system_class': spec['system_class'], 'size': d['size'],
                                         'warnings': len(d['warnings']), 'replaces': (old or {}).get('status')})
    out = {'key': key, 'status': 'proposed', 'file': str(s.system_file(key)), 'files': files,
           'check': _summary(d)}
    if d['mapping'] is not None:
        out['mapping'] = map_save(study, key, d['mapping'], spec['system_class'],
                                  rationale=f'from the bead numbers of system {key}' + (f': {rationale}' if rationale
                                                                                        else ''))['status']
    out['next'] = ('show the user the species, chain ends, size and (with bead numbers) the partition, masses, bonded '
                   'types and every warning; record their approval with system approve')
    return out


@tool('system', 'approve', "Record the user's approval of the proposed system (and of the mapping that came with its "
      'bead numbers, if that mapping is still the proposed one). Only after the user has explicitly approved it; give '
      'their words in --note.', {
          'study': P(required=True), 'key': P(required=True), 'note': P(help="the user's approval, in their words")},
      mutating=True)
def system_approve(study, key, note=None):
    s = Study(study)
    reg = s.load()
    rec = s.system(key)
    if rec is None:
        raise ToolError(f'no system proposed for {key!r} in study {study!r} (system save)')
    out = {'key': key}
    if rec['status'] == 'approved':
        out.update(status='approved', approved=rec.get('approved'), note='already approved')
    else:
        rec.update(status='approved', approved=now(), approved_by=os.environ.get('ASCG_ACTOR', 'user'),
                   approval_note=note)
        s.system_file(key).write_text(json.dumps(rec, indent=1) + '\n')
        s.event(reg, 'system_approved', key, {'note': note})
        out.update(status='approved', file=str(s.system_file(key)))
    m = s.mapping(key)
    if rec.get('mapping_from_system') and m and m['status'] == 'proposed':
        d = _derive(rec['spec'], key)
        if d['mapping'] is not None and json.dumps(m['spec'], sort_keys=True) == json.dumps(d['mapping'], sort_keys=True):
            out['mapping'] = map_approve(study, key, note=note)['status']
        else:
            out['mapping'] = 'proposed (it differs from the system\'s bead numbers: approve it separately)'
    elif m:
        out['mapping'] = m['status']
    return out


@tool('system', 'show', 'A system saved in a study: spec, status, derived size and chain ends, files (recipe, mapping '
      'library).', {'study': P(required=True), 'key': P(required=True)})
def system_show(study, key):
    rec = Study(study).system(key)
    if rec is None:
        raise ToolError(f'no system {key!r} in study {study!r} (system save)')
    return rec
