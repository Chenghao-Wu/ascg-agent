"""aa / map / targets: atomistic references and mapping. The built-in polymer path (AutoPoly v2.0 OPLS-AA, RadonPy-21,
GROMACS) is wrapped as dry-run plans; existing or user-supplied mapped references are imported and audited."""
import json
from pathlib import Path

from ..compute import remote, target
from ..registry import P, PolicyError, ToolError, tool
from ..study import Study


def _ref_target(s, reg, task):
    rec = s.task(reg, task)
    return rec, target(rec['target'])


@tool('aa', 'import', 'Register an existing mapped AA reference (topology, targets, mapped trajectories, start '
      'configurations per state) for a task at the user-approved state points: preflight, reference QA and summary. '
      'The fit uses exactly the approved states (any N >= 1); each needs a reference at that temperature.', {
          'study': P(required=True), 'task': P(required=True, help='task key: a short identifier (the library key, if any)'),
          'target': P(required=True, help='compute target that holds the reference'),
          'ref': P(required=True, help='absolute path of the mapped reference on the target'),
          'states': P('array', required=True, help='the user-approved state points of this task: T in K, optionally '
                      '@P in atm (or bar with the unit), e.g. 400,500 or 450 or 400,400@1000; never a default')},
      mutating=True)
def aa_import(study, task, ref, states, **kw):
    s = Study(study)
    reg = s.load()
    t = target(kw['target'])
    old = reg['tasks'].get(task)
    if old and old.get('jobs'):
        raise PolicyError(f'task {task} already has jobs ({old.get("status")}) on {old.get("target")}; importing again '
                          'would orphan them - use a new task key (e.g. {task}_2) or cancel the task first')
    if not ref.startswith('/'):
        raise ToolError('--ref must be an absolute path on the target')
    info = remote(t, 'ref_info', {'ref': ref}, timeout=300)
    from ascg_core.ibi import states as st
    try:
        want = list(st.parse(states))
    except ValueError as e:
        raise PolicyError(f'state points: {e}')
    have = list(st.parse(info['reference_states']))
    extra = [T for T in want if T not in have]
    if extra:
        raise PolicyError(f'no AA reference at {st.labels(extra)} ({st.describe(extra)}; the reference holds '
                          f'{st.describe(have)}): the approved states need AA references first (ascg-aa-simulate; ask '
                          'the user before new AA simulations); never fit without them or replace the states')
    broken = [T for T in want if st.label(T) in (info.get('missing_files') or {})]
    if broken:
        raise PolicyError(f'the reference is incomplete at {st.labels(broken)}: {info["missing_files"]}')
    pf = remote(t, 'preflight', {'ref': ref, 'temperatures': st.labels(want)}, timeout=300)
    qa = remote(t, 'aa_qa', {'ref': ref}, timeout=900)
    rec = {'key': task, 'target': t['name'], 'ref': ref, 'task_dir': f"{t['work_root'].rstrip('/')}/{study}/{task}",
           'status': 'registered', 'state_points': st.labels(want), 'reference_states': st.labels(have),
           'preflight_ok': pf['ok'], 'qa_flags': qa['flags'],
           'failed_checks': [c for c in pf['checks'] if not c['ok']]}
    m = s.mapping(task)                               # a mapping saved / approved for this system (map save)
    if m:
        rec['mapping'] = {'status': m['status'], 'file': str(s.mapping_file(task))}
    reg['tasks'][task] = rec
    s.event(reg, 'task_registered', task, {'state_points': st.labels(want), 'preflight_ok': pf['ok'],
                                           'qa_flags': len(qa['flags'])})
    unused = [T for T in have if T not in want]
    return {'task': task, 'state_points': st.labels(want), 'state_points_described': st.describe(want),
            'fit': 'single-state IBI' if len(want) == 1 else f'multistate IBI over {len(want)} states',
            'reference_states_not_used': st.labels(unused), 'preflight': pf, 'reference': info, 'qa': qa}


@tool('aa', 'qa', 'Reference QA again (equilibration, chain relaxation by quarters, analysis range, state dependence).', {
    'study': P(required=True), 'task': P(required=True)})
def aa_qa(study, task):
    s = Study(study)
    rec, t = _ref_target(s, s.load(), task)
    return remote(t, 'aa_qa', {'ref': rec['ref']}, timeout=900)


SUPPORTED_MAPPING_FORMATS = ('ascg-repeat-unit-groups',)
SMILES_MAPPING_FORMATS = ('rdkit-heavy-atom-indices', 'annotated-smiles')
SMILES_CLASSES = ('polymer', 'liquid', 'liquid_mixture')


def _mapping_supported(entry):
    """map-aa applies repeat-unit groups, and the SMILES forms (annotated-smiles, rdkit-heavy-atom-indices) of
    polymers (repeat unit with two attachment points), molecular liquids and mixtures."""
    fmt = (entry.get('mapping') or {}).get('format')
    return fmt in SUPPORTED_MAPPING_FORMATS or (fmt in SMILES_MAPPING_FORMATS
                                                and entry.get('system_class') in SMILES_CLASSES)
DEFAULT_EQUILIBRATION = {'polymer': 'eq21', 'liquid': 'direct', 'liquid_mixture': 'direct'}


@tool('aa', 'plan', 'All-atom path as a plan (not executed): AutoPoly v2.0 OPLS-AA build from the system\'s recipe, '
      'typing / charge audit, neutralization, GROMACS conversion (with a force-field override where the library '
      'gives one), equilibration (polymers: RadonPy 21-step schedule at the hottest state; liquids and mixtures: '
      'direct compression), one branch per approved state point, mapping. The recipe comes from a system saved in '
      'the study (--study; system save, from SMILES) or from the user\'s system library.',
      {'task': P(required=True, help='system key (system save) or library key (ascg catalogue search)'),
       'states': P('array', required=True, help='user-approved state points: T in K, optionally @P in atm, e.g. '
                   '400,500 or 300 or 400,400@1000'),
       'study': P(help='study holding the system (system save)'),
       'beads': P('integer', default=1000)})
def aa_plan(task, states, beads, study=None):
    from ascg_core.ibi import states as st
    from ..config import load_library
    try:
        temps = st.parse(states)
    except ValueError as e:
        raise PolicyError(f'state points: {e}')
    sysrec = Study(study).system(task) if study else None
    notes = []
    if sysrec:
        files = sysrec['files']
        rec = json.loads(Path(files['recipes']).read_text())
        key, rfile, cls, ff, source = task, '<system>/recipes.json', sysrec['system_class'], None, \
            {'system': f'{task} in study {study}', 'status': sysrec['status']}
        mappings = '<system>/library.json' if 'library' in files else None
        if sysrec['status'] != 'approved':
            notes.append('the system is still proposed: the user approves it (system approve) before any AA build')
        notes.append(f'<system> is {Path(files["recipes"]).parent} here; the target that builds needs a copy')
        charge_fix = None
    else:
        lib = load_library()
        s = (lib or {}).get('systems', {}).get(task)
        if not s or not s.get('recipe'):
            return {'task': task, 'known_recipe': False, 'steps': [],
                    'note': 'no recipe: describe the system with SMILES and save it in the study (system check, '
                            'system save, system approve; skill ascg-specify-system), then aa plan --study S'}
        base = lib['path'].parent
        rec = json.loads((base/s['recipe']['file']).read_text())
        key, rfile = s['recipe']['task'], f"<library>/{s['recipe']['file']}"     # the target has its own copy
        cls = s.get('system_class') or rec['tasks'][key].get('class', 'polymer')
        ff = f"<library>/{s['force_field_override']}" if s.get('force_field_override') else None
        mappings = f'<library>/{lib["path"].name}' if _mapping_supported(s) else None
        source = {'library': lib['name']}
        if not mappings:
            m = s.get('mapping') or {}
            notes.append(f"mapping: format {m.get('format')!r} is not applied by map-aa yet" if m else
                         'mapping: none in the library - agree one with the user (ascg-mapping)')
        notes.append(f'<library> is the directory of the system library ({base} here); use its copy on the target')
        charge_fix = rec['tasks'][key].get('charge_fix_opls_types')
    t = rec['tasks'][key]
    equil = (rec.get('equilibration') or {}).get(cls) or DEFAULT_EQUILIBRATION.get(cls, 'eq21')
    eq_dir = 'eq21' if equil == 'eq21' else 'equil'
    eq_ns = 10 if cls == 'polymer' else 4
    labels = st.labels(temps)
    hot = max(x.T for x in temps)
    sized = 'chains' in t or 'molecules' in t
    steps = [
        f'python -m ascg_core build-opls --task-key {key} --recipes {rfile}'
        + ('' if sized else f' --target-beads {beads}') + f' --out runs/aa   # AutoPoly v2.0 OPLS-AA, system in runs/aa/{key}',
        f'ascg aa audit --target <T> --system-dir <abs>/runs/aa/{key}   # typing / charge audit: stop on findings',
        f'python -m ascg_core neutralize-aa --system-dir runs/aa/{key} --task-key {key} --recipes {rfile}'
        '   # only recorded charge fixes (charge_fix_opls_types); a large residual is a typing error',
        f'python -m ascg_core prepare-gmx-protocol --system-dir runs/aa/{key} --out runs/aa/{key}/protocol '
        f'--states {",".join(labels)} --equilibration {equil} --eq-ns {eq_ns}' + (f' --ff-override {ff}' if ff else '')
        + f'   # {"EQ21" if equil == "eq21" else "direct compression"} at {hot:g} K, branches '
        + ', '.join('T' + l for l in labels),
        f'python -m ascg_core.aa.gmx_protocol runs/aa/{key}/protocol/{eq_dir} '
        + ' '.join(f'runs/aa/{key}/protocol/T{l}' for l in labels)
        + '   # grompp + mdrun per step, resumes from checkpoints (one job; or the equilibration job, then one job per branch)',
    ]
    if mappings:
        steps.append(f'python -m ascg_core map-aa --system-dir runs/aa/{key} --task-key {key} --recipes {rfile} '
                     f'--mappings {mappings} ' + ' '.join(f'--run {l}=runs/aa/{key}/protocol/T{l}' for l in labels)
                     + f' --out runs/cg_ref/{key}   # mapped references, one per state')
    else:
        steps.append('# mapping: map-aa needs a mapping (bead numbers in the system SMILES, or the library\'s)')
    notes.insert(0, 'AA simulations are expensive: confirm chemistry, size, state points and compute with the user '
                    'first (ascg-aa-simulate); then ascg aa import the mapped reference.')
    if ff:
        notes.append('the force-field override replaces the OPLS-AA parameters after conversion: a non-OPLS choice '
                     'the user approves')
    if charge_fix:
        notes.append(f'recorded charge adjustment on OPLS types {charge_fix} (report it)')
    return {'task': task, 'known_recipe': True, **source, 'system_class': cls, 'equilibration': equil,
            'state_points': labels, 'ff_override': ff, 'steps': steps, 'notes': notes}



@tool('aa', 'audit', 'OPLS-AA typing and charge audit of a built all-atom system (AutoPoly / moltemplate directory '
      'with system.data and system.in.charges): net charge per molecule, type / charge / environment tables, and '
      'hints for known typing pitfalls (ester CH2 typed as ether, aromatic ester carbonyl typed aliphatic, nitrile '
      'CH3 typed as alkane, untyped atoms). Run it before neutralization and before any simulation.', {
          'target': P(required=True, help='compute target that holds the build'),
          'system_dir': P(required=True, help='absolute path of the build directory on the target'),
          'tolerance': P('number', default=1e-4, help='largest accepted |net charge| per molecule (e)'),
          'tables': P('boolean', default=False, help='include the full type tables')})
def aa_audit(system_dir, tolerance, tables, **kw):
    if not system_dir.startswith('/'):
        raise ToolError('--system-dir must be an absolute path on the target')
    return remote(target(kw['target']), 'aa_audit', {'system_dir': system_dir, 'tolerance': tolerance,
                                                     'tables': tables}, timeout=600)


@tool('map', 'review', 'Mapping audit of a registered reference: masses, bead types, molecules, bonded topology, '
      '1-2 / 1-3 exclusions, dihedrals; with the system library\'s mapping of the reference\'s system (looked up by '
      'the reference\'s own task name, so a task moved under a new key still finds it).',
      {'study': P(required=True), 'task': P(required=True)})
def map_review(study, task):
    from ..config import load_library
    s = Study(study)
    rec, t = _ref_target(s, s.load(), task)
    out = remote(t, 'map_review', {'ref': rec['ref'], 'task': task}, timeout=300)
    lib = load_library()
    systems = (lib or {}).get('systems', {})
    key = out.get('reference_task') if out.get('reference_task') in systems else task
    entry = systems.get(key)
    out['library_mapping'] = ({'library': lib['name'], 'key': key, 'mapping_status': entry.get('mapping_status'),
                               'mapping': entry.get('mapping')} if entry else None)
    return out


@tool('map', 'show', 'Mapping of a system: the one saved in a study (with --study; map save / approve), else the '
      'user\'s system library entry (bead sites, types, format and approval status).', {
          'task': P(required=True, help='system key (study mapping) or library key (ascg catalogue search)'),
          'study': P(help='study whose saved mapping to show first')})
def map_show(task, study=None):
    from ..config import load_library
    if study:
        m = Study(study).mapping(task)
        if m:
            return {'task': task, 'source': f'study {study}', 'status': m['status'], 'system_class': m['system_class'],
                    'spec': m['spec'], 'bead_types': m['bead_types'], 'species': m['species'],
                    'warnings': m['warnings'], 'rationale': m.get('rationale'), 'approved': m.get('approved'),
                    **({} if m['status'] == 'approved' else
                       {'approval': 'not approved: confirm the partition with the user (map approve)'})}
    lib = load_library()
    s = (lib or {}).get('systems', {}).get(task)
    if not s:
        raise ToolError(f'{task!r} is not in a configured system library; its mapping must come from the user')
    m = s.get('mapping')
    out = {'task': task, 'name': s.get('name'), 'system_class': s.get('system_class'),
           'resolution': s.get('resolution'), 'mapping_status': s.get('mapping_status'), 'mapping': m,
           'library': lib['name']}
    if m and not _mapping_supported(s):
        out['note'] = (f"mapping format {m.get('format')!r} is not applied by map-aa yet for this system class: a "
                       'mapped reference for it needs that support first (tell the user)')
    if s.get('mapping_status', '').startswith(('proposed', 'atom_partition_to_verify', 'benchmark')):
        out['approval'] = 'not approved: confirm the atom partition with the user before using it'
    return out


SYSTEM_CLASSES = ('polymer', 'liquid', 'liquid_mixture')


def _spec_arg(spec):
    """--spec: a JSON object, JSON text or the path of a JSON file."""
    if isinstance(spec, dict):
        return spec
    text = str(spec).strip()
    try:
        if text.startswith('{'):
            return json.loads(text)
        f = Path(text).expanduser()
        if f.is_file():
            return json.loads(f.read_text())
    except json.JSONDecodeError as e:
        raise ToolError(f'--spec is not valid JSON: {e}')
    raise ToolError('--spec: give the mapping as a JSON object (inline) or the path of a JSON file')


def _check(spec, system_class):
    try:
        from ascg_core.mapping.spec import check
    except ImportError as e:
        raise ToolError(f'map check needs RDKit in the Python that runs ascg (pip install ascg-agent; or point '
                        f'ASCG_PYTHON at a Python that has it): {e}')
    return check(spec, system_class)


@tool('map', 'check', 'Check an atom-to-bead mapping before anything is built (read-only): every heavy atom in exactly '
      'one bead, bead masses (hydrogens follow their heavy atom), the bead graph and the bond / angle / dihedral types '
      'the model will have, errors and warnings. A species is a molecule or a polymer repeat unit with its two * '
      'attachment points (head first); mixtures are {"species": [...]}. Forms: annotated-smiles - every heavy atom '
      'carries its bead number as atom-map number, bracket atoms with explicit H ([CH2:1]), plus "bead_types" '
      '{number: type}; or rdkit-heavy-atom-indices - "smiles" plus "bead_sites" with 1-based atom indices in the SMILES '
      '(attachment points counted).', {
          'spec': P('object', required=True, help='the mapping: a JSON object, JSON text or a JSON file'),
          'system_class': P(enum=SYSTEM_CLASSES, help='polymer, liquid or liquid_mixture (checks the species kind)')})
def map_check(spec, system_class=None):
    return _check(_spec_arg(spec), system_class)


@tool('map', 'save', 'Record a proposed mapping of a system in a study (it must pass map check). It stays "proposed" '
      'until the user approves it (map approve). A new proposal replaces an earlier one; earlier versions are kept in '
      'the history.', {
          'study': P(required=True), 'key': P(required=True, help='system key: the task key the reference will use'),
          'spec': P('object', required=True, help='the mapping (as for map check)'),
          'system_class': P(enum=SYSTEM_CLASSES, required=True),
          'rationale': P(help='why this partition: the chemistry and the trade-offs shown to the user')},
      mutating=True)
def map_save(study, key, spec, system_class, rationale=None):
    from ..study import now
    s = Study(study)
    reg = s.load()
    spec = _spec_arg(spec)
    rep = _check(spec, system_class)
    if not rep['ok']:
        raise PolicyError('map check fails; fix the mapping first: ' + '; '.join(rep['errors']))
    old = s.mapping(key)
    history = (old or {}).get('history', [])
    if old:
        history.append({k: old.get(k) for k in ('status', 'spec', 'proposed', 'approved', 'approved_by')})
    rec = {'key': key, 'system_class': system_class, 'status': 'proposed', 'spec': spec,
           'normalised': rep['normalised'], 'bead_types': rep['bead_types'], 'warnings': rep['warnings'],
           'species': [{k: sp[k] for k in ('smiles', 'kind', 'beads', 'mass_amu', 'bonds', 'angles', 'dihedrals')}
                       for sp in rep['species']],
           'rationale': rationale, 'proposed': now(), 'history': history}
    f = s.mapping_file(key)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec, indent=1) + '\n')
    s.event(reg, 'mapping_proposed', key, {'bead_types': rep['bead_types'], 'warnings': len(rep['warnings']),
                                           'replaces': (old or {}).get('status')})
    return {'key': key, 'status': 'proposed', 'file': str(f), 'check': rep,
            'next': 'show the user the partition, masses, bonded types and warnings; record their approval with '
                    'map approve'}


@tool('map', 'approve', "Record the user's approval of the proposed mapping of a system. Only after the user has "
      'explicitly approved this partition; give their words in --note.', {
          'study': P(required=True), 'key': P(required=True), 'note': P(help="the user's approval, in their words")},
      mutating=True)
def map_approve(study, key, note=None):
    import os
    from ..study import now
    s = Study(study)
    reg = s.load()
    rec = s.mapping(key)
    if rec is None:
        raise ToolError(f'no mapping proposed for {key!r} in study {study!r} (map save)')
    if rec['status'] == 'approved':
        return {'key': key, 'status': 'approved', 'approved': rec.get('approved'), 'note': 'already approved'}
    rec.update(status='approved', approved=now(), approved_by=os.environ.get('ASCG_ACTOR', 'user'), approval_note=note)
    s.mapping_file(key).write_text(json.dumps(rec, indent=1) + '\n')
    s.event(reg, 'mapping_approved', key, {'note': note, 'bead_types': rec['bead_types']})
    if key in reg['tasks']:
        reg['tasks'][key]['mapping'] = {'status': 'approved', 'file': str(s.mapping_file(key))}
        s.save(reg)
    return {'key': key, 'status': 'approved', 'file': str(s.mapping_file(key))}


def _abs(path, what):
    if not str(path).startswith('/'):
        raise ToolError(f'--{what} must be an absolute path on the target')
    return str(path)


def _runs(run, timestep_fs=None):
    """--run STATE=TRAJECTORY[+MORE][,ENERGY] per user-approved state point -> {state label: {...}}."""
    from ascg_core.ibi import states as st
    items = []
    for r in run:                        # the CLI splits list values on commas: re-attach an energy file to its run
        if '=' not in str(r) and items:
            items[-1] += ',' + str(r)
        else:
            items.append(str(r))
    out = {}
    for r in items:
        if '=' not in r:
            raise ToolError(f'--run {r!r}: expected STATE=TRAJECTORY[,ENERGY], e.g. 300=/abs/prod.xtc,/abs/energy.xvg')
        state, files = r.split('=', 1)
        try:
            label = st.label(st.parse_one(state.strip()))
        except ValueError as e:
            raise PolicyError(f'state point {state!r}: {e}')
        traj, _, energy = files.partition(',')
        segs = [_abs(x.strip(), 'run') for x in traj.split('+') if x.strip()]
        out[label] = {'trajectory': segs if len(segs) > 1 else segs[0],
                      'energy': _abs(energy.strip(), 'run') if energy.strip() else None,
                      'timestep_fs': timestep_fs}
    if not out:
        raise ToolError('give at least one --run STATE=TRAJECTORY[,ENERGY]')
    return out


@tool('ref', 'inspect', 'Bring-your-own AA data on a target (read-only): atoms, bonds, molecules by formula, residue '
      'names with their atom names, frames, time span and frame spacing, box volume (NVT or NPT), density, energy-file '
      'averages (T, P). With a mapping (--mapping, or the study mapping --study/--key) also the bead assignment: beads '
      'and bead masses per molecule type, bond / angle / dihedral types. Topology: .tpr, .top, LAMMPS data, PSF, ... '
      '(with bonds); trajectory: .xtc, .trr, .dcd, LAMMPS dump, ...; energy: gmx energy .xvg, .edr (pyedr), LAMMPS log.', {
          'target': P(required=True, help='compute target that holds the files'),
          'topology': P(required=True, help='absolute path on the target'),
          'trajectory': P(help='absolute path on the target (one segment)'),
          'energy': P(help='absolute path of an energy file on the target'),
          'mapping': P('object', help='a mapping to try (JSON object, JSON text or a local JSON file)'),
          'study': P(help='with --key: use that study\'s saved mapping'), 'key': P(help='system key of the saved mapping'),
          'timestep_fs': P('number', help='LAMMPS dumps without a log: the integration timestep in fs')})
def ref_inspect(topology=None, trajectory=None, energy=None, mapping=None, study=None, key=None, timestep_fs=None,
                **kw):
    t = target(kw['target'])
    if mapping is None and study and key:
        m = Study(study).mapping(key)
        if m is None:
            raise ToolError(f'no mapping saved for {key!r} in study {study!r}')
        mapping = m['spec']
    payload = {'topology': _abs(topology, 'topology'), 'timestep_fs': timestep_fs,
               'trajectory': _abs(trajectory, 'trajectory') if trajectory else None,
               'energy': _abs(energy, 'energy') if energy else None,
               'mapping': _spec_arg(mapping) if mapping is not None else None}
    return remote(t, 'ref_inspect', payload, timeout=1800)


@tool('ref', 'build', 'Build the mapped reference of bring-your-own AA data on its target, with the study\'s APPROVED '
      'mapping (map save + map approve) and one --run per user-approved state point; then register it with aa import. '
      'Discard equilibration with --start-ps; thin long trajectories with --stride (about 500-1000 frames per state).', {
          'study': P(required=True), 'key': P(required=True, help='system / task key (its mapping must be approved)'),
          'target': P(required=True), 'topology': P(required=True, help='absolute path on the target'),
          'run': P('array', required=True, help='STATE=TRAJECTORY[+SEGMENT...][,ENERGY] per state point, e.g. '
                                                  '300=/abs/prod.xtc,/abs/energy.xvg or 400@1000=/abs/a.xtc'),
          'out': P(help='absolute output directory on the target (default <work_root>/refs/<study>/<key>)'),
          'start_ps': P('number', default=0.0, help='discard frames (and energy rows) before this time'),
          'stride': P('integer', default=1, help='use every n-th frame'),
          'timestep_fs': P('number', help='LAMMPS dumps without a log: the integration timestep in fs'),
          'r_max_nm': P('number', help='RDF range (default 1.5 nm; must be below half the box)'),
          'bond_bin_nm': P('number'), 'angle_bin_deg': P('number')}, mutating=True)
def ref_build(study, key, topology, run, out=None, start_ps=0.0, stride=1, timestep_fs=None, r_max_nm=None,
              bond_bin_nm=None, angle_bin_deg=None, **kw):
    s = Study(study)
    reg = s.load()
    m = s.mapping(key)
    if m is None:
        raise PolicyError(f'no mapping for {key!r} in study {study!r}: propose it (map check, map save) and get the '
                          "user's approval (map approve) first")
    if m['status'] != 'approved':
        raise PolicyError(f'the mapping of {key!r} is {m["status"]}: the user must approve it (map approve) first')
    t = target(kw['target'])
    runs = _runs(run, timestep_fs)
    out = _abs(out, 'out') if out else f"{t['work_root'].rstrip('/')}/refs/{study}/{key}"
    payload = {'task': key, 'topology': _abs(topology, 'topology'), 'runs': runs, 'mapping': m['spec'], 'out': out,
               'start_ps': start_ps, 'stride': stride, 'r_max_nm': r_max_nm, 'bond_bin_nm': bond_bin_nm,
               'angle_bin_deg': angle_bin_deg}
    res = remote(t, 'ref_build', payload, timeout=7200)
    states = {T: {k: v.get(k) for k in ('frames', 'density_g_cm3', 'temperature_K_mean', 'pressure_bar_mean',
                                         'kappa_T_per_bar', 'ensemble')}
              for T, v in res['summary']['states'].items()}
    s.event(reg, 'reference_built', key, {'target': t['name'], 'ref': out, 'states': list(states),
                                          'qa_flags': len(res['qa']['flags'])})
    return {'key': key, 'ref': out, 'states': states, 'qa': res['qa'],
            'next': f'ascg aa import --study {study} --task {key} --target {t["name"]} --ref {out} '
                    f'--states {",".join(states)}'}


@tool('targets', 'review', 'Target distributions of a registered reference (RDFs, bonds, angles and the AA dihedral '
      'distributions the fit will use): bins, roughness (noise), unsampled bins, range checks.', {
    'study': P(required=True), 'task': P(required=True)})
def targets_review(study, task):
    s = Study(study)
    rec, t = _ref_target(s, s.load(), task)
    return remote(t, 'targets_review', {'ref': rec['ref']}, timeout=900)
