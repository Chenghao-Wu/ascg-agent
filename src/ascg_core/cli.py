"""Command line of the numerical core: all-atom build, audit, GROMACS protocol and mapped references.

    python -m ascg_core <command> ...      (also reachable as `ascg <command> ...`)

The IBI engine has its own entry points: python -m ascg_core.ibi.pipeline | runner | selfcheck | plots.
Recipes and mappings default to the project's benchmark (benchmarks/aa_recipes.json and the system library
benchmarks/ascg_library.json); give --recipes / --mappings for other systems.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT/'benchmarks'                     # the project's benchmark: 40 systems
COMMANDS = ('build-opls', 'neutralize-aa', 'audit-opls', 'cg-topology', 'prepare-gmx-protocol', 'gmx-status',
            'map-aa')


def _json(path, default):
    return json.loads((Path(path) if path else default).read_text())


def _mapping_inputs(a):
    """(recipes, mappings) for the CG topology: a mappings file in repeat-unit groups, or a system library
    (ascg-library/1), whose rdkit-heavy-atom-indices mappings of liquids are converted (mapping.formats)."""
    recipes = _json(a.recipes, BENCH/'aa_recipes.json')
    mappings = _json(a.mappings, BENCH/'ascg_library.json')
    if mappings.get('schema') == 'ascg-library/1':
        from .mapping.formats import library_inputs
        return library_inputs(a.task_key, mappings, recipes)
    return recipes, mappings


def _mapping_parser(sub, name, hlp):
    m = sub.add_parser(name, help=hlp)
    m.add_argument('--system-dir', required=True)
    m.add_argument('--task-key', required=True)
    m.add_argument('--recipes', default=None, help='recipes JSON (default benchmarks/aa_recipes.json)')
    m.add_argument('--mappings', default=None, help='a system library (ascg-library/1; default '
                   'benchmarks/ascg_library.json) or a mappings JSON in repeat-unit groups')
    return m


def parser():
    ap = argparse.ArgumentParser(prog='ascg_core', description='All-atom build, audit, GROMACS protocol and mapped '
                                 'references')
    sub = ap.add_subparsers(dest='action', required=True)

    b = sub.add_parser('build-opls', help='Build an AA system with AutoPoly v2.0 OPLS-AA from a recipe')
    b.add_argument('--task-key', required=True, help='task key in the recipes file')
    b.add_argument('--recipes', default=None, help='recipes JSON (default benchmarks/aa_recipes.json)')
    b.add_argument('--mappings', default=None, help='for --target-beads: a system library (default '
                   'benchmarks/ascg_library.json) or a mappings JSON in repeat-unit groups')
    b.add_argument('--out', required=True)
    b.add_argument('--seed', type=int, default=42)
    b.add_argument('--topology-only', action='store_true', help='geometry and OPLS units only; no packing / moltemplate')
    b.add_argument('--chains', type=int, default=80, help='fixed chain count (superseded by --target-beads or the '
                   "recipe's chains)")
    b.add_argument('--target-beads', type=int, default=None,
                   help='chains = round(target / CG beads per chain), at least --min-chains (study: 1000)')
    b.add_argument('--min-chains', type=int, default=20)

    n = sub.add_parser('neutralize-aa', help='Write SYSTEM_DIR/system.neutral.data with every molecule neutral')
    n.add_argument('--system-dir', required=True)
    n.add_argument('--task-key', required=True, help='reads charge_fix_opls_types from the recipes file')
    n.add_argument('--recipes', default=None, help='recipes JSON (default benchmarks/aa_recipes.json)')

    a = sub.add_parser('audit-opls', help='Validate an AutoPoly LAMMPS data file')
    a.add_argument('--data', required=True)
    a.add_argument('--expected-density', type=float, default=None)

    _mapping_parser(sub, 'cg-topology', 'Map an AutoPoly system to CG beads and report the CG topology')
    m = _mapping_parser(sub, 'map-aa', 'Mapped-AA CG reference (trajectories, targets, start configurations) per state')
    m.add_argument('--run', action='append', required=True,
                   help='STATE=DIR for every approved state, STATE as 400, 400@1000 (atm) or 400_P1000; DIR a GROMACS '
                        'protocol branch (steps.json) or a LAMMPS run directory')
    m.add_argument('--out', required=True)
    m.add_argument('--r-max-nm', type=float, default=1.5)
    m.add_argument('--bond-bin-nm', type=float, default=0.002, help='bond histogram bin width (target resolution)')
    m.add_argument('--angle-bin-deg', type=float, default=2.0, help='angle histogram bin width')

    g = sub.add_parser('prepare-gmx-protocol', help='Convert an AutoPoly system to GROMACS and write the protocol: '
                       'equilibration (EQ21 or direct) and one branch per approved state point')
    g.add_argument('--system-dir', required=True, help='AutoPoly dir with system.neutral.data and system.in.settings')
    g.add_argument('--out', required=True, help='task root: writes gmx/, the equilibration dir and T<label>/ per state')
    g.add_argument('--states', '--temperatures', dest='states', required=True,
                   help='the user-approved state points: T in K, optionally @P in atm (or bar with the unit), e.g. '
                        '400,500 or 350,450 or 400,400@1000; never a default')
    g.add_argument('--eq21-temperature', type=float, default=None, help='EQ21 temperature (default: the hottest state)')
    g.add_argument('--ff-override', default=None, help='force-field JSON applied to the converted topology by chemical '
                   'environment (e.g. a published siloxane model where OPLS-AA lacks terms)')
    g.add_argument('--equilibration', choices=('eq21', 'direct'), default='eq21',
                   help='eq21: soft-core + EM + RadonPy EQ21 (polymers); direct: soft-core + EM + NVT + NPT '
                        'compression and release (molecular liquids)')
    g.add_argument('--eq21-scale', type=float, default=1.0, help='shorten every EQ21 stage (tests only)')
    g.add_argument('--pre-ns', type=float, default=1.0, help='v-rescale + C-rescale pre-equilibration before NH + PR')
    g.add_argument('--dihedrals', choices=('rb', 'fourier'), default='rb',
                   help='OPLS dihedrals as Ryckaert-Bellemans (exact; GPU listed forces) or Fourier (CPU)')
    g.add_argument('--eq-ns', type=float, default=10.0)
    g.add_argument('--prod-ns', type=float, default=10.0)
    g.add_argument('--frame-ps', type=float, default=10.0)

    s = sub.add_parser('gmx-status', help='Progress of GROMACS protocol roots')
    s.add_argument('roots', nargs='+')
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if a.action == 'build-opls':
        from .aa.autopoly_v2 import chains_for_beads, generate_opls_aa, study_recipe
        rfile = Path(a.recipes) if a.recipes else BENCH/'aa_recipes.json'
        recipes = json.loads(rfile.read_text())
        if a.task_key not in recipes['tasks']:
            raise SystemExit(f'No recipe for {a.task_key!r} in {rfile}')
        chains = a.chains
        if a.target_beads:
            _, mappings = _mapping_inputs(a)
            chains = chains_for_beads(a.task_key, recipes, mappings, target_beads=a.target_beads,
                                      min_chains=a.min_chains)
        recipe = study_recipe(a.task_key, a.out, recipes=recipes, chains=chains, seed=a.seed)
        print(json.dumps(generate_opls_aa(recipe, pack=not a.topology_only), indent=2))
    elif a.action == 'neutralize-aa':
        from .aa.lammps_audit import neutralize_molecules
        recipes = _json(a.recipes, BENCH/'aa_recipes.json')
        types = recipes['tasks'][a.task_key].get('charge_fix_opls_types', [])
        d = Path(a.system_dir)
        report = neutralize_molecules(d/'system.data', d/'system.neutral.data', adjust_opls_types=types)
        (d/'system.neutral.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    elif a.action == 'audit-opls':
        from .aa.lammps_audit import audit_lammps_data
        print(json.dumps(audit_lammps_data(a.data, expected_density=a.expected_density).to_dict(), indent=2))
    elif a.action == 'cg-topology':
        from .mapping.cg_mapping import build_cg_topology, read_data, read_layout
        recipes, mappings = _mapping_inputs(a)
        d = Path(a.system_dir)
        topo = build_cg_topology(a.task_key, read_layout(d), read_data(d/'system.neutral.data'), recipes, mappings)
        t = topo.to_dict()
        print(json.dumps({'task': a.task_key, 'beads': len(topo.bead_types), 'types': topo.type_names,
                          'beads_per_type': {n: topo.bead_types.count(n) for n in topo.type_names},
                          'bead_mass_range_amu': [float(topo.bead_masses_amu.min()), float(topo.bead_masses_amu.max())],
                          'bonds': len(topo.bonds), 'angles': len(topo.angles), 'exclusions': len(topo.exclusions),
                          'bond_types': t['bond_types'], 'angle_types': t['angle_types']}))
    elif a.action == 'map-aa':
        from .ibi.states import parse_one
        from .mapping.cg_reference import prepare_reference
        recipes, mappings = _mapping_inputs(a)
        runs = {parse_one(x.split('=', 1)[0]): x.split('=', 1)[1] for x in a.run}    # STATE=DIR per approved state
        print(json.dumps(prepare_reference(a.task_key, a.system_dir, runs, a.out, recipes=recipes, mappings=mappings,
                                           r_max_nm=a.r_max_nm, bond_bin_nm=a.bond_bin_nm,
                                           angle_bin_deg=a.angle_bin_deg), indent=2))
    elif a.action == 'prepare-gmx-protocol':
        from .aa.gmx_protocol import write_protocol
        from .aa.lmp2gmx import POLAR_H_LJ, convert
        d, root = Path(a.system_dir), Path(a.out)
        (root/'gmx').mkdir(parents=True, exist_ok=True)
        info = convert(d/'system.neutral.data', root/'gmx'/'system.gro', root/'gmx'/'system.top',
                       settings=d/'system.in.settings', title=f'{d.name} AutoPoly OPLS-AA', polar_h_lj=POLAR_H_LJ,
                       dihedral_style=a.dihedrals)
        if a.ff_override:
            from .aa.ff_override import apply as ff_apply
            top = root/'gmx'/'system.top'
            top.rename(root/'gmx'/'system.autopoly_oplsaa.top')
            info['ff_override'] = ff_apply(root/'gmx'/'system.autopoly_oplsaa.top', top, a.ff_override)
            info['ff_override']['file'] = str(Path(a.ff_override).resolve())
        out = write_protocol(root/'gmx', root, eq21_scale=a.eq21_scale, temperatures=a.states,
                             eq21_temperature=a.eq21_temperature, equilibration=a.equilibration,
                             branch_kwargs={'pre_ns': a.pre_ns, 'eq_ns': a.eq_ns, 'prod_ns': a.prod_ns,
                                            'frame_ps': a.frame_ps})
        print(json.dumps({'conversion': info, **out}, indent=2))
    elif a.action == 'gmx-status':
        from .aa.gmx_protocol import status
        for r in map(Path, a.roots):
            dirs = [k for k in ('eq21', 'equil') if (r/k/'steps.json').is_file()]
            dirs += sorted((p.name for p in r.glob('T*') if (p/'steps.json').is_file()),
                           key=lambda n: float(n[1:].split('_P')[0]) if n[1:].split('_P')[0].replace('.', '', 1).isdigit()
                           else 0.0)
            print(json.dumps({'root': str(r), **{k: (lambda s: f"{s['done']}/{s['steps']}")(status(r/k)) for k in dirs}}))


if __name__ == '__main__':
    main()
