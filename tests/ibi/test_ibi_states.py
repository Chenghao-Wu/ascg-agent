"""User-approved state points (design Revision 6): any N >= 1 (temperature, pressure) points through the IBI engine,
the pipeline, the reports and the AA protocol; the 400 / 500 K benchmark runs keep their behaviour."""
import json
from pathlib import Path

import numpy as np
import pytest

from ascg_core.ibi import pipeline as P
from ascg_core.ibi import states as st
from ascg_core.ibi.tables import TPotential, solve_ab
from ascg_core.ibi.update import apply_tcorrection


def test_parse_label_and_weights():
    assert st.parse('500,400') == (400, 500) and st.parse([450.0]) == (450,)
    assert st.parse('298.15 350') == (298.15, 350) and st.label(298.150) == '298.15' and st.label(400.0) == '400'
    for bad in ('', '400,400', '-5', '9000'):
        with pytest.raises(ValueError):
            st.parse(bad)
    assert st.weights((400, 500)) == {400: 0.5, 500: 0.5}
    assert st.weights((400, 500), w400=0.3) == pytest.approx({400: 0.3, 500: 0.7})
    assert st.weights((350, 400, 450), '1,1,2') == pytest.approx({350: 0.25, 400: 0.25, 450: 0.5})
    assert st.weights((350, 450), '450:3,350:1') == pytest.approx({350: 0.25, 450: 0.75})
    with pytest.raises(ValueError):
        st.weights((350, 450), w400=0.5)                       # the old option belongs to the benchmark pair
    with pytest.raises(ValueError):
        st.weights((350, 450), '1,2,3')
    assert st.of_run({'settings': {}}) == st.BENCHMARK_TEMPERATURES_K          # runs before temperatures were recorded
    assert st.of_run({'settings': {'temperatures': [450]}}) == (450,)
    assert st.describe((350, 400, 450)) == '350 K, 400 K and 450 K'


def test_pressure_state_points():
    s = st.parse('500, 400@1000, 400, 450@1013.25bar')
    assert st.labels(s) == ['400', '400_P1000', '450_P1000', '500']          # 1013.25 bar = 1000 atm
    assert s[1].T == 400 and s[1].P == 1000 and s[1].pressure_bar == pytest.approx(1013.25)
    assert st.parse_one('T400_P1000') == s[1] and st.parse_one({'temperature_K': 400, 'pressure_atm': 1000}) == s[1]
    assert s[0] == 400 and s[1] != 400 and {400: 'x'}[s[0]] == 'x'                # 1 atm states are their temperature
    assert st.describe(s[:3]) == '400 K; 400 K, 1000 atm and 450 K, 1000 atm' and s[1].pretty() == '400 K, 1000 atm'
    with pytest.raises(ValueError):
        st.parse('400@1000,400@1013.25bar')                                     # the same state twice
    with pytest.raises(ValueError):
        st.parse('400@-5')
    assert st.of_run({'settings': {'states': ['400', '400_P1000']}}) == st.parse('400,400@1000')
    assert st.weights(s[:2], '400:1,400_P1000:3') == pytest.approx({s[0]: 0.25, s[1]: 0.75})
    assert json.loads(json.dumps(st.jsonable({s[1]: [s[0]]}))) == {'400_P1000': ['400']}
    a, b = solve_ab({s[0]: np.array([1.0]), s[1]: np.array([3.0])})            # one temperature: one table set
    np.testing.assert_allclose(a, 2.0)
    np.testing.assert_allclose(b, 0.0)
    a, b = solve_ab({s[0]: np.array([1.0]), s[1]: np.array([3.0])}, {s[0]: 0.75, s[1]: 0.25})
    np.testing.assert_allclose(a, 1.5)
    from ascg_core.ibi.update import reith_amplitudes, virial_amplitudes
    kts = {x: 3.0 for x in s[:2]}
    amp = reith_amplitudes({s[0]: 1.0, s[1]: 1.0}, kts, target_bar={s[0]: 1.01325, s[1]: 1013.25})
    assert abs(amp[s[0]]) < 1e-3 and amp[s[1]] > 0                             # below its target: push up


def test_solve_ab_for_one_two_and_more_states():
    u = {400: np.array([1.0, 2.0]), 500: np.array([3.0, 0.0])}
    a, b = solve_ab(u)
    for T in u:
        np.testing.assert_allclose(a + b * T, u[T], atol=1e-12)      # two states: exact (weights do not matter)
    a1, b1 = solve_ab({450: np.array([1.0, 2.0])})
    np.testing.assert_allclose(a1, [1.0, 2.0])
    np.testing.assert_allclose(b1, 0.0)                              # one state: B = 0
    lin = {T: 2.0 + 0.01 * T * np.ones(3) for T in (350, 400, 450)}
    a3, b3 = solve_ab(lin)
    np.testing.assert_allclose(b3, 0.01, atol=1e-12)                 # linear data: exact
    np.testing.assert_allclose(a3, 2.0, atol=1e-9)
    bent = {350: np.zeros(1), 400: np.ones(1), 450: np.zeros(1)}
    eq = solve_ab(bent)
    wt = solve_ab(bent, {350: 0.1, 400: 0.8, 450: 0.1})
    assert abs((wt[0] + wt[1] * 400)[0] - 1.0) < abs((eq[0] + eq[1] * 400)[0] - 1.0)  # weight pulls the fit
    tp = TPotential('bond', np.linspace(0.3, 0.6, 31), np.zeros(31), np.zeros(31))
    x = np.linspace(0.3, 0.6, 10)
    new = apply_tcorrection(tp, x, {450: np.full(10, 0.5)})
    np.testing.assert_allclose(new.a + new.b * 450, 0.5)
    np.testing.assert_allclose(new.b, 0.0)


def _toy_reference(path, temps):
    from ascg_core.ibi.selfcheck import toy_system
    sysm, pos, box = toy_system(n_chains=2, length=6)
    path.mkdir(parents=True)
    (path/'topology.json').write_text(json.dumps({
        'type_names': ['A'], 'bead_types': [0] * sysm.n, 'bonds': sysm.bonds.tolist(), 'angles': sysm.angles.tolist(),
        'exclusions': sysm.bonds.tolist() + sysm.excl13.tolist(), 'bead_masses_amu': [28.0] * sysm.n,
        'bead_molecule': sysm.mol.tolist(), 'bond_types': ['A-A'], 'angle_types': ['A-A-A']}))
    states = {}
    for T in temps:
        L = st.label(T)       # 1 atm: '400'; other pressures: '400_P1000'
        np.savez(path/f'targets_T{L}.npz', **{'rdf/A-A': np.ones(20), 'bond/A-A': np.ones(20), 'angle/A-A-A': np.ones(20)})
        np.savez(path/f'start_T{L}.npz', pos_nm=pos, box_nm=box)
        np.savez(path/f'mapped_T{L}.npz', pos_nm=np.repeat(pos[None], 8, 0))
        states[L] = {'density_g_cm3': 0.8, 'density_se': 0.001, 'density_drift_g_cm3': 0.0, 'rg_nm': 0.5,
                     'ree_rms_nm': 1.0, 'r_max_over_half_box': 0.5, 'frames': 8}
    (path/'summary.json').write_text(json.dumps({'task': 'toy', 'n_beads': sysm.n, 'type_names': ['A'],
                                                 'bond_types': ['A-A'], 'angle_types': ['A-A-A'], 'states': states}))
    return path


def test_preflight_checks_the_approved_states(tmp_path):
    ref = _toy_reference(tmp_path/'toy', (350, 400, 298.15, st.parse_one('400@1000')))
    pf = P.preflight(ref)
    assert pf['ok'] and pf['states'] == ['298.15', '350', '400', '400_P1000']
    assert P.preflight(ref, '350')['ok'] and P.preflight(ref, [350])['states'] == ['350']
    assert P.preflight(ref, '400,400@1000')['states'] == ['400', '400_P1000']
    bad = P.preflight(ref, '350,500')
    assert not bad['ok'] and '500' in next(c['detail'] for c in bad['checks'] if not c['ok'])
    bad = P.preflight(ref, '350@2')
    assert not bad['ok'] and '2 atm' in next(c['detail'] for c in bad['checks'] if not c['ok'])


def _validation(temps, dens=0.01, rdf=0.05, rg=(0.5, 0.5)):
    seeds = [{'rg_nm': rg[0], 'density_error': dens}]
    return {'seeds': 1, 'validated': True, 'states': {st.label(T): {
        'aa': {'density_g_cm3': 0.8, 'rg_nm': rg[1], 'ree_rms_nm': 1.0}, 'seeds': seeds, 'density_error_mean': dens,
        'rdf_nrmse_worst': rdf, 'pass_all_seeds': abs(dens) <= 0.05 and rdf <= 0.15} for T in temps}}


def test_classify_and_convergence_follow_the_run_states(tmp_path):
    assert P.classify(_validation((450,)))[0] == 'validated'
    v = _validation((350, 400, 450))
    v['states']['450'].update(density_error_mean=0.08, pass_all_seeds=False)
    cls, why = P.classify(v)
    assert cls == 'failed under budget' and why == ['450 K: density +8.0 %, worst RDF 0.050']
    (tmp_path/'workflow').mkdir()
    hist = {'iteration': 6, 'family': 'dihedral', 'rdf_450': {'A-A': 0.02}, 'p_bar_450': 3.0, 'density_error_450': 0.01}
    (tmp_path/'workflow'/'state.json').write_text(json.dumps(
        {'completed': 6, 'settings': {'temperatures': [450]}, 'history': [hist]}))
    ok, why = P.converged(tmp_path)
    assert ok and '450 K' in why


def test_reports_and_aggregate_with_mixed_state_sets(tmp_path):
    for task, temps in (('a', (400, 500)), ('b', (450,)), ('c', (350, 400, 450)), ('d', st.parse('400,400@1000'))):
        d = tmp_path/'runs'/task
        (d/'cg'/'ibi'/'final').mkdir(parents=True)
        (d/'workflow').mkdir()
        (d/'workflow'/'state.json').write_text(json.dumps({'completed': 42, 'history': [],
                                                           'settings': {'states': st.labels(st.parse(temps))}}))
        (d/'cg'/'ibi'/'final'/'validation.json').write_text(json.dumps(_validation(temps)))
        P.write_reports(tmp_path/'ref', d, {'task': task, 'class': 'validated', 'reasons': [], 'iterations': 42})
        summ = (d/'report'/'user_summary.md').read_text()
        assert all(f'| {st.parse_one(T).pretty()} |' in summ for T in temps)
        assert st.describe(temps) in summ
    rows = P.aggregate(tmp_path/'runs', tmp_path/'out')
    assert {r['task']: r['states'] for r in rows} == {'a': ['400', '500'], 'b': ['450'], 'c': ['350', '400', '450'],
                                                      'd': ['400', '400_P1000']}
    md = (tmp_path/'out'/'benchmark.md').read_text()
    assert '| b | validated | 450 |' in md and '350 / 400 / 450' in md and '400 / 400_P1000' in md
    assert (tmp_path/'out'/'benchmark.png').exists()


def test_gmx_protocol_branches_at_the_approved_states(tmp_path):
    from ascg_core.aa.gmx_protocol import eq21_steps, write_protocol
    hot = [s for s in eq21_steps(temperature_K=650) if s[1].get('ref-t') == 750.0]
    assert hot                                                  # hot stages at T + 100 K above 600 K
    g = tmp_path/'gmx'
    g.mkdir()
    (g/'system.top').write_text('; test\n')
    (g/'system_sc.top').write_text('; test\n')
    (g/'system.gro').write_text('test\n0\n1 1 1\n')
    out = write_protocol(g, tmp_path/'root', temperatures='298.15,350,350@1000', eq21_scale=0.001)
    assert out['eq21_temperature_K'] == 350.0 and out['branch_dirs'] == ['T298.15', 'T350', 'T350_P1000']
    hp = (tmp_path/'root'/'T350_P1000'/'eq.mdp').read_text()
    assert '1013.25' in hp and '1000 atm' in (tmp_path/'root'/'T350_P1000'/'pre.mdp').read_text()
    assert (tmp_path/'root'/'T298.15'/'prod.mdp').exists() and 'ref-t' in (tmp_path/'root'/'T350'/'eq.mdp').read_text()
    assert '298.15' in (tmp_path/'root'/'T298.15'/'eq.mdp').read_text()


def test_benchmark_recipes_and_direct_equilibration(tmp_path):
    """The benchmark's 40 recipes parse (polymers of 9-10 units at about 1000 CG beads, liquids 1000 molecules, the
    mixture 500 + 500) and the liquids' direct equilibration protocol is written with one T300 branch."""
    from ascg_core.aa.autopoly_v2 import study_recipe
    from ascg_core.aa.gmx_protocol import write_protocol
    root = Path(__file__).resolve().parents[2]
    rec = json.loads((root/'benchmarks'/'aa_recipes.json').read_text())
    assert len(rec['tasks']) == 40
    for key, t in rec['tasks'].items():
        r = study_recipe(key, str(tmp_path), recipes=rec)
        if t['class'] == 'polymer':
            assert sum(p.chain_num for p in r.polymers) == t['chains'] >= 20
            assert all(len(p.sequence) in (9, 10) for p in r.polymers)
            if t.get('beads_per_repeat'):
                assert abs(t['chains'] * len(r.polymers[0].sequence) * t['beads_per_repeat'] - 1000) <= 20
        else:
            assert all(len(p.sequence) == 1 and '[*]' not in p.sequence[0] for p in r.polymers)
            assert sum(p.chain_num for p in r.polymers) == 1000
    g = tmp_path/'gmx'
    g.mkdir()
    for f in ('system.top', 'system_sc.top'):
        (g/f).write_text('; test\n')
    (g/'system.gro').write_text('test\n0\n1 1 1\n')
    out = write_protocol(g, tmp_path/'liq', temperatures='300', equilibration='direct', branch_kwargs={'eq_ns': 4})
    assert out['equilibration_dir'] == 'equil' and out['branch_dirs'] == ['T300']
    steps = json.loads((tmp_path/'liq'/'equil'/'steps.json').read_text())['steps']
    assert [s['name'] for s in steps] == ['em_sc', 'relax_sc', 'em', 'nvt', 'compress', 'release']
    assert '1013.25' in (tmp_path/'liq'/'equil'/'compress.mdp').read_text()
    br = json.loads((tmp_path/'liq'/'T300'/'steps.json').read_text())['steps']
    assert br[0]['conf'].endswith('equil/release.gro') and br[1]['nsteps'] == 4_000_000


def test_pdms_force_field_override(tmp_path):
    """benchmarks/ff/pdms_oplsaa_xiang2025.json on a minimal MD1M siloxane (Me3Si-O-SiMe2-O-SiMe3): benchmark charges
    (neutral molecule), LJ types, bonds, all angles / dihedrals regenerated from the bond graph, unknown terms refused."""
    from ascg_core.aa.ff_override import apply
    root = Path(__file__).resolve().parents[2]
    el, bonds = [], []

    def add(e, to=None):
        el.append(e)
        if to is not None:
            bonds.append((to, len(el)))
        return len(el)

    def methyl(si):
        c = add('C', si)
        for _ in range(3):
            add('H', c)
    s1 = add('Si'); [methyl(s1) for _ in range(3)]
    o1 = add('O', s1); s2 = add('Si', o1); [methyl(s2) for _ in range(2)]
    o2 = add('O', s2); s3 = add('Si', o2); [methyl(s3) for _ in range(3)]
    mass = {'Si': 28.086, 'O': 15.999, 'C': 12.011, 'H': 1.008}
    top = ['[ defaults ]', '  1  3  yes  0.5  0.5', '', '[ atomtypes ]', '  t1  1.0  0.0  A  0.1  0.1', '',
           '[ moleculetype ]', '  M1  3', '', '[ atoms ]']
    top += [f'  {i}  t1  1  M1  {e}{i}  {i}  0.0  {mass[e]}' for i, e in enumerate(el, 1)]
    top += ['', '[ bonds ]'] + [f'  {a}  {b}  1  0.15  1000.0' for a, b in bonds] + ['', '[ pairs ]', '',
                                                                                     '[ system ]', '  t', '',
                                                                                     '[ molecules ]', '  M1  1']
    (tmp_path/'in.top').write_text('\n'.join(top) + '\n')
    rep = apply(tmp_path/'in.top', tmp_path/'out.top', root/'benchmarks'/'ff'/'pdms_oplsaa_xiang2025.json')
    txt = (tmp_path/'out.top').read_text()
    q = [float(l.split()[6]) for l in txt.split('[ atoms ]')[1].split('[ bonds ]')[0].splitlines()
         if l.strip() and not l.startswith(';')]
    assert abs(sum(q)) < 1e-9 and q[s2 - 1] == 2.121 and q[s1 - 1] == 1.874 and q[o1 - 1] == -1.258
    assert rep['bonds'] == len(bonds) and 'ff_SiD' in txt and '  1  145.000000  167.360000' in txt   # Si-O-Si
    assert rep['dihedrals'] > 0 and rep['dihedrals'] < rep['dihedrals_considered']                    # zeros omitted
    bad = tmp_path/'bad.top'
    bad.write_text((tmp_path/'in.top').read_text().replace('  1  t1  1  M1  Si1  1  0.0  28.086',
                                                           '  1  t1  1  M1  Si1  1  0.0  14.007'))
    with pytest.raises(ValueError):
        apply(bad, tmp_path/'x.top', root/'benchmarks'/'ff'/'pdms_oplsaa_xiang2025.json')
