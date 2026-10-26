"""GROMACS protocol files (soft-core start, EQ21, branches)."""
import json


def test_gromacs_protocol_files(tmp_path):
    from ascg_core.aa.gmx_protocol import write_protocol, status
    sysd = tmp_path/'gmx'
    sysd.mkdir()
    (sysd/'system.top').write_text('[ atomtypes ]\n  t1  1.008  0.0  A  0.25  0.1\n[ atoms ]\n'
                                   '  1  t1  1  M1  H1  1  0.4  1.0080\n')
    (sysd/'system.gro').write_text('x\n')
    write_protocol(sysd, tmp_path/'pe')
    eq = json.loads((tmp_path/'pe'/'eq21'/'steps.json').read_text())
    names = [s['name'] for s in eq['steps']]
    assert names == ['em_sc', 'relax_sc', 'em'] + [f'stage_{k:02d}' for k in range(1, 22)]
    assert eq['steps'][0]['topology'].endswith('system_sc.top') and eq['steps'][2]['topology'].endswith('system.top')
    sc = (sysd/'system_sc.top').read_text()
    assert '  t1_B  1.008  0.0  A  0.25  0.0' in sc and '0.4  1.0080  t1_B  0.0  1.0080' in sc
    mdp = {n: dict(l.split('=', 1) for l in (tmp_path/'pe'/'eq21'/f'{n}.mdp').read_text().splitlines()[1:])
           for n in names}
    mdp = {n: {k.strip(): v.strip() for k, v in d.items()} for n, d in mdp.items()}
    assert mdp['em']['integrator'] == 'steep' and mdp['em']['constraints'] == 'none'
    s1, s9, s21 = mdp['stage_01'], mdp['stage_09'], mdp['stage_21']
    assert s1['gen-vel'] == 'yes' and s1['ref-t'] == '600.0' and s1['pcoupl'] == 'no' and s1['nsteps'] == '50000'
    assert all(mdp[n]['constraints'] == 'none' for n in names)
    assert s9['pcoupl'] == 'C-rescale' and s9['ref-p'] == '50662.5' and s9['tcoupl'] == 'v-rescale'
    assert s21['ref-p'] == '1.0132' and s21['nsteps'] == '800000' and s21['continuation'] == 'yes'
    assert all(mdp[n]['dt'] == '0.001' for n in names[3:]) and mdp['relax_sc']['dt'] == '0.001'
    assert all(mdp[n]['verlet-buffer-tolerance'] == '-1' and mdp[n]['rlist'] == '1.6' for n in names[:3])
    assert all(mdp[n]['rvdw'] == '1.2' and mdp[n]['rlist'] == '1.3' for n in names[3:-1])     # stages 1-20
    assert s21['rvdw'] == '1.5' and s21['rcoulomb'] == '1.5' and 'rlist' not in s21 and 'nstlist' not in s21
    r = mdp['relax_sc']
    assert r['free-energy'] == 'yes' and r['init-lambda'] == '0.95' and abs(float(r['delta-lambda']) * 20000 + 0.95) < 1e-12
    assert r['ref-t'] == '300.0' and r['sc-coul'] == 'yes' and 'free-energy' not in mdp['em']
    assert eq['steps'][3]['conf'].endswith('em.gro') and eq['steps'][3]['cpt'] is None
    assert eq['steps'][4]['cpt'].endswith('stage_01.cpt') and eq['steps'][2]['conf'].endswith('relax_sc.gro')
    for T in (400, 500):
        b = json.loads((tmp_path/'pe'/f'T{T}'/'steps.json').read_text())
        assert [s['name'] for s in b['steps']] == ['pre', 'eq', 'prod']
        assert b['steps'][0]['conf'].endswith('eq21/stage_21.gro') and b['steps'][0]['cpt'] is None
        assert b['steps'][1]['cpt'].endswith('pre.cpt') and b['steps'][2]['cpt'].endswith('eq.cpt')
        pre, e, p = ((tmp_path/'pe'/f'T{T}'/f'{n}.mdp').read_text() for n in ('pre', 'eq', 'prod'))
        assert 'C-rescale' in pre and 'v-rescale' in pre and 'gen-vel                 = yes' in pre
        assert 'nsteps                  = 1000000' in pre
        for text in (e, p):
            assert 'nose-hoover' in text and 'Parrinello-Rahman' in text and f'= {float(T)}' in text
            assert 'nsteps                  = 10000000' in text and 'constraints             = none' in text
        assert 'gen-vel                 = no' in e and 'continuation            = yes' in e and 'nstxout-compressed      = 0' in e
        for text in (e, p):
            assert 'nstpcouple              = 3\n' in text and 'nsttcouple              = 3\n' in text
            assert 'nstcalcenergy           = 3\n' in text
        assert 'nstxout-compressed      = 10000' in p and 'nstenergy               = 999' in p
    assert status(tmp_path/'pe'/'T400') == {'steps': 3, 'done': 0, 'next': 'pre'}
