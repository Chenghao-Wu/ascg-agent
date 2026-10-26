from ascg_core.aa.lmp2gmx import convert, one_four_pairs

DATA = """LAMMPS data file via write_data

6 atoms
2 atom types
5 bonds
1 bond types
4 angles
1 angle types
3 dihedrals
1 dihedral types
1 impropers
1 improper types

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 12.011
2 1.008

Pair Coeffs # lj/cut/coul/long

1 0.066 3.5
2 0.03 2.5

Bond Coeffs # harmonic

1 268 1.529

Angle Coeffs # harmonic

1 58.35 112.7

Dihedral Coeffs # opls

1 1.3 -0.05 0.2 0

Improper Coeffs # cvff

1 2.5 -1 2

Atoms # full

1 1 1 -0.18 1.0 1.0 1.0 0 0 0
2 1 1 -0.12 2.5 1.0 1.0 0 0 0
3 1 1 -0.12 3.0 2.4 1.0 0 0 0
4 1 1 -0.18 4.5 2.4 1.0 0 0 0
5 1 1 0.3 19.5 2.4 1.0 -1 0 0
6 1 2 0.3 6.0 2.4 1.0 0 0 0

Bonds

1 1 1 2
2 1 2 3
3 1 3 4
4 1 4 5
5 1 4 6

Angles

1 1 1 2 3
2 1 2 3 4
3 1 3 4 5
4 1 3 4 6

Dihedrals

1 1 1 2 3 4
2 1 2 3 4 5
3 1 2 3 4 6

Impropers

1 1 4 3 5 6
"""


def test_units_pairs_and_unwrapping(tmp_path):
    (tmp_path/'x.data').write_text(DATA)
    info = convert(tmp_path/'x.data', tmp_path/'x.gro', tmp_path/'x.top')
    assert info == {'atoms': 6, 'molecules': 1, 'pairs14': 3, 'box_nm': [2.0, 2.0, 2.0]}
    top = (tmp_path/'x.top').read_text()
    assert '  1  3  yes  0.5  0.5' in top                                         # geometric, 1-4 x 0.5
    assert 't1  12.0110  0.0  A  0.350000  0.276144' in top                       # sigma nm, eps kJ/mol
    assert '  1  2  1  0.152900  224262.4000' in top                             # kb = 2 K * 418.4
    assert '  1  2  3  1  112.700000  488.272800' in top                         # kt = 2 K * 4.184
    assert '  1  2  3  4  5  5.439200  -0.209200  0.836800  0.000000' in top      # Fourier C_n = K_n
    assert '  4  3  5  6  4  180.0  10.460000  2' in top                         # cvff d=-1 -> phi_s 180
    gro = (tmp_path/'x.gro').read_text().splitlines()
    # atom 5 has image -1 in x: unwrapped to 19.5 - 20 = -0.5 A -> -0.05 nm (molecule whole)
    assert gro[2 + 4][20:30].strip() == '-0.05000'


def test_one_four_pairs_ring_counts_shortest_path():
    ring = [(i, (i + 1) % 6) for i in range(6)]                                  # benzene skeleton
    assert one_four_pairs(6, ring) == [(0, 3), (1, 4), (2, 5)]                   # para pairs only


def test_coefficients_from_settings_file(tmp_path):
    """AutoPoly data files carry no Coeffs sections; moltemplate's system.in.settings does."""
    import re
    head, tail = DATA.split('Atoms # full')
    body = re.sub(r'\n(Pair|Bond|Angle|Dihedral|Improper) Coeffs # [a-z/]+\n\n(?:[^\n]+\n)+', '\n', head)
    (tmp_path/'x.data').write_text(body + 'Atoms # full' + tail)
    (tmp_path/'s.in').write_text('pair_coeff 1 1 0.066 3.5\npair_coeff 2 2 0.03 2.5\nbond_coeff 1 268 1.529\n'
                                  'angle_coeff 1 58.35 112.7\ndihedral_coeff 1 1.3 -0.05 0.2 0 # c\n'
                                  'improper_coeff 1 2.5 -1 2\n')
    convert(tmp_path/'x.data', tmp_path/'a.gro', tmp_path/'a.top', settings=tmp_path/'s.in')
    (tmp_path/'y.data').write_text(DATA)
    convert(tmp_path/'y.data', tmp_path/'b.gro', tmp_path/'b.top')
    strip = lambda f: [l for l in f.read_text().splitlines() if not l.startswith('; generated')]  # noqa: E731
    assert strip(tmp_path/'a.top') == strip(tmp_path/'b.top')


def test_polar_hydrogens_get_small_lj(tmp_path):
    from ascg_core.aa.lmp2gmx import POLAR_H_LJ
    data = DATA.replace('2 0.03 2.5', '2 0.0 0.0')            # type 2 (H) without LJ, as OPLS H(O)
    (tmp_path/'x.data').write_text(data)
    info = convert(tmp_path/'x.data', tmp_path/'x.gro', tmp_path/'x.top', polar_h_lj=POLAR_H_LJ)
    assert info['polar_h_lj'] == {'types': [2], 'atoms': 1, 'sigma_nm': 0.1, 'epsilon_kJ_mol': 0.1925}
    top = (tmp_path/'x.top').read_text()
    assert 't2  1.0080  0.0  A  0.100000  0.192500' in top and 't1  12.0110  0.0  A  0.350000  0.276144' in top
    convert(tmp_path/'x.data', tmp_path/'y.gro', tmp_path/'y.top')
    assert 't2  1.0080  0.0  A  0.000000  0.000000' in (tmp_path/'y.top').read_text()   # default: OPLS as is
