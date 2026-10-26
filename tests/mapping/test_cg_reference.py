"""map-aa helpers: whole molecules across periodic boundaries, GROMACS energy files."""
import numpy as np


def test_make_whole_and_xvg(tmp_path):
    from ascg_core.mapping.cg_reference import make_whole, read_xvg, whole_plan
    box = np.array([10.0, 10.0, 10.0])
    chain = np.array([[9.5, 5, 5], [0.5, 5, 5], [1.5, 9.9, 5], [2.0, 0.2, 5]])    # wrapped across x and y
    plan = whole_plan([(0, 1), (1, 2), (2, 3)], 4)
    w = make_whole(chain, box, plan)
    assert np.allclose(w[:, 0], [9.5, 10.5, 11.5, 12.0]) and np.isclose(w[3, 1], 10.2)
    (tmp_path/'e.xvg').write_text('# gmx\n@ s0 legend "Temperature"\n@ s1 legend "Pressure"\n'
                                  '@ s2 legend "Volume"\n@ s3 legend "Density"\n'
                                  '0 400 1.01325 64.0 900\n1 401 -1.01325 64.2 898\n')
    t = read_xvg(tmp_path/'e.xvg')
    assert np.allclose(t['Press'], [1, -1]) and np.allclose(t['Volume_nm3'], [64.0, 64.2]) and t['Temp'][1] == 401
