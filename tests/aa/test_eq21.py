"""RadonPy EQ21 schedule used by the GROMACS protocol."""
import pytest

from ascg_core.aa.eq21 import RadonPy21Config, stages


def test_radonpy_exact_schedule():
    p=stages()
    assert len(p)==21
    assert [s.ensemble for s in p]==['nvt','nvt','npt']*7
    assert [s.pressure_atm for s in p if s.ensemble=='npt']==[1000,30000,50000,25000,5000,500,1]
    assert [s.steps for s in p if s.ensemble=='npt']==[50000,50000,50000,5000,5000,5000,800000]
    assert sum(s.steps for s in p)==1560000
    assert [s.temperature_K for s in p[:3]]==[600,300,300]


def test_custom_pressures_and_steps():
    cfg=RadonPy21Config(pressure_atm=5, max_pressure_atm=10000, timestep_fs=2.0)
    assert [s.pressure_atm for s in stages(cfg) if s.ensemble=='npt']==[200,6000,10000,5000,1000,100,5]


def test_invalid_configs():
    with pytest.raises(ValueError): RadonPy21Config(step_list=((1,2,3),))
    with pytest.raises(ValueError): RadonPy21Config(pressure_ratios=(1.0,))
    with pytest.raises(ValueError): RadonPy21Config(shake_bond_types=(-1,))
