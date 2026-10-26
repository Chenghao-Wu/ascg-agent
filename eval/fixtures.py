"""Data a task needs, written into each run's data/ directory before the agent starts (outside its workspace, so it
cannot change them). Synthetic only: no host or user data, so every task runs anywhere."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def byo_pentane(dst, *, n_side=3, box_A=40.0, n_frames=6, npt=False, thermo='ramp', formats=('gromacs',), seed=1):
    """Liquid-style n-pentane AA data (RDKit geometry, rigid random moves; tests/mapping/byo_system.py): GROMACS
    system.top + traj.xtc + energy.xvg in gromacs/, LAMMPS system.data + traj.lammpstrj + log.lammps in lammps/.
    The defaults give 27 molecules in 64 nm3 (a gas) with temperature and pressure climbing by 1 per frame; a dense
    liquid is n_side=6, box_A=34.7 (216 molecules, 0.62 g/cm3), with npt=True (the box breathes by 0.2 % in length)
    and thermo='noisy' (T and P fluctuate around 300 K and 1 atm)."""
    sys.path.insert(0, str(ROOT/'tests'/'mapping'))
    try:
        import byo_system as bs
    finally:
        sys.path.pop(0)
    s = bs.build('CCCCC', n_side=n_side, box_A=box_A, n_frames=n_frames, seed=seed, npt=npt, npt_amp=0.002)
    out = {}
    if 'gromacs' in formats:
        bs.write_gromacs(s, Path(dst)/'gromacs', thermo=thermo)
        out['gromacs'] = str(Path(dst)/'gromacs')
    if 'lammps' in formats:
        bs.write_lammps(s, Path(dst)/'lammps', thermo=thermo)
        out['lammps'] = str(Path(dst)/'lammps')
    return out


FIXTURES = {'byo_pentane': byo_pentane}


def make(spec, dst):
    """spec: {"kind": name, "args": {...}} -> what the fixture wrote (paths)."""
    Path(dst).mkdir(parents=True, exist_ok=True)
    return FIXTURES[spec['kind']](dst, **spec.get('args', {}))
