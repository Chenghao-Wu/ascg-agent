"""The CG system of one polymer/mapping task, from a mapped-AA reference directory (runs/cg_ref/<task>,
written by `ascg map-aa`: topology.json, summary.json, targets_T*.npz, start_T*.npz).

Bead types, bonds and angles follow the reference topology; bond / angle types are named like the
targets ('A-B', 'A-B-C', canonical order). Nonbonded exclusions are the reference's 1-2 and 1-3 pairs
(`topology['exclusions']`); the 1-3 pairs that are not bonds go to fastMD as bonds of an extra
zero-energy type. fastMD masses are per type: the per-type mean bead mass is used (end beads that
carry an extra H differ by ~1 amu, which leaves equilibrium structure unchanged); densities are
always computed from the true bead masses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import fastmd_io as fio

AMU_PER_NM3_TO_G_CM3 = 1.66053906660e-3


@dataclass
class CGSystem:
    task: str
    type_names: list
    types: np.ndarray            # [n] 0-based bead types
    masses: np.ndarray           # [n] true bead masses (amu)
    mol: np.ndarray              # [n] molecule index
    bonds: np.ndarray            # [nb, 2]
    bond_type: np.ndarray        # [nb] index into bond_types
    bond_types: list
    angles: np.ndarray           # [na, 3]
    angle_type: np.ndarray       # [na]
    angle_types: list
    excl13: np.ndarray           # [n13, 2] 1-3 pairs that are not bonds
    pair_keys: list              # 'A-A', 'A-B', ... in combinations_with_replacement order
    pair_index: list             # [(ti, tj)] matching pair_keys
    dihedrals: np.ndarray = None    # [nd, 4] when tabulated dihedrals are used (ibi.dihedrals), else None
    dihedral_type: np.ndarray = None
    dihedral_types: list = None
    dihedral_style: str = None      # 'table' (plain U(phi)) or 'cbt' (sin^3 theta1 sin^3 theta2 U(phi))

    @property
    def n(self):
        return len(self.types)

    @property
    def type_masses(self):
        return np.array([self.masses[self.types == t].mean() for t in range(len(self.type_names))])

    @property
    def total_mass(self):
        return float(self.masses.sum())

    def density_g_cm3(self, volume_nm3):
        return self.total_mass * AMU_PER_NM3_TO_G_CM3 / np.asarray(volume_nm3, float)

    def write_data(self, path, pos, box, *, velocities=None):
        """fastMD data file: real bonds (types 0..nbt-1), 1-3 exclusion bonds (type nbt), angles."""
        nbt = len(self.bond_types)
        bonds = np.concatenate([self.bonds, self.excl13]) if len(self.excl13) else self.bonds
        btypes = np.concatenate([self.bond_type, np.full(len(self.excl13), nbt)]) if len(self.excl13) else self.bond_type
        dih = {} if self.dihedrals is None else dict(dihedrals=self.dihedrals, dihedral_types=self.dihedral_type,
                                                       n_dihedral_types=len(self.dihedral_types))
        fio.write_data(path, pos=pos, box=box, types=self.types, type_masses=self.type_masses, mol=self.mol,
                       bonds=bonds, bond_types=btypes, n_bond_types=nbt + (1 if len(self.excl13) else 0),
                       angles=self.angles, angle_types=self.angle_type, n_angle_types=len(self.angle_types),
                       velocities=velocities, title=f'ascg CG {self.task}', **dih)

    def add_dihedrals(self, style='cbt'):
        """Enumerate the dihedrals (paths of three bonds) and their types; style 'table' or 'cbt' (ibi.dihedrals)."""
        from .dihedrals import STYLES, enumerate_dihedrals
        if style not in STYLES:
            raise ValueError(f'dihedral style {style!r} not in {STYLES}')
        self.dihedrals, self.dihedral_type, self.dihedral_types = enumerate_dihedrals(self)
        self.dihedral_style = style
        return self


def load_system(ref_dir) -> CGSystem:
    ref = Path(ref_dir)
    topo = json.loads((ref/'topology.json').read_text())
    summary = json.loads((ref/'summary.json').read_text())
    names = list(topo['type_names'])
    types = np.asarray(topo['bead_types'], int)
    tname = [names[t] for t in types]
    bonds = np.asarray(topo['bonds'], int).reshape(-1, 2)
    angles = np.asarray(topo['angles'], int).reshape(-1, 3)
    bond_types, angle_types = list(summary['bond_types']), list(summary['angle_types'])
    btype = np.array([bond_types.index('-'.join(sorted((tname[a], tname[b])))) for a, b in bonds], int)
    atype = np.array([angle_types.index('-'.join(min([tname[a], tname[b], tname[c]], [tname[c], tname[b], tname[a]])))
                      for a, b, c in angles], int) if len(angles) else np.zeros(0, int)
    bonded = {tuple(sorted(map(int, b))) for b in bonds}
    excl = {tuple(sorted(map(int, e))) for e in topo['exclusions']}
    if not bonded <= excl:
        raise ValueError(f'{ref}: some bonds are missing from the exclusion list')
    excl13 = np.array(sorted(excl - bonded), int).reshape(-1, 2)
    pairs = [(a, b) for a in range(len(names)) for b in range(a, len(names))]
    return CGSystem(task=summary.get('task', ref.name), type_names=names, types=types,
                    masses=np.asarray(topo['bead_masses_amu'], float), mol=np.asarray(topo['bead_molecule'], int),
                    bonds=bonds, bond_type=btype, bond_types=bond_types, angles=angles, angle_type=atype,
                    angle_types=angle_types, excl13=excl13, pair_keys=[f'{names[a]}-{names[b]}' for a, b in pairs],
                    pair_index=pairs)


def load_start(ref_dir, temperature_K):
    """Mapped-AA start configuration (unwrapped positions, nm) and the AA mean box at T."""
    from .states import label
    s = np.load(Path(ref_dir)/f'start_T{label(temperature_K)}.npz')
    return np.asarray(s['pos_nm'], float), np.asarray(s['box_nm'], float)
