"""Source-faithful RadonPy EQ21step schedule (stage ensembles, temperatures, pressures and lengths), used by the
GROMACS protocol (gmx_protocol).

The schedule is transcribed from RadonPy/RadonPy radonpy/sim/preset/eq.py (Equilibration.eq21step). It is a protocol
adapter, not a call to RadonPy's EQ21step.exec (which also runs its own packing and sampling). Some configuration
fields (SHAKE types, pair style, overlap relaxation) belong to the former LAMMPS renderer and are validated only.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

RADONPY_SOURCE = "https://github.com/RadonPy/RadonPy/blob/develop/radonpy/sim/preset/eq.py"
DEFAULT_STEP_LIST = (
    (50_000, 50_000, 50_000),
    (50_000, 100_000, 50_000),
    (50_000, 100_000, 50_000),
    (50_000, 100_000, 5_000),
    (5_000, 10_000, 5_000),
    (5_000, 10_000, 5_000),
    (5_000, 10_000, 800_000),
)
DEFAULT_PRESS_RATIOS = (0.02, 0.60, 1.00, 0.50, 0.10, 0.01)


@dataclass(frozen=True)
class AAStage:
    number: int
    block: int
    ensemble: str
    steps: int
    temperature_K: float
    pressure_atm: float | None
    neighbor_check: str | None = None


@dataclass(frozen=True)
class RadonPy21Config:
    temperature_K: float = 300.0
    high_temperature_K: float = 600.0
    pressure_atm: float = 1.0
    max_pressure_atm: float = 50_000.0
    timestep_fs: float = 1.0
    step_list: tuple[tuple[int, int, int], ...] = DEFAULT_STEP_LIST
    pressure_ratios: tuple[float, ...] = DEFAULT_PRESS_RATIOS
    thermostat_damp_fs: float = 100.0
    barostat_damp_fs: float = 1000.0
    production_steps: int = 1_000_000  # Separate paper-compatible holdout; RadonPy eq3 defaults to 5e6.
    production_stride: int = 1_000
    thermo_stride: int = 1_000
    seed: int = 42
    shake_bond_types: tuple[int, ...] = ()
    shake_angle_types: tuple[int, ...] = ()
    # Explicit nonbonded override, e.g. 'lj/cut/coul/long 15.0' (paper 15 A cutoff);
    # None keeps AutoPoly's oplsaa.lt 'lj/charmm/coul/long 9.0 11.0'.
    pair_style: str | None = None
    data_file: str = 'system.data'
    # False when per-atom charges come from the data file (e.g. a neutralized copy);
    # system.in.charges would reset them per type.
    include_charges: bool = True
    # Overlap removal before the schedule (not part of EQ21step): soft repulsion
    # ramped 0 -> overlap_relax_max_kcal over the run, bonded terms on, Coulomb off,
    # nve/limit + Langevin. AutoPoly's rigid-unit chain growth leaves atom clashes.
    # Per-atom neighbor capacity: at >1 g/cm3 with a 15 A cutoff + 2 A skin an atom has
    # ~2500 neighbors, above LAMMPS's default `one` of 2000 (overflow in the 30000 atm stage).
    neigh_one: int = 6000
    overlap_relax_steps: int = 0
    overlap_relax_cutoff_A: float = 2.5
    overlap_relax_max_kcal: float = 50.0
    overlap_relax_temperature_K: float = 300.0
    # Neighbor-list check in the last EQ21 stage (long NPT at the target pressure). RadonPy
    # uses 'no' (rebuild every step) for every NPT stage; at 1 atm the box changes slowly and
    # 'yes' (rebuild when an atom moved half the skin, as in production) is ~2.5x faster.
    # It changes performance only, not the dynamics.
    final_npt_neighbor_check: str = 'no'

    def __post_init__(self):
        if len(self.step_list) != 7 or any(len(triple) != 3 or any(not isinstance(x, int) or x <= 0 for x in triple)
                                            for triple in self.step_list):
            raise ValueError("RadonPy eq21step requires exactly seven positive integer step triples")
        if len(self.pressure_ratios) != 6 or any(x <= 0 for x in self.pressure_ratios):
            raise ValueError("RadonPy eq21step needs six positive pressure ratios")
        if min(self.temperature_K, self.high_temperature_K, self.pressure_atm, self.max_pressure_atm,
               self.timestep_fs, self.thermostat_damp_fs, self.barostat_damp_fs) <= 0:
            raise ValueError("Thermodynamic and integration parameters must be positive")
        if self.high_temperature_K < self.temperature_K:
            raise ValueError("High temperature cannot be below target temperature")
        # production_steps = 0: EQ21 only (the revised protocol branches 400/500 K from its restart)
        if self.production_steps < 0 or self.production_stride <= 0 or self.thermo_stride <= 0:
            raise ValueError("Production and output settings must be positive")
        if self.seed <= 0:
            raise ValueError("LAMMPS velocity RNG seed must be positive")
        if any(x <= 0 for x in self.shake_bond_types + self.shake_angle_types):
            raise ValueError("SHAKE interaction type IDs must be positive")
        if self.pair_style is not None and not re.fullmatch(r'lj/[a-z/]+coul/long [0-9. ]+', self.pair_style):
            raise ValueError("pair_style override must be an lj/.../coul/long style with cutoffs")
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', self.data_file):
            raise ValueError("data_file must be a file name inside the system directory")
        if self.overlap_relax_steps < 0:
            raise ValueError("overlap_relax_steps must be >= 0")
        if self.overlap_relax_steps and self.pair_style is None:
            raise ValueError("overlap relaxation needs an explicit pair_style to restore afterwards")
        if self.final_npt_neighbor_check not in ('yes', 'no'):
            raise ValueError("final_npt_neighbor_check must be 'yes' or 'no'")


def stages(config: RadonPy21Config = RadonPy21Config()) -> tuple[AAStage, ...]:
    """Exact default ensemble/temperature/pressure/step sequence in RadonPy EQ21step."""
    ps = tuple(x * config.max_pressure_atm for x in config.pressure_ratios) + (config.pressure_atm,)
    out = []
    for block, (triple, p) in enumerate(zip(config.step_list, ps), start=1):
        hot, cool, press = triple
        last = block == len(config.step_list)
        out.extend((
            AAStage(len(out)+1, block, "nvt", hot, config.high_temperature_K, None, "yes"),
            AAStage(len(out)+2, block, "nvt", cool, config.temperature_K, None, None),
            AAStage(len(out)+3, block, "npt", press, config.temperature_K, p,
                    config.final_npt_neighbor_check if last else "no"),
        ))
    return tuple(out)
