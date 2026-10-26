"""Multistate iterative Boltzmann inversion with fastMD sampling.

Modules:
  fastmd_io  fastMD input/output: LAMMPS data, table files, run configs, thermo and trajectories
  system     the CG system of one polymer from a mapped-AA reference (runs/cg_ref/<task>)
  tables     shared pair / bond / angle tables (Boltzmann-inverted start, fastMD table files)
Units throughout: nm, ps, amu, kJ/mol; fastMD temperatures are k_B T in kJ/mol.
"""
