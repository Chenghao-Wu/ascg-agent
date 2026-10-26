# Third-party components

ascg-agent is MIT-licensed. It calls, imports or follows the following; none of them is redistributed here, except
the files listed at the end.

| Component | Use | Licence |
|---|---|---|
| [fastMD](https://github.com/Chenghao-Wu/fastMD) | CG molecular dynamics engine (external program and its numpy-only Python reader) | MIT |
| [AutoPoly v2.0](https://github.com/WuGroup-XJTLU/AutoPoly) | AA polymer and liquid builds, OPLS-AA typing | MIT |
| [moltemplate](https://github.com/jewettaij/moltemplate) OPLS-AA data (through AutoPoly) | OPLS-AA atom types and parameters | MIT |
| [GROMACS](https://www.gromacs.org) | AA equilibration and production (external program) | LGPL-2.1 |
| [RadonPy](https://github.com/RadonPy/RadonPy) | The EQ21 equilibration schedule of the polymer AA references (transcribed; RadonPy is not imported) | BSD-3-Clause |
| [RDKit](https://www.rdkit.org) | Mappings from SMILES, mapping checks, bead assignment | BSD-3-Clause |
| [MDAnalysis](https://www.mdanalysis.org) | Reading AA trajectories (the `aa` extra) | LGPL-3.0+ |
| numpy, scipy, matplotlib, networkx, PyYAML | Numerics, plotting, graphs, compute profiles | BSD-style, PSF (matplotlib), MIT (PyYAML) |

Files in this repository derived from third-party code or data:
- `src/ascg_core/aa/eq21.py`: the EQ21 schedule (stage ensembles, temperatures, pressure ratios and stage lengths)
  transcribed from RadonPy's `radonpy/sim/preset/eq.py` (Copyright (c) 2026, RadonPy developers; BSD-3-Clause). Cite
  Y. Hayashi, J. Shiomi, J. Morikawa, R. Yoshida, npj Comput. Mater. 8, 222 (2022) when using it.
- `patches/opls_lt_2024_ascg_additions.fdefn`: typing rules added to AutoPoly; every type and charge is a standard
  OPLS-AA entry of moltemplate's `oplsaa2024.lt`.
- `benchmarks/ff/pdms_oplsaa_xiang2025.json`: OPLS-AA/Qeq parameters for PDMS from Xiang et al., J. Phys. Chem. B
  (2025), as published in the PDMS_benchmark repository; cite that paper when using them.
