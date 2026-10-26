---
name: ascg-setup-engine
description: Install and check fastMD with tabulated and combined bending-torsion dihedrals for ascg: build plan, ctest, and the deterministic engine selfcheck.
---
# fastMD engine

ascg needs fastMD main b35e30a or later (Chenghao-Wu/fastMD): `dihedral_table ... [cbt]` and the NPT barostat fix.
`ascg compute doctor --target T` checks a host in one go (connection, Python dependencies, deployed code, fastMD
build and version, GPUs, engine gate).

1. `ascg compute test --target T` shows `fastmd_present` and the binary path (FASTMD_BINARY from the env line).
2. Missing or old build: `ascg engine setup --target T --cuda-arch "86;89"` prints the build commands. Building on a
   shared machine is the user's decision: show the commands, run them only if the user asks.
3. `ascg engine selfcheck --target T` on workstations: fastMD energy of a toy melt must equal numpy with 1-2 / 1-3
   exclusions, the CBT dihedral energy must equal numpy, and a short NPT run of a CG liquid of small stiff molecules
   must keep its volume (builds without the barostat fix collapse). Every ascg job repeats this check before fitting.
4. A failing selfcheck is an infrastructure problem (wrong build, wrong GPU architecture, broken environment), never
   a reason to change the model.
