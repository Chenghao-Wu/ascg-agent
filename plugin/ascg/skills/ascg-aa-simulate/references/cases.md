# Atomistic simulation cases (GROMACS reference production)

General patterns: signature -> decision. Systems are deliberately not named.

| Case | Signature | Decision |
|---|---|---|
| Engine choice | GROMACS vs LAMMPS on the same system | measure ns/day first; GROMACS on CPUs / GPUs by measured speed |
| Neighbor-list overflow (LAMMPS) | crash at tens of katm: more neighbors than the default page | larger neighbor pages |
| Cut-off vs box during compression | the box shrinks below 2 x rlist at very high pressure | shorter cut-off for the compression stages, full cut-off afterwards |
| Barostat runaway | box grows many-fold in 100 ps; Parrinello-Rahman every 10 steps resonating with X-H stretch (1 fs) or C-C stretch (2 fs) | couple every 1-3 steps; stochastic (C-rescale) pre-equilibration |
| Constraints / time step | hydroxyl ends rotating > 30 deg per step at 2 fs with H-bond constraints | 1 fs, flexible bonds |
| Slow NPT stages | neighbor list rebuilt every step | rebuild on displacement |
| Missed pairs in dilute boxes | crash in hot stages of a dilute start box | fixed pair-list buffer: verlet-buffer-tolerance -1, rlist 1.6 nm, nstlist 10 |
| Dilute liquid box | compressing at 1 atm is slow (vapour condenses into droplets) | direct equilibration: NVT, C-rescale at 1000 atm (liquid density in ~0.1 ns), then the state pressure |
| Wall time | runs longer than the queue limit | restart segments |
| Node failure | all jobs on one node die within minutes | resubmit from the last restart; avoid the node |
| Chain size changes strongly between states | Rg changes by > 10 % between two melt states | check Tm / Tg against the lower state; flag before CG fitting |
| Glassy reference state | state temperature near or below the (oligomer) Tg | flag to the user; check Rg / density drift |
| System size | inconsistent bead counts across systems | ~1000 CG beads, >= 20 chains; liquids ~1000 molecules |
