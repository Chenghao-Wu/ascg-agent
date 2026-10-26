---
name: ascg-aa-simulate
description: Plan, run or import atomistic reference simulations at every approved state point (T, P) and check them before use: equilibration (polymers - RadonPy 21-step schedule; liquids - direct compression), production, engine settings, failure handling, reference QA.
---
# Atomistic references

**Import first.** If mapped references exist (topology.json, summary.json, targets_T*.npz, mapped_T*.npz,
start_T*.npz), register them with the approved state points:
`ascg aa import --study S --task K --target T --ref /abs/path --states 400,500`.

Any approved set works:
- `--states 450` for one state;
- `--states 350,400,450`;
- `--states 400,400@1000` for 400 K at 1 and 1000 atm;
- labels such as T400_P1000.

Every approved state needs a reference at that (T, P). States of the reference that the user did not approve are not
fitted; the tool lists them. Approved states without a reference are refused: those need AA simulations first (below,
with the user's approval).

`aa import` runs preflight and QA. Tell the user about every QA flag:
- *state dependence of chain size* (Rg changes by > 10 % between two melt states): the lower state may be below the
  melting point or glassy, or not equilibrated, and the CG fit will likely miss chain size there;
- *chain relaxation* (Rg drift across the production) and *density drift*: equilibration incomplete;
- *analysis range* beyond half the box: system too small;
- *pressure*: the AA run's mean pressure is not the state's pressure, from a wrong label or barostat; do not fit it.

**Run new references** for a system saved and approved in the study (`ascg system save` / `approve`,
ascg-specify-system), on a target with AA settings (`compute doctor` lists AutoPoly and GROMACS). The AA engine is
GROMACS (fastMD is never used for AA).
1. `ascg aa build --study S --task K --target T`: AutoPoly build, neutralization, typing audit and CG topology, as one
   job (minutes). `ascg aa status --study S --task K` until `built`.
2. Read the build with the user: atoms, molecules, net charge, the audit verdict and every finding (ascg-aa-build).
   Findings block step 3 until they are resolved with the user.
3. `ascg aa submit --study S --task K --states <approved> --dry-run`: the state points, run lengths and the estimated
   core-hours and wall time. Show them; the user decides. Then the same command with `--approval "<their words>"`
   instead of `--dry-run`. Never shorten the runs (`--eq-ns`, `--prod-ns`) unless the user asks; say so in reports.
4. Wait with `ascg study watch --study S` (AA events) or `ascg aa status`; it shows each directory's progress and
   ns/day. After a time limit or a node failure (`md_failed`, log tail in the jobs), `ascg aa submit ... --resume`
   continues from the checkpoints with the approved plan.
5. `ascg aa map --study S --task K` (the study's approved mapping), then `ascg aa import` with the reference path that
   `aa status` gives and the approved states.
6. `ascg aa cancel --study S --task K` stops this campaign's jobs only.

The protocol the tools write (`ascg aa plan --task K --study S --states ...` shows it as commands):
- **Polymers:**
  - dilute build and soft-core relaxation;
  - the RadonPy 21-step compression / decompression schedule as GROMACS stages, at the hottest approved
    temperature;
  - one branch per approved state (C-rescale pre-equilibration at (T, P), then 10 ns Nose-Hoover +
    Parrinello-Rahman equilibration and 10 ns production).
- **Molecular liquids and mixtures:** `prepare-gmx-protocol --equilibration direct`:
  - soft-core relaxation and minimization;
  - 100 ps NVT;
  - 0.5 ns C-rescale at 1000 atm (the dilute box reaches liquid density in ~0.1 ns);
  - 0.5 ns at the state pressure;
  - the branch, with 4 ns equilibration and 10 ns production. One state (e.g. 300 K, 1 atm) is typical.
- **Pressure:** GROMACS pressures are in bar (bar = atm x 1.01325); record both. A branch at another pressure is
  `--states 400@1000`.

**Check before use:**
- glassy or semi-crystalline states (state temperature near or below the Tg / Tm of the chains simulated; short
  chains have a lower Tg): tell the user the reference may not be an equilibrium melt;
- liquid densities against experiment for the force field used: more than ~5 % off -> check the force field and
  cut-off treatment before fitting.

On Slurm targets the equilibration is one job and each branch one job after it; on a workstation (or in GPU mode) one
job runs them in turn. Never submit AA jobs by raw ssh.

Engine failures seen in practice (references/cases.md):
- neighbor-list overflow at high pressure;
- cut-off larger than half the box during compression;
- barostat runaway when the coupling interval resonates with bond vibrations;
- crashes of hydroxyl ends at 2 fs;
- slow NPT stages that rebuild neighbor lists every step;
- missed pairs in dilute boxes (fixed pair-list buffer);
- wall-time limits (use restart segments);
- node failures (resubmit from the last restart).
