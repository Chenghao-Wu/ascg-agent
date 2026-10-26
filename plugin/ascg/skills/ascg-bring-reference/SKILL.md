---
name: ascg-bring-reference
description: Use the user's own all-atom simulation as the reference (GROMACS, LAMMPS or any MDAnalysis-readable topology and trajectory): inspect the files, agree the state points and the mapping (atom names per residue or SMILES), build the mapped reference with ascg ref build and register it with aa import. Use when the user already has AA trajectories.
---
# Bring your own AA reference

1. **Ask what they have.** For each state point the user wants the model to reproduce:
   - a topology with bonds: `.tpr`, `.top`, LAMMPS data file, PSF;
   - the production trajectory: `.xtc`, `.trr`, `.dcd`, LAMMPS dump; several segments are fine;
   - an energy file if there is one: `gmx energy` `.xvg`, `.edr`, or the LAMMPS log;
   - the temperature and pressure of that run;
   - which compute target holds the files.

   The state points are the user's decision; never infer them from the files alone.
2. **Inspect the files** before anything else:
   `ascg ref inspect --target X --topology /abs/topol.tpr --trajectory /abs/prod.xtc --energy /abs/energy.xvg`.
   - Show the user the molecules (count and formula), residue names, frames, time span and frame spacing, NVT or NPT,
     density, and the mean T and P.
   - **Stop and ask the user before building** when the files contradict what they said:
     - mean T more than about 1 % off the stated temperature;
     - mean P far from the stated pressure, beyond its fluctuation;
     - a density that is not a liquid or melt (below about 0.3 g/cm3);
     - the wrong molecules, or a trajectory far too short.
     Do not build or register until the user confirms the files or gives the right ones.
   - LAMMPS dumps without a log need `--timestep-fs`.
3. **Mapping** (ascg-mapping, `references/formats.md`):
   - With residue and atom names (GROMACS, PSF): the `residue-atom-names` form, beads by atom names per residue.
     Use the names `ref inspect` lists; hydrogens that are not listed follow their heavy atom; polymers need one
     residue per monomer, with the end residues listed too.
   - Without names (LAMMPS data files): the annotated-smiles form, molecules only. Every molecule is matched by its
     heavy-atom graph.
   - Try the mapping on the real topology: `ascg ref inspect ... --mapping '<json>'` gives beads and bead masses per
     molecule type and the bonded types. Show it to the user.
   - Record it: `ascg map save --study S --key K --spec '<json>' --system-class ...`, then, after the user explicitly
     approves it, `ascg map approve --study S --key K --note "<their words>"`.
4. **Build the reference:**
   `ascg ref build --study S --key K --target X --topology /abs/topol.tpr --run 300=/abs/prod.xtc,/abs/energy.xvg`.
   - Give one `--run STATE=TRAJECTORY[+SEGMENT][,ENERGY]` per approved state point.
   - Give `--start-ps` when the trajectory still contains equilibration.
   - Give `--stride` when it has many more than about 1000 frames.
   - The tool refuses a mapping the user has not approved.
5. **Register it:** `ascg aa import --study S --task K --target X --ref <ref from ref build> --states <the approved
   states>`.
   - Tell the user every QA flag:
     - **density drift:** the run may not be equilibrated;
     - **pressure off the state:** the label is wrong, or there was no barostat;
     - **chain relaxation:** Rg drifts.
   - NVT data give no compressibility.
   - Then continue with the targets review and the fit (ascg-study).
