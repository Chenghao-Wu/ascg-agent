# Bring your own AA trajectory

If you already have all-atom simulations of your system, ascg maps them onto beads and uses them as the reference.
This is the fastest way in: no AA builds, no AA runs. With an agent, describe what you have:

> I have an all-atom GROMACS run of liquid n-pentane at 300 K and 1 atm on target "local": topology
> /data/pentane/topol.tpr, trajectory /data/pentane/prod.xtc, energies /data/pentane/energy.edr. I want a
> three-bead model, CH3-CH2 | CH2 | CH2-CH3. Prepare the reference in a study called pentane.

The agent inspects the files, checks them against what you said, proposes the mapping for your approval, builds the
reference and registers it. It stops and asks when the files contradict your description (density, temperature,
pressure, molecules, length).

## What you need

Per state point (temperature, pressure) the model should reproduce:

| | Accepted |
|---|---|
| Topology, with bonds | GROMACS `.tpr` or `.top` (+ `.itp`), LAMMPS data file, PSF, or anything MDAnalysis reads with bonds |
| Production trajectory | `.xtc`, `.trr`, `.dcd`, LAMMPS dump, ...; several segments are fine |
| Energies (optional, recommended) | `gmx energy` `.xvg`, `.edr`, the LAMMPS log (`units real` or `metal`) |

- The files sit on a compute target (`--target`), and that target's Python needs MDAnalysis (`pip install -e '.[aa]'`
  there, or `ascg compute deploy --target T --install-deps`).
- NPT runs give the compressibility too; NVT runs are accepted, without it.
- About 500-1000 frames per state over a few ns is plenty; thin longer runs with `--stride`, drop equilibration with
  `--start-ps`.

## The steps (what the agent runs)

1. **Inspect** (read-only): atoms, bonds, molecules by formula, residue and atom names, frames, time span, box (NVT /
   NPT), density, energy averages; with `--mapping`, also the beads and bonded types per molecule type.

   ```bash
   ascg ref inspect --target local --topology /data/pentane/topol.tpr --trajectory /data/pentane/prod.xtc \
       --energy /data/pentane/energy.edr
   ```

2. **Mapping**, in one of two forms, checked before anything is saved:
   - **atom names per residue**, for topologies with names (`.tpr`, `.top`, PSF):

     ```json
     {"format": "residue-atom-names",
      "residues": {"PNT": [{"type": "END", "atoms": ["C1", "C2"]},
                           {"type": "MID", "atoms": ["C3"]},
                           {"type": "END", "atoms": ["C4", "C5"]}]}}
     ```

     Hydrogens not listed follow their heavy atom; residue names may be globs (`M*`); polymers list one residue per
     monomer, end residues included.
   - **annotated SMILES**, for molecules in any topology (LAMMPS data files have no names): every heavy atom carries
     its bead number, `[CH3:1][CH2:1][CH2:2][CH2:3][CH3:3]` with `"bead_types": {"1": "END", "2": "MID", "3": "END"}`;
     molecules are matched by their bond graph.

   ```bash
   ascg study init --study pentane
   ascg map save --study pentane --key pentane --system-class liquid --spec mapping.json --rationale "..."
   ascg ref inspect --target local --topology ... --trajectory ... --study pentane --key pentane
   ascg map approve --study pentane --key pentane --note "your words"      # after you approve
   ```

3. **Build** the mapped reference with the approved mapping, one `--run` per state point (`T` or `T@P` in atm):

   ```bash
   ascg ref build --study pentane --key pentane --target local --topology /data/pentane/topol.tpr \
       --run "300=/data/pentane/prod.xtc,/data/pentane/energy.edr" --start-ps 1000
   ```

   It writes mapped trajectories, targets (RDFs, bond / angle / dihedral distributions), start configurations and a
   summary (density, compressibility, chain size), and runs the reference QA.

4. **Register** it for fitting at the approved state points:

   ```bash
   ascg aa import --study pentane --task pentane --target local --ref <the ref path from step 3> --states 300
   ```

   Read every QA flag: temperature or density far from the stated state, equilibration drift, chain size not
   converged, RDF range beyond half the box.

Then the fit runs as for any other reference ([quick start](quickstart.md)).

## Notes

- Molecules split across the periodic box are made whole per frame.
- LAMMPS dump times are converted with the log's timestep (or `--timestep-fs`).
- The default RDF range is 1.5 nm; it must stay below half the box (`--r-max-nm`).
- Several state points: one `--run` per state, each from its own AA run; the fit uses exactly the states you
  approve.
