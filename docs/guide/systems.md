# Systems, mappings and AA builds

ascg has no built-in systems. A system comes from your request: a description the agent turns into SMILES, your
own AA files ([bring your own](bring-your-own.md)), or a **system library** you point ascg to.

## A new system from SMILES

One SMILES per species describes what ascg builds, and the same SMILES can carry the CG mapping as bead numbers. You
can say it in words ("PMMA, 10-mers, two beads per repeat unit: backbone and ester") and the agent writes:

```json
{"system_class": "polymer",
 "species": [{"name": "MMA", "smiles": "*[CH2:1][C:1](*)([CH3:1])[C:2](=[O:2])[O:2][CH3:2]",
              "bead_types": {"1": "BB", "2": "EST"}}]}
```

```bash
ascg system check --spec system.json --key pmma          # species, chain ends, size, AutoPoly units, mapping check
ascg system save --study S --key pmma --spec system.json --rationale "..."
ascg system approve --study S --key pmma --note "your words"     # approves the system and its mapping
ascg aa plan --task pmma --study S --states 400,500
```

- **Polymers:** the repeat unit with two `*`, head first. Defaults, which the agent tells you: 10 repeat units,
  atactic, H end caps, chains for about 1000 CG beads (at least 20). The start and end units are the repeat unit with
  one `*` capped. H caps suit addition polymers; condensation polymers get their end units as SMILES (e.g. methyl
  esters), and `system check` warns when an H cap would make an aldehyde or a silane end.
- **Liquids and mixtures:** each molecule's SMILES, with a count or a mole fraction (1000 molecules by default).
- **Mapping:** annotate every heavy atom with its bead number, or none (then the mapping is agreed separately with
  `map save` / `map approve`).
- **Copolymers:** one species per monomer, plus `"sequence"` (`"S*3 BD*4 S*3"`, `"E VA E VA ..."`) or
  `"random"` (`{"E": 0.5, "P": 0.5}` with `"repeat_units"`; a new sequence per chain). `system check` lists the bond,
  angle and dihedral types the sequence creates. Graft, branched and gradient polymers are not covered.

The full format is in the [system reference](../../plugin/ascg/skills/ascg-specify-system/references/system.md).
From the system, `system save` derives the AutoPoly recipe and the mapping file that the AA build and `map-aa` read.
For all 29 homopolymer units and the 4 copolymers of the benchmark, the derived recipes equal the hand-written
ones (with methyl caps or given end units where the benchmark chose them), and AutoPoly builds them identically.

## System library

`ASCG_LIBRARY=/path/library.json`, `./ascg.library.json` or `~/.config/ascg/library.json` (`ASCG_LIBRARY=none` turns
it off). `ascg catalogue search --query "<name>"` finds entries by name, key or alias; `ascg map show --task K` shows
an entry's mapping and its approval status.

```json
{"schema": "ascg-library/1", "name": "my systems",
 "systems": [
  {"key": "n_pentane", "id": "L01", "name": "n-Pentane", "system_class": "liquid", "aliases": ["pentane"],
   "resolution": "3 beads per molecule", "mapping_status": "approved",
   "mapping": {"format": "annotated-smiles", "smiles": "[CH3:1][CH2:1][CH2:2][CH2:3][CH3:3]",
               "bead_types": {"1": "END", "2": "MID", "3": "END"}},
   "recipe": {"file": "my_recipes.json", "task": "n_pentane"}, "force_field_override": null}]}
```

- `system_class`: `polymer`, `liquid` or `liquid_mixture` (mixtures list their `components` by library `id`).
- `mapping`: the annotated-smiles form, or `rdkit-heavy-atom-indices`; polymers give one repeat unit with its two `*`
  attachment points, head first ([mapping formats](../../plugin/ascg/skills/ascg-mapping/references/formats.md)).
- `mapping_status`: anything other than `approved` makes the agent confirm the partition with you first.
- `recipe` (optional): the AutoPoly build of the AA system, relative to the library file.
- `force_field_override` (optional): parameters (JSON, as in `benchmarks/ff/`) that replace the OPLS-AA library's
  for a system it types badly.

`benchmarks/ascg_library.json` is the project's benchmark (40 systems: 33 polymers, 6 molecular liquids, 1 mixture),
with their AA recipes in `benchmarks/aa_recipes.json` ([benchmark README](../../benchmarks/README.md)).

## Mappings

A mapping is always yours to approve. The agent proposes one from your description ("two beads per repeat unit"),
checks it with `ascg map check` (partition, bead masses, bonded types, warnings) and records your approval with
`ascg map save` and `ascg map approve`. Guidelines it follows: functional groups whole, 2-5 heavy atoms per bead,
chemically distinct beads get distinct types, at most 16 bead types (fastMD).

## AA references

Three ways to a mapped reference, per approved state point:

1. **Your own trajectories** ([bring your own](bring-your-own.md)): `ascg ref inspect`, `ascg ref build`,
   `ascg aa import`.
2. **A mapped reference made elsewhere** in ascg's layout (`topology.json`, and per state `mapped_T*.npz`,
   `targets_T*.npz`, `start_T*.npz`, plus `summary.json`): `ascg aa import --study S --task K --target T --ref DIR
   --states ...`.
3. **ascg builds and runs the AA system** (AutoPoly v2.0 and GROMACS on a target with AA settings,
   [compute](compute.md#aa-settings)), for a system saved and approved in the study:

   ```bash
   ascg aa build --study S --task K --target T            # AutoPoly build, neutralization, typing audit (one job)
   ascg aa status --study S --task K                      # build size, net charge, audit verdict and findings
   ascg aa submit --study S --task K --states 400,500 --dry-run      # plan, run lengths, core-hours, wall time
   ascg aa submit --study S --task K --states 400,500 --approval "your words"
   ascg aa map --study S --task K                         # mapped reference, then:
   ascg aa import --study S --task K --target T --ref <from aa status> --states 400,500
   ```

   The protocol follows the paper: 1000 CG beads; polymers equilibrated with RadonPy's 21-step compression /
   decompression schedule at the hottest state, liquids directly; then per state a 1 ns pre-equilibration, 10 ns
   (polymers) or 4 ns (liquids) equilibration and 10 ns production at (T, P). An audit finding blocks the runs until
   it is resolved; the estimate must stay under the target's core-hour cap; `aa submit --resume` continues after a
   time limit or a node failure from the GROMACS checkpoints. For batch campaigns outside a study, the scripts in
   `cluster/aa/` (`build.sh`, `launch_gmx.sh`) run the same commands from a recipes file.

### Recipe files

For batch work (many systems, copolymers), a recipes file gives AutoPoly the building blocks (complement SMILES with `[*]` attachment points; the first and last
units carry the end groups) and each task's composition:

```json
{"version": 1,
 "defaults": {"chains": 80, "repeat_units": 10, "tacticity": "atactic"},
 "units": {"E": {"first": "CC[*]", "middle": "[*]CC[*]", "last": "[*]CC"}},
 "states": {"polymer": "400,500", "liquid": "300"},
 "tasks": {"pe": {"class": "polymer", "sequence": "E*10"}}}
```

See `benchmarks/aa_recipes.json` (homopolymers, copolymers, liquids, a mixture, force-field overrides) for complete
examples.

## State points

A state point is a temperature and a pressure: `300` (1 atm), `400@1000` (atm) or `400@1013.25bar`. You approve
them; there is no default. The fit uses exactly the approved states:
- one state: ordinary IBI, valid at that state;
- two temperatures: every table U(x, T) = A(x) + B(x) T is exact at both;
- more states, or states that differ only in pressure: a weighted least-squares compromise (every gate still applies
  at every state).

Every approved state needs its own AA reference.
