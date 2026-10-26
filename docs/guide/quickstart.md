# Quick start

From nothing to a validated CG model of a molecular liquid, on one GPU workstation, with your own AA trajectory.
Each step works with an agent (Claude Code or Codex) or by hand; the commands below are what the agent runs.

## 1. Install (10 minutes)

```bash
git clone https://github.com/Chenghao-Wu/ascg-agent && cd ascg-agent
conda create -n ascg -c conda-forge python=3.11 rdkit && conda activate ascg
python -m pip install -e '.[aa]'
```

## 2. Try a mapping (no GPU, no data)

Ask the agent for "a three-bead model of liquid n-hexane", or check a mapping yourself:

```bash
ascg map check --system-class liquid \
    --spec '{"format": "annotated-smiles", "smiles": "[CH3:1][CH2:1][CH2:2][CH2:2][CH2:3][CH3:3]", "bead_types": {"1": "END", "2": "MID", "3": "END"}}'
```

It reports the partition, bead masses, the bond / angle / dihedral types the model will have, and warnings (beads
outside 2-5 heavy atoms, unbonded atoms in a bead, very unequal masses).

## 3. Connect your GPU and build fastMD (you run these)

```bash
ascg compute add --name local --transport local --gpus 0 \
    --env "export FASTMD_BINARY=$HOME/software/fastMD/build/fastmd"
ascg engine setup --target local --dest $HOME/software/fastMD --cuda-arch 86   # prints the build commands: run them
ascg compute doctor --target local
```

`doctor` must end with `"ok": true`; its engine gate checks fastMD's energies against numpy and its NPT barostat
([compute](compute.md)).

## 4. Start the agent

- Claude Code: `claude --plugin-dir plugin/ascg`, then `/ascg:study <your request>`.
- Codex: `ascg init codex --workspace ~/cg-work`, then `codex -p ascg -C ~/cg-work`.

A request:

> Coarse-grain liquid n-hexane at 300 K and 1 atm with three beads per molecule (CH3-CH2 | CH2-CH2 | CH2-CH3).
> My GROMACS run is on target "local": /data/hexane/topol.tpr, /data/hexane/prod.xtc, /data/hexane/energy.edr.
> Budget: 2 GPU-hours.

## 5. What happens

1. **Proposal.** The agent restates the system, state point, mapping, model, gates and budget, and waits for your
   approval.
2. **Reference** ([bring your own](bring-your-own.md)): inspect the files, check the mapping on your topology,
   record your approval, build the mapped reference, register it with its QA.
3. **Review** of the mapping and the targets (`ascg map review`, `ascg targets review`).
4. **Fit** (`ascg run submit`): multistate IBI on fastMD with the locked protocol ([protocol](../PROTOCOL_v1.md)):
   - engine gate on the GPU first;
   - stage 1: pair potentials only;
   - stage 2: pair, bond, angle and dihedral updates with the virial pressure correction;
   - NPT checks every 6 iterations.

   The agent waits with `ascg study watch` and reports progress, GPU-hours spent and the projected total.
5. **Decisions.** If a fit stops on a gate or does not converge, the agent diagnoses it and takes at most one bounded
   action at a time (extend, rewind with a smaller timestep or damping), or hands the task back to you.
6. **Validation** on held-out NPT runs (3 seeds per state): density within 5 % of the AA reference and every pair RDF
   NRMSE at most 0.15. The outcome is `validated`, `needs expert input` (gates met, chain size off by more than
   10 %), `failed under budget` or `unsupported`.
7. **Report** (`ascg report task`): a plain-language summary, a technical report, a figure and the model files
   ([results](results.md)).

A 1000-bead liquid or melt takes about 0.5-1.5 GPU-hours on one RTX 3090 / 4090.

## No AA trajectory yet?

- Describe the system in words or SMILES ("PMMA, 10-mers, two beads per repeat unit"): the agent writes it as a
  system (`ascg system check`, `save`; you approve), plans the AA build and runs (`ascg aa plan`) and asks before any
  AA simulation ([systems](systems.md#a-new-system-from-smiles)).
- Systems in a system library (yours, or the benchmark library in `benchmarks/ascg_library.json` with
  `ASCG_LIBRARY`) carry their recipes already.
- A mapped reference made elsewhere (targets, mapped trajectories, start configurations): `ascg aa import`.
