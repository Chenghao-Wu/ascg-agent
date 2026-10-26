# Results and the model

## Outcome classes

Every task ends in one class, decided by the tools from the held-out validation, never by the agent:

| Class | Meaning |
|---|---|
| `validated` | at every approved state, over 3 held-out NPT seeds: density within 5 % of the AA reference and every pair RDF NRMSE at most 0.15; chain size within 10 % of AA |
| `needs expert input` | the gates are met but the chain size is off by more than 10 %, or the agent escalated the task with its reasons |
| `failed under budget` | no model met the gates within the protocol's iteration and action budget, or a gate stopped the fit (e.g. broken bonds) |
| `unsupported` | the reference or the engine cannot run the task (preflight) |

The rules are in the [protocol](../PROTOCOL_v1.md); a study locks them by hash, and only an approved amendment changes
them.

## Reports

`ascg report task --study S --task K` fetches, checksum-verified, into `<studies>/S/tasks/K/`:

| File | Content |
|---|---|
| `report/user_summary.md` | plain language: the class, density error, worst RDF, chain size per state, GPU-hours, fitting history |
| `report/technical_report.md` | protocol, outcome, engine build, costs, per-seed validation |
| `report/structure_density.png` | RDFs and bonded distributions, CG vs AA, and the density per state |
| `report/outcome.json`, `report/costs.json` | the same in machine form |
| `cg/ibi/final/validation.json` | every held-out run's numbers |
| `workflow/state.json` | the fit's history, settings and engine provenance |

`ascg report study --study S` adds a benchmark table and figure over the study's tasks.

## The model files

`ascg report task --study S --task K --model-files` also fetches `report/model_files/`:

```
model_files/
  README.md
  T300/                  one directory per state point (T300_P1000 for 300 K at 1000 atm)
    tables/              pair, bond, angle and dihedral tables at that temperature
    system.data          a start configuration (mapped from the AA reference) with velocities
    run_npt.conf         a ready fastMD NPT input (the held-out validation settings)
```

Run a state with fastMD:

```bash
cd model_files/T300 && $FASTMD_BINARY run_npt.conf
```

- Units: nm, ps, amu, kJ/mol; thermostat and barostat targets in fastMD's units (k_B T in kJ/mol, pressure in
  kJ/mol/nm3).
- Tables: pair (energy, force), bond (energy, force), angle (degrees, force per degree), dihedral (degrees, combined
  bending-torsion).
- Bond type `EXCL13` is a zero-energy bond that excludes 1-3 pairs from the pair potential; keep it.
- With two or more temperatures, every table is U(x, T) = A(x) + B(x) T; A and B are in the task directory under
  `cg/ibi/final/tables/potentials_tdep.json`, for any temperature between the fitted ones.

## The full record

On the compute target, the task directory (`<work_root>/<study>/<task>`) keeps everything:
- `request/`: the specification and approvals;
- `workflow/`: state, provenance, retries, job scripts and logs;
- `cg/ibi/`: every iteration (immutable), the final tables, the held-out runs;
- `report/`.

In the study directory, `events.jsonl` and `calls.jsonl` record every event and every tool call with who made it.
`ascg study explain --study S --task K` gathers the reference, protocol hash, amendments, fastMD build, jobs and
interventions.
