# Host scripts

CG fits do not need these scripts: ascg submits those jobs itself (`ascg run submit`; targets in
`~/.config/ascg/compute.yaml`, whose `env` line sources an environment file from `env/`).

| Path | Use |
|---|---|
| `site.example.sh` | Template of `site.local.sh` (not tracked): cluster paths, GROMACS module, partitions, QoS, accounts |
| `env/slurm.sh` | Environment of a Slurm cluster (reads `site.local.sh`); `env/common.sh` is the shared part |
| `env/workstation.example.sh` | Template environment of a GPU workstation without a scheduler |
| `aa/build.sh` -> `aa/build_aa.slurm` | AutoPoly v2.0 OPLS-AA builds of a recipes file (array job): build, neutralization, typing audit, CG topology when mapped |
| `aa/apply_autopoly_patch.sh` | Adds `patches/opls_lt_2024_ascg_additions.fdefn` to the pinned AutoPoly checkout (run once before builds) |
| `aa/launch_gmx.sh` -> `aa/gmx_run.slurm`, `aa/map_aa.slurm` | GROMACS protocol per task from the recipe (class, approved states, equilibration): CPU jobs (equilibration, then one per state branch) or one GPU job; optional mapping job (`MAP=1`); `DRYRUN=1` prints the submissions |
| `ibi/ibi_protocol_v1.slurm` | The scripted protocol v1 job (`ASCG_ENV=<env file>`), to reproduce the benchmark without an agent |

All-atom runs on a Slurm cluster:

```bash
source cluster/env/slurm.sh
bash cluster/aa/build.sh benchmarks/aa_recipes.json runs/aa pmma pdms                    # or all tasks
bash cluster/aa/launch_gmx.sh benchmarks/aa_recipes.json runs/aa pmma pdms
STATES=400,500 MODE=gpu MAP=1 bash cluster/aa/launch_gmx.sh benchmarks/aa_recipes.json runs/aa pe
```

State points are always explicit: from the recipes file (`states` per class) or `STATES`, never a default.
