# Compute: a local GPU workstation or your cluster

CG fits run on compute targets, declared in your profile `~/.config/ascg/compute.yaml` (or the file named by
`ASCG_COMPUTE`). ascg talks to a target in one of two ways:

| | Local GPU | Your HPC cluster |
|---|---|---|
| Transport | `local`: ascg runs on the machine with the GPU | `ssh`: an ssh alias that logs in without a password |
| Scheduler | `none`: jobs start as background processes | `slurm`: jobs are submitted with sbatch |
| Code | this clone | a copy kept in step with `ascg compute deploy` |

Adding a target and deploying code are your steps: agents give you the command to run, they do not run it (a
target's `env` line runs as a shell command). Agents run `ascg compute doctor` and `ascg compute hosts` themselves.

## A local GPU workstation

```bash
ascg compute add --name local --transport local --gpus 0 \
    --env "export FASTMD_BINARY=$HOME/software/fastMD/build/fastmd"
ascg compute doctor --target local
```

- `--gpus`: the GPU indices ascg may use. With the default policy `avoid_foreign`, ascg never starts work on a GPU
  that runs anyone else's process, and checks again at launch (`--gpu-policy shared` turns that off).
- `--env`: a shell line run before every ascg call on the target; use it for `FASTMD_BINARY` and anything else the
  environment needs (`source ~/miniconda3/bin/activate ascg`, module loads).
- Task directories go to `<clone>/runs/ascg` unless you give `--work-root`.

## A Slurm cluster over ssh

```bash
ascg compute add --name hpc --host myhpc --scheduler slurm \
    --root /work/me/ascg-agent --work-root /work/me/ascg-runs \
    --env "source /work/me/ascg-agent/cluster/env/slurm.sh" \
    --qos gpu --partition gpu --gres gpu:1 --max-jobs 4 --time 08:00:00
ascg compute deploy --target hpc --install-deps
ascg compute doctor --target hpc
```

- `--host`: an alias from `~/.ssh/config` that logs in non-interactively (`ssh myhpc true` must work).
- `--root`: where the code lives on the cluster; `--work-root`: where task directories go (use the work file system
  when home has a quota).
- `--env`: puts Python, the code and `FASTMD_BINARY` on the path. `cluster/env/slurm.sh` reads your
  `cluster/site.local.sh` (a copy of `cluster/site.example.sh`, never committed); `ascg compute deploy` copies it.
- Slurm: `--qos` (one, or several comma-separated: each job goes to the first QoS with GPUs left under its per-user
  cap), `--partition`, `--gres`, `--account`, `--cpus`, `--mem`, `--time`; `--max-jobs` caps ascg's jobs on the
  target.
- GPUs exist only inside jobs: every job runs the engine gate on its own GPU before fitting.

## AA settings

A target that builds and runs all-atom systems (`aa build`, `aa submit`, `aa map`) needs AutoPoly v2.0 in its Python
and GROMACS, plus batch settings per stage, given once with `--aa`:

```bash
ascg compute add --name hpc ... --aa '{"env": "module load gromacs/2025.1", "gmx": "gmx_mpi",
  "mdrun": "mpirun -np 1 gmx_mpi mdrun", "require_cpu_flag": "avx512f", "max_core_h": 2000,
  "build": {"qos": "cpu", "partition": "cpu", "cpus": 2, "time": "02:00:00"},
  "md":    {"qos": "cpu", "partition": "cpu", "cpus": 16, "time": "2-00:00:00"},
  "map":   {"qos": "cpu", "partition": "cpu", "cpus": 2, "mem": "24G", "time": "12:00:00"}}'
```

- `env`: the shell line that makes GROMACS available in the MD jobs (after the target's own `env` line).
- `mode`: `cpu` (default: the equilibration job, then one job per state) or `gpu` (one job, `gpu` stage settings).
- `require_cpu_flag`: MD jobs stop on nodes without it (a GROMACS build for one instruction set).
- `max_core_h`: the cap per task; `aa submit` refuses a larger estimate.
- `compute doctor` checks AutoPoly, MDAnalysis and GROMACS on the target.

## Deploy: the same code everywhere

`ascg compute deploy --target T` copies this clone's tracked code (src, cluster, benchmarks, patches, pyproject, and
your untracked `cluster/site.local.sh`) to the target and checks the md5 of every file there. It never deletes files,
and it refuses while ascg work runs on the target (`--force` overrides). `--install-deps` pip-installs missing Python
packages into the target's Python, keeping numpy and scipy at their installed versions.

`ascg run submit` refuses a target whose code differs from your clone, so deploy after every update.

## Doctor

`ascg compute doctor --target T` checks, and gives the fix for each problem:

| Check | What it means |
|---|---|
| connection | the target answers (ssh, root, env line) |
| dependencies | numpy, scipy, matplotlib, RDKit, networkx, PyYAML in the target's Python; MDAnalysis optional |
| code | the target's code digest equals your clone's |
| fastMD, fastMD version | `FASTMD_BINARY` exists; its source is fastMD main b35e30a or later |
| GPUs | GPUs visible and which are free of other processes (workstations) |
| engine gate | on a free GPU (workstations): fastMD energies vs numpy with 1-2 / 1-3 exclusions, CBT dihedral energies vs numpy, and NPT stability of a small stiff-molecule liquid; Slurm jobs run it at their start |
| scheduler | Slurm answers |

`--no-engine` skips the engine gate.

## Sharing machines

- `ascg compute hosts` shows each target's GPUs (this study's processes vs everyone else's) and your queued jobs.
- ascg cancels only its own jobs (`ascg run cancel`) and never touches other processes.
