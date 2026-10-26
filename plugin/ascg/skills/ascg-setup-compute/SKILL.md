---
name: ascg-setup-compute
description: Set up and check compute targets for ascg (ssh or local; Slurm or plain workstations): write the user profile with ascg compute add, test it, and respect other users on shared machines.
---
# Connect compute

Targets live in the user's profile (`~/.config/ascg/compute.yaml`, `./ascg.compute.yaml` or `$ASCG_COMPUTE`), never in
the plugin. Ask the user which machines and accounts may be used; never use an account or alias they did not name.

1. `ascg compute list`. For a new target ask for: ssh alias (non-interactive login must work), scheduler (slurm or
   none), the directory that holds the ascg / ascg_core sources (`root`), where task directories may be written
   (`work_root`; on clusters with home quotas use the work file system), the environment line (`env`, e.g.
   `source /path/env.sh` that puts Python, the sources and FASTMD_BINARY on the path), Slurm qos / partition / gres /
   max_jobs, workstation GPU indices.
2. Writing a target is the user's action: its `env` line runs as a shell command on every call. Give the user the
   exact `ascg compute add --name ...` command to run in their own terminal (the guard and the MCP server do not run
   it for agents); the same holds for `ascg compute deploy --target ...`, which copies the code to a remote target.
   Then `ascg compute doctor --target ...` (or `compute test`): python and dependencies, code, `fastmd_present` and
   its version, GPUs, engine gate, scheduler. If fastMD is missing: skill ascg-setup-engine.
3. `ascg compute hosts`: GPUs show this study's processes (`own`) and everything else (`foreign`). Default policy
   `avoid_foreign`: ascg never places work on a GPU with someone else's process, and it re-checks at launch.

## Pitfalls seen in practice
- Cluster module systems may need `source /etc/profile` before `module load` inside the env line.
- Home directories over quota: put `work_root` (and builds) on the work file system.
- Queues have per-user submit limits; ascg backs off at `max_jobs`, but tell the user when work waits in a queue.
- Login nodes have no GPUs: engine selfchecks on Slurm targets run inside the jobs.
