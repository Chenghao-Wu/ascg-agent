---
name: ascg-fit-ibi
description: Fit the CG model with multistate IBI in fastMD through ascg: submit, wait for events, read progress, and take bounded actions when a task stops or misses convergence.
---
# Fit with multistate IBI

Protocol (locked per study; `ascg study explain` shows it, `ascg study verify` checks the lock):
- stage 1: pair-only IBI, 15 iterations;
- stage 2: pair, pair, pair, bond, angle, dihedral to iteration 42, with the virial pressure correction;
- in-loop NPT checks at cycle ends;
- then the convergence check: in-loop density within 4 %, worst RDF <= 0.15;
- the bond gate: a bond is broken beyond the length where its table rises 50 k_B T above its minimum, in two
  consecutive frames or at twice that length in any frame. It stops fitting and validation runs; in-loop NPT
  checks only record it.

A converged task validates automatically, with a `phase` event (validating).

1. `ascg run submit --study S --task K` (placement by policy: a free GPU on workstations, the queue on clusters).
2. `ascg study watch --study S --timeout 540` - blocks up to 9 minutes and returns at the first wake event
   (awaiting_decision, stopped_on_gate, done, error); give the shell call a 10-minute timeout (Claude Code Bash tool:
   timeout 600000) and call it again until every task is finished. Do not sleep-poll; progress events give
   iteration, worst RDF, pressure and NPT density.
3. Typical healthy progress: pressure thousands of bar after stage 1, below 100 bar within 3-6 iterations of stage 2;
   worst RDF NRMSE ~0.01-0.03.
4. `awaiting_decision` (convergence missed) or `stopped_on_gate` (e.g. broken bonds): diagnose (ascg-diagnose), then
   ONE of: `ascg run act --action extend|extend_lower_damping|rewind_smaller_timestep|rewind_lower_damping ...
   --rationale "..."` or `ascg run escalate --message "..."`. Budget: 3 actions, 36 extra iterations per task; the tool
   refuses beyond it. After a gate stop the last tables cannot be continued: rewind. A gate stop before iteration 0
   completed has nothing to rewind to: `--action restart_smaller_timestep --dt-factor ...` restarts from the
   Boltzmann-inverted start at a smaller timestep (also for a task escalated for that reason).
   - First run the same command with `--dry-run`. It shows the plan the job will run: a rewind back inside stage 1
     finishes stage 1 before stage 2. It also shows the projected total GPU-hours (a smaller timestep is slower
     per iteration) and whether the budget allows the action.
   - If the projection exceeds the approved compute budget, say so in the rationale. Ask the user if they are
     available; otherwise record it with `study note` and report it.
5. `queued` on a cluster: `study status` gives the scheduler's expected start (`queue.expected_wait_h`). A wait far
   beyond the fit itself (~1 h) is a placement decision: if the same reference exists on another target, `ascg run
   cancel` the queued job and `ascg aa import` + `run submit` it there under a new task key (e.g. <task>_2); record why
   with `ascg study note`. A cancelled task stays in the registry as cancelled.
6. `error`: an infrastructure problem (crash, node failure). The job already retried; check `ascg diag digest` and
   `ascg compute hosts`, then resubmit or tell the user.
7. Budget: the default protocol needs about 0.3-1 GPU-hour for a ~1000-bead polymer. Small-molecule liquids with
   several beads per molecule, or bigger systems, can need more. Quote the measured projection
   (`study status` -> `budget.cost.projected_total_gpu_h`) once a few iterations have run, not a rule of thumb.
