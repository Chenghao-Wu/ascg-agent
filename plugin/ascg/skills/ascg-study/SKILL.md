---
name: ascg-study
description: Run a systematic coarse-graining study end to end with the ascg tools: specify the system, get approvals, connect compute, import or build atomistic references, review the mapping and targets, fit with multistate IBI, wait for events, diagnose, validate and report. Use for any request to coarse-grain a molecule or polymer.
---
# Run a coarse-graining study

All numbers come from `ascg` tools (JSON on stdout; `ascg tools` lists them, `ascg <group> <action> --help` shows
parameters; a parameter error returns the tool's usage line). Never compute or invent results yourself; never edit
study files, protocols or gate code; never run cluster commands that bypass `ascg` (a guard blocks them).

**One `ascg` command per shell call.** Pipes, loops, `python -c` and `ssh` are refused, so do not filter or repeat
with the shell. Read files (skill references, logs) with your file-reading tool. A refused shell command does not
mean the ascg tools are unavailable: keep using `ascg` commands. The tools do the filtering:
- several iterations go in one call (`--iterations 1,5,8`);
- one-of defaults are filled in (the only task, the only state, the latest iteration);
- `compute hosts --targets X` limits the output to one target;
- a single required value may be given positionally (`ascg catalogue search <name>`).

## Procedure
1. **Specify** (skill ascg-specify-system): identity, composition, state points, resolution, budget. Ask the user when
   anything is ambiguous; a mapping is always the user's decision.
2. **Compute** (ascg-setup-compute): `ascg compute list`; if empty or failing, set up a target with the user.
   `ascg compute hosts` before placing work.
3. **Study**: `ascg study init --study <name> --description "..."` (locks the protocol; arm agentic by default).
4. **Reference.**
   - Existing mapped reference: `ascg aa import ... --states <the approved state points>` (ascg-aa-simulate).
   - The user's own AA trajectories (GROMACS, LAMMPS, ...): ascg-bring-reference (`ref inspect`, mapping,
     `ref build`, then `aa import`).
   - Otherwise ascg builds and runs the AA system (ascg-specify-system, ascg-aa-build, ascg-aa-simulate): the system
     from SMILES (`system save`, the user approves), `aa build`, the audit with the user, `aa submit --dry-run`, the
     user's approval of the cost, `aa submit --approval`, `aa map`, `aa import`.
   Read every QA flag of `aa import` and tell the user about warnings before fitting.
5. **Mapping and targets** (ascg-mapping, ascg-targets): `ascg map review`, `ascg targets review`.
6. **Fit** (ascg-fit-ibi): `ascg run submit` per task; then wait with `ascg study watch --study S --timeout 540` (shell
   timeout 10 minutes; repeat until a wake event; never sleep-poll). `study status` / `study watch` show each task's
   GPU-hours spent and projected total (`budget.cost`) and the actions left: compare them with the approved budget.
7. **On awaiting_decision or stopped_on_gate**: get a diagnosis (Claude Code: delegate to the `ascg:ascg-diagnostician`
   subagent; otherwise follow ascg-diagnose yourself). Check the plan and cost of the action with `ascg run act ...
   --dry-run`, then take ONE bounded action with `ascg run act ... --rationale` or `ascg run escalate`. Rule changes are only proposed (`ascg amend propose`); the user approves them.
8. **Done**: `ascg report task`; for important claims ask the `ascg:ascg-reviewer` subagent for an independent check.
   Finish with `ascg report study` and a short plain-language summary with the classes and the numbers from the
   reports.

## Records
Everything you decide goes through the tools (events, interventions, rationales). Observations or corrections that no
tool records: `ascg study note --study S [--task K] --message "..."` (append-only). Never edit study files.

## Ask the user before
- chemistry or mapping choices;
- new state points or AA simulations;
- compute beyond the approved budget. Use the projection from `study status` or `run act --dry-run`. When the user is
  not available, say so in the action's rationale and in a `study note`, and report the overrun;
- any rule change;
- anything irreversible on shared machines.

## Outcome classes
validated / needs expert input / failed under budget / unsupported - report them as they come, including failures.
