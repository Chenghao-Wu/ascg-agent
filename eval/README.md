# Agent evaluation

The same tasks under Claude Code (plugin) and Codex (`ascg init codex` settings), with frozen tools, isolated runs
and automatic scoring.

```bash
python eval/harness.py run --out ~/ascg-eval/BATCH --agents claude codex --replicates 3      # all tasks
python eval/harness.py run --out ~/ascg-eval/BATCH --tasks map_dmso_2bead --agents claude-solo
python eval/harness.py score ~/ascg-eval/BATCH                                              # summary.md, summary.json
```

Needs `claude` and `codex` logged in, and a Python with ascg's dependencies (run the harness with it). Runs cost
money: Claude Code reports dollars per run, Codex reports tokens.

## What a batch records

- `batch.json`: the frozen tools (a copy of the tracked `src/ascg`, `src/ascg_core` and `plugin/ascg` in `code/`, with
  their digest, the git commit and whether the checkout was dirty), the agent CLI versions and the Python. A batch keeps
  one tool version; rerunning `run` on it adds missing runs and refuses if the checkout's tools changed.
- `runs/<task>/<arm>_r<k>/`: `task.json` (the task as run), `meta.json` (prompt, model, times, exit), the agent's
  `transcript.jsonl` and `stderr.log`, `calls.jsonl` (every ascg tool call with its arguments and result), the agent's
  `workspace/`, `studies/`, `data/` (task fixtures) and `compute.json`, and `score.json` after scoring.

## Isolation

- Each run has its own studies root, compute profile and workspace. No system library unless the task names one.
- **Claude Code:** the plugin from the frozen copy; `--permission-mode dontAsk` with the allow-list `Bash(ascg *)`,
  Read, Glob, Grep, Skill, TodoWrite (+ Agent in the `claude` arm, not in `claude-solo`); no auto-memory, no user
  settings or MCP servers; reads of this checkout and of `~/.claude/projects` denied.
- **Codex:** `--ignore-user-config`; the shell in the workspace-write sandbox without network; the ascg tools over
  MCP only, from the frozen copy; approval policy `never`. Codex can read any file, so keep batches away from data the
  agent must not see.
- Batches must lie outside this checkout (agents would read the benchmark library and the tests) and outside /tmp
  and `$TMPDIR` (Codex can write there).

## Tasks

`tasks/<id>.json`: `prompt` (`{data}` is the run's data directory), `held_out` (true when no skill case or benchmark
run describes the system), `library`, `compute` (`local`: a target `here` on this machine with the frozen code;
`user`: the user's `ASCG_COMPUTE` profile, for tasks on real hosts), `fixture` (synthetic data from `fixtures.py`),
`max_turns`, `timeout_min` and `checks`:

| Check | Passes when |
|---|---|
| `called` | the tool was called (successfully with `"ok": true`) |
| `not_called` | none of the listed tools was called |
| `result` | a successful call returned `r` with `expr` true: the last call, or any call with `"which": "any"` |
| `final` | the final answer matches `regex` |

Checks read the call log, i.e. what the ascg tools did, not what the agent said, except `final`.

| Task | Held out | Expected |
|---|---|---|
| `map_hexane_3bead`, `map_polystyrene_2bead` | no (benchmark systems) | a checked mapping with the expected partition; nothing saved |
| `map_pnipam_3bead`, `map_dmso_2bead` | yes | a checked mapping with the asked resolution; nothing saved |
| `byo_pentane_liquid` | yes | synthetic GROMACS liquid that matches the description: inspect, save + approve the mapping, build, register; no fit |
| `byo_gas_labelled_liquid` | yes | the same request on a gas with drifting T and P: inspect, then stop and say why |
| `system_pma_2bead`, `system_mixture`, `system_block_copolymer` | yes | a polymer / a liquid mixture / a block copolymer ascg builds, from a description: annotated SMILES, `system check`, `system save` with the right size and partition, `aa plan`; nothing approved or run |

Fits and decision fixtures (recorded failing tasks on real hosts) use `"compute": "user"` and are kept outside this
directory with `--tasks-dir`, because they name the user's hosts and task directories.

## Metrics (score.json, summary)

Checks passed; wall time; cost (Claude Code) or tokens (Codex); ascg calls, failed calls and calls refused by policy;
agent shell commands (Claude Code: ascg and other, refused by the allow-list; Codex: commands and MCP calls); reads
of the tool sources; subagents; the final answer.
