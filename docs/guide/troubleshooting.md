# Troubleshooting

Start with `ascg compute doctor --target T`: every failed check comes with its fix.

| Symptom | Cause and fix |
|---|---|
| `ascg: command not found` (shell) | the package is not installed in this Python: `pip install -e .` in the clone, or use `plugin/ascg/bin/ascg` |
| the plugin's `ascg` says ascg is not installed | a marketplace-installed plugin has no sources next to it: install the package in a Python on your PATH, or set `ASCG_PYTHON` / `ASCG_SRC` |
| `map check` fails with an RDKit import error | the launcher picked a Python without RDKit: set `ASCG_PYTHON` to the right one (the choice is cached in `~/.cache/ascg/python`) |
| `run submit` refuses: the code on T differs from this checkout | `ascg compute deploy --target T` |
| doctor: `fastMD version` | the build predates fastMD main b35e30a (tabulated / CBT dihedrals, NPT fix): rebuild as `ascg engine setup --target T` prints |
| doctor: engine gate NPT stability fails | a fastMD without the symmetric NPT barostat: liquids of small stiff molecules collapse at short barostat damping; rebuild fastMD |
| doctor: engine gate skipped, no free GPU | every listed GPU runs someone else's process; ascg waits rather than share (`--gpu-policy shared` to allow sharing) |
| a job waits in the Slurm queue | `ascg compute hosts` shows the expected start; several `--qos` values let ascg pick the QoS with room |
| `ref inspect`: MDAnalysis missing | `ascg compute deploy --target T --install-deps`, or `pip install MDAnalysis` in the target's Python |
| `ref inspect`: no bonds | the topology must carry bonds: use the `.tpr` or `.top`, a LAMMPS data file or a PSF, not a `.gro` / `.pdb` |
| QA flags after `aa import` | density or temperature far from the stated state, drifting density, unconverged chain size, RDF range beyond half the box: check the AA run before fitting |
| `amend approve` refuses | approvals need an interactive terminal: run it yourself in a separate terminal window (agent shells and Claude Code's `!` are not interactive) |
| Codex cannot reach a target | targets are reached only through the MCP tools; check `~/.codex/ascg.config.toml` (rerun `ascg init codex`) and that `ssh <alias> true` works from your shell |
| `ascg init codex` refuses: inside the writable roots | the studies root, compute profile, library, Python or code lies inside the workspace, /tmp or `$TMPDIR`; move it or choose another workspace |

## Fits that stop

A fit can stop at a decision point (`awaiting_decision`, not converged within the budget) or on a gate
(`stopped_on_gate`, e.g. bonds stretched past their table). The agent diagnoses the run with the `diag` tools
(`diag digest`, `diag bonds`, `diag structure`, `diag pressure`, `diag chains`, `audit tables`) and takes one bounded
action with a written rationale, or escalates to you. Typical cases (the case library of the ascg-diagnose skill):
- **bonds beyond their break length in consecutive frames** (a stiff bond, early in the fit): rewind with a smaller
  timestep; at iteration 0, restart with a smaller timestep;
- **density converging slowly** with good structure and the pressure falling: extend by 2-3 schedule cycles;
- **chain size off by more than 10 %** with the gates met: a limit of the reference or the mapping, not of the fit;
  escalated to you with the evidence;
- **crashes and lost jobs**: retried automatically (twice per stage), then resubmitted.

Rule changes (gates, metrics, budgets) are never actions: they are amendments, proposed by the agent and approved by
you.
