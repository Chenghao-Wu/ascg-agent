# ascg-agent: agentic systematic coarse-graining

ascg-agent builds systematic coarse-grained (CG) models of polymers and soft matter from all-atom (AA) simulations.
A coding agent (Claude Code or Codex) drives the work through deterministic tools, and you approve the scientific
decisions: the system, its state points, the mapping and the budget.

1. **AA reference:** your own GROMACS / LAMMPS trajectories, or an AutoPoly v2.0 OPLS-AA build run with GROMACS at
   each approved state point (T, P).
2. **Mapping:** atom-to-bead mapping from your description ("three beads per molecule"), checked by the tools and
   approved by you; mapped targets and start configurations.
3. **Multistate IBI with fastMD:** tabulated pair, bond, angle and combined bending-torsion dihedral potentials,
   U(x, T) = A(x) + B(x) T over the approved states, with a virial pressure correction.
4. **Independent validation:** held-out fastMD NPT runs at every state; density within 5 % and every pair RDF NRMSE
   at most 0.15; plain-language reports and ready-to-run model files.

Numerical work is deterministic. The agent proposes, runs, waits, diagnoses within bounded recovery actions and
reports; the tools refuse anything that would change gates, metrics or outcomes, and rule changes need your approval
at a terminal.

> **Status:** pre-release, developed alongside a paper; the repository becomes public at submission.

## Quick start

```bash
git clone https://github.com/Chenghao-Wu/ascg-agent && cd ascg-agent
conda create -n ascg -c conda-forge python=3.11 rdkit && conda activate ascg
python -m pip install -e '.[aa]'
ascg compute add --name local --transport local --gpus 0 \
    --env "export FASTMD_BINARY=$HOME/software/fastMD/build/fastmd"
ascg engine setup --target local --dest $HOME/software/fastMD --cuda-arch 86   # prints the fastMD build: run it
ascg compute doctor --target local
```

Then start an agent and say what you want to coarse-grain:

```bash
claude --plugin-dir plugin/ascg                                 # Claude Code: /ascg:study <request>
ascg init codex --workspace ~/cg-work && codex -p ascg -C ~/cg-work      # Codex
```

The [quick start](docs/guide/quickstart.md) walks through a first model from your own AA trajectory.

## Guides

| Guide | For |
|---|---|
| [Install](docs/guide/install.md) | Python environment, fastMD, the optional AA build tools |
| [Compute](docs/guide/compute.md) | a local GPU workstation or your Slurm cluster over ssh |
| [Agents](docs/guide/agents.md) | Claude Code and Codex, containment, what stays with you |
| [Bring your own AA trajectory](docs/guide/bring-your-own.md) | GROMACS / LAMMPS / MDAnalysis files to a reference |
| [Systems](docs/guide/systems.md) | system libraries, mappings, AutoPoly recipes, state points |
| [Results](docs/guide/results.md) | outcome classes, reports, model files |
| [Troubleshooting](docs/guide/troubleshooting.md) | common problems, stopped fits |
| [Protocol v1](docs/PROTOCOL_v1.md) | every setting, gate and outcome rule of the fit |
| [Evaluation](eval/README.md) | both agents on fixed tasks with frozen tools |

## Scripted use (no agent)

The same protocol without an agent, on a mapped reference:

```bash
python -m ascg_core.ibi.pipeline preflight --ref runs/cg_ref/pe
python -m ascg_core.ibi.pipeline run --ref runs/cg_ref/pe --task-dir runs/ibi_v1/pe --gpus inherit
```

## Repository

| Path | Content |
|---|---|
| `src/ascg_core/` | Numerical core: `aa/` (AutoPoly build, OPLS audits, GROMACS protocol), `mapping/` (mapping specs, CG topology, mapped references, bring-your-own), `ibi/` (states, tables, updates, fastMD I/O, runner, pipeline, engine gate) |
| `src/ascg/` | Deterministic tool layer: `ascg <group> <action>`, JSON in and out, fail-closed policies, study registry, compute targets, MCP server, `ascg init codex` |
| `plugin/ascg/` | Agent plugin: skills, case libraries, subagents, commands, guard hook |
| `benchmarks/` | The benchmark as user data: 40 systems (33 polymers, 6 molecular liquids, 1 mixture) with CG mappings and AA recipes, force-field overrides; see `benchmarks/README.md` |
| `cluster/` | Templates for a Slurm cluster and a workstation; site values go in an untracked `cluster/site.local.sh` |
| `docs/` | User guides and the pre-registered protocol |
| `eval/` | Agent evaluation harness and tasks |
| `tests/` | Unit and integration tests |

## Tests

```bash
python -m pip install -e '.[aa,dev]' && python -m pytest       # GPU tests run when FASTMD_BINARY and a GPU are available
```

## Licence and citation

MIT (see `LICENSE`); third-party components in `THIRD_PARTY.md`. Please cite it as given in `CITATION.cff`.
