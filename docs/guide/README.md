# ascg-agent user guide

| Guide | For |
|---|---|
| [Quick start](quickstart.md) | install, a first mapping, a GPU, and a first model from your own AA trajectory |
| [Install](install.md) | Python environment, fastMD, the optional AA build tools |
| [Compute](compute.md) | a local GPU workstation or your Slurm cluster: `compute add`, `deploy`, `doctor` |
| [Agents](agents.md) | Claude Code (plugin) and Codex (`ascg init codex`); what stays with you |
| [Bring your own AA trajectory](bring-your-own.md) | GROMACS / LAMMPS / MDAnalysis files to a mapped reference |
| [Systems](systems.md) | system libraries, mappings, AutoPoly recipes and AA builds, state points |
| [Results](results.md) | outcome classes, reports, the model files and how to run them |
| [Troubleshooting](troubleshooting.md) | common problems and stopped fits |

Reference:
- [Protocol v1](../PROTOCOL_v1.md): every setting, gate and outcome rule of the fit;
- [Mapping formats](../../plugin/ascg/skills/ascg-mapping/references/formats.md);
- [Evaluation harness](../../eval/README.md): the agents on fixed tasks, with frozen tools;
- `ascg tools`, and `ascg <group> <action> --help`: every tool and its parameters.
