# Install

ascg runs from a clone of this repository: the Python package (`ascg`, `ascg_core`), the agent plugin and skills
(`plugin/ascg`) and the host templates (`cluster/`) come together. Remote hosts get the same code with
`ascg compute deploy`.

## Python

Python 3.10 or newer. RDKit is the dependency that most often needs care; conda-forge or the PyPI wheel both work.

```bash
git clone https://github.com/Chenghao-Wu/ascg-agent && cd ascg-agent
conda create -n ascg -c conda-forge python=3.11 rdkit && conda activate ascg     # or any Python >= 3.10
python -m pip install -e '.[aa]'          # aa: MDAnalysis, for reading your own AA trajectories
ascg tools                                # lists every tool; `ascg <group> <action> --help` shows one
```

- Core dependencies: numpy, scipy, matplotlib, RDKit, networkx, PyYAML.
- `aa` extra: MDAnalysis, needed only on the host that reads AA trajectories (`ascg ref inspect`, `ascg ref build`).
- `dev` extra: pytest. `python -m pytest` runs the CPU tests; the GPU tests run when `FASTMD_BINARY` is set and a GPU
  is free.

## fastMD (CG engine, on the GPU hosts)

[fastMD](https://github.com/Chenghao-Wu/fastMD) runs every CG simulation. ascg needs fastMD main b35e30a or later:
tabulated and combined bending-torsion dihedrals, and the symmetric NPT barostat. It needs CUDA and CMake.

For a target (here the workstation target `local` of [compute](compute.md)):

```bash
ascg engine setup --target local --dest $HOME/software/fastMD --cuda-arch 86    # prints the commands; runs nothing
```

The commands clone fastMD, check out the pinned commit, configure with your GPU architecture (86 for RTX 30xx /
A40, 89 for RTX 40xx / L40, 80 for A100, 90 for H100), build, and run ctest on a GPU. The target's environment must
point at the binary (`FASTMD_BINARY=$HOME/software/fastMD/build/fastmd` in its `--env` line); `ascg compute doctor`
then checks the build and runs the engine gate.

## Optional: building AA references yourself

Only needed when ascg should build and simulate the all-atom system (instead of [bringing your own](bring-your-own.md)
trajectories):
- [AutoPoly v2.0](https://github.com/WuGroup-XJTLU/AutoPoly/tree/v2.0) with the OPLS-AA additions in `patches/`
  (`cluster/aa/apply_autopoly_patch.sh`);
- GROMACS 2025 on the cluster;
- the Slurm templates in `cluster/` (copy `cluster/site.example.sh` to `cluster/site.local.sh` and fill in your
  paths, modules, partitions and accounts).

## Agents

- **Claude Code:** the plugin in `plugin/ascg` ([agents](agents.md#claude-code)).
- **Codex:** `ascg init codex --workspace DIR` ([agents](agents.md#codex)).

The plugin's `ascg` launcher finds a Python with ascg and RDKit by itself. Set `ASCG_PYTHON` if it picks the wrong
one.
