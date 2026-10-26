"""engine: fastMD (main at FASTMD_REF: tabulated / CBT dihedrals, NPT barostat fix) - selfcheck and build plan."""
from ..compute import remote, target
from ..registry import P, PolicyError, ToolError, tool

FASTMD_REPO = 'https://github.com/Chenghao-Wu/fastMD'
FASTMD_REF = 'b35e30a'                      # fastMD main with tabulated / CBT dihedrals (#3) and the NPT fix (#4)


@tool('engine', 'selfcheck', 'Deterministic engine gate on a target: fastMD energy of a toy melt vs numpy (1-2 / 1-3 '
      'exclusions), CBT dihedral energy vs numpy, and NPT stability of a CG liquid of small stiff molecules (builds '
      'without the barostat fix collapse). Workstations only; Slurm jobs run it first anyway.', {
          'target': P(required=True), 'gpu': P('integer', help='GPU index (default: first GPU without foreign '
                                                               'processes)'), 'dihedrals': P('boolean', default=True)})
def engine_selfcheck(dihedrals, gpu=None, **kw):
    t = target(kw['target'])
    if t['scheduler'] == 'slurm':
        raise PolicyError(f'{t["name"]} is a Slurm target: GPUs are only available inside jobs; every ascg job runs '
                          'the selfcheck before fitting (see workflow/ascg_job_*.out)')
    if gpu is None:
        free = [g['index'] for g in remote(t, 'gpus') if not g['foreign'] and
                (t.get('gpus') is None or g['index'] in t['gpus'])]
        if not free:
            raise PolicyError(f'{t["name"]}: no GPU free of other users\' processes')
        gpu = free[0]
    return remote(t, 'selfcheck', {'gpu': gpu, 'dihedrals': dihedrals}, timeout=600)


@tool('engine', 'setup', 'Build plan for fastMD (main at the commit ascg expects) on a target (prints the commands; does not run '
      'them - building on shared systems is the user\'s decision).', {
          'target': P(required=True), 'cuda_arch': P(default='86;89'), 'dest': P(help='install directory')})
def engine_setup(cuda_arch, dest=None, **kw):
    t = target(kw['target'])
    dest = dest or t['root'].rstrip('/') + '/../software/fastMD-ascg'
    cmds = [f'git clone {FASTMD_REPO} {dest}', f'git -C {dest} checkout {FASTMD_REF}',
            f'cmake -S {dest} -B {dest}/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="{cuda_arch}" '
            '-DFASTMD_BUILD_TESTS=ON -DFASTMD_BUILD_PYTHON=OFF', f'cmake --build {dest}/build -j 16',
            f'(on a GPU) ctest --test-dir {dest}/build --output-on-failure',
            f'point the target at the binary: its env line exports FASTMD_BINARY={dest}/build/fastmd (ascg compute add '
            f'--name {t["name"]} ... --env "export FASTMD_BINARY={dest}/build/fastmd"); then: ascg compute doctor '
            f'--target {t["name"]}']
    return {'target': t['name'], 'commands': cmds, 'note': 'run these yourself or ask the agent explicitly; '
            'on clusters load the CUDA and compiler modules first'}
