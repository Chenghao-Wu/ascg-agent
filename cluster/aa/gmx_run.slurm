#!/bin/bash
#SBATCH --job-name=aa-gmx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
# GROMACS protocol runner: every unfinished step of the given protocol directories in order (e.g. ROOT/eq21
# ROOT/T400 ROOT/T500), one rank x OpenMP threads; resumes from .cpt. Submitted by aa/launch_gmx.sh.
#   sbatch ... gmx_run.slurm cpu|gpu DIR [DIR ...]      (MAXH: mdrun -maxh, below the job's time limit)
set -eo pipefail
MODE=${1:?cpu|gpu}; shift
source "${ASCG_ROOT:?source cluster/env/slurm.sh before submitting}/cluster/env/slurm.sh"
module purge; module load "$CLUSTER_GROMACS_MODULE"
[ -z "${AA_REQUIRE_CPU_FLAG:-}" ] || grep -q "$AA_REQUIRE_CPU_FLAG" /proc/cpuinfo || { echo "ERROR: $(hostname) lacks $AA_REQUIRE_CPU_FLAG (see AA_REQUIRE_CPU_FLAG in site.local.sh)"; exit 2; }
echo "user $USER host $(hostname) mode $MODE cpus $SLURM_CPUS_PER_TASK gpu ${CUDA_VISIBLE_DEVICES:-none} start $(date)"
GPUF=""; [ "$MODE" = gpu ] && GPUF="-nb gpu -pme gpu -bonded gpu -update auto"
python -m ascg_core.aa.gmx_protocol "$@" --mdrun "mpirun -np 1 gmx_mpi mdrun" \
  --mdrun-flags "-ntomp $SLURM_CPUS_PER_TASK" --gpu-flags "$GPUF" --maxh "${MAXH:-46}"
for d in "$@"; do grep -h "^Performance" $d/*.log 2>/dev/null | tail -1 | sed "s#^#$d #"; done
echo "done $(date)"
