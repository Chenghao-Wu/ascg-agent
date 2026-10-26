#!/bin/bash
#SBATCH --job-name=aa-map
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=12:00:00
# Mapped CG reference of one task from every branch of its GROMACS protocol (protocol.json branch_dirs).
#   sbatch ... map_aa.slurm KEY PROTOCOL_ROOT OUTROOT SYSTEM_ROOT RECIPES [MAPPINGS]
set -eo pipefail
KEY=${1:?key}; PROOT=${2:?protocol root}; OUTROOT=${3:?out root}; SYSROOT=${4:?system root}; REC=${5:?recipes}
source "${ASCG_ROOT:?source cluster/env/slurm.sh before submitting}/cluster/env/slurm.sh"
MAPPINGS=${6:-$ASCG_ROOT/benchmarks/ascg_library.json}
RUNS=$(python -c "import json, sys; print(' '.join(f'--run {d[1:]}={sys.argv[2]}/{d}' for d in json.load(open(sys.argv[1]))['branch_dirs']))" \
       "$PROOT/protocol.json" "$PROOT")
mkdir -p "$OUTROOT"
echo "user $USER host $(hostname) start $(date)"
python -m ascg_core map-aa --system-dir "$SYSROOT/$KEY" --task-key "$KEY" --recipes "$REC" --mappings "$MAPPINGS" \
  $RUNS --out "$OUTROOT/$KEY" > "$OUTROOT/$KEY.summary.json"
echo "done $(date)"
