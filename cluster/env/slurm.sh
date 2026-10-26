# Slurm cluster (login or compute nodes):
#   source <checkout>/cluster/env/slurm.sh
# Paths and QoS come from cluster/site.local.sh (copy cluster/site.example.sh). Slurm jobs find this file through
# ASCG_ROOT, which sbatch passes on from the submitting shell.
_ascg_here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$_ascg_here/cluster/site.local.sh" || { echo "cluster/site.local.sh missing (copy cluster/site.example.sh)" >&2; return 1; }
export ASCG_ROOT="$CLUSTER_ROOT"
export PATH="$CLUSTER_PYTHON_ENV/bin:$PATH"
export FASTMD_ROOT="${FASTMD_ROOT:-$CLUSTER_FASTMD_ROOT}"
source "$ASCG_ROOT/cluster/env/common.sh"
