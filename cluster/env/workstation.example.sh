# GPU workstation without a scheduler. Copy to cluster/env/<name>.sh, adjust the paths and point the compute target's
# `env` at it (~/.config/ascg/compute.yaml), e.g.
#   env: source /home/USER/ascg-agent/cluster/env/<name>.sh
export ASCG_ROOT="${ASCG_ROOT:-$HOME/ascg-agent}"
export PATH="$HOME/envs/ascg/bin:$PATH"                      # Python environment with ascg-agent installed
export FASTMD_ROOT="${FASTMD_ROOT:-$HOME/software/fastMD}"   # fastMD build (main b35e30a or later)
source "$ASCG_ROOT/cluster/env/common.sh"
