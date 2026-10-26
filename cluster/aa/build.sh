#!/bin/bash
# Submit AutoPoly builds (aa/build_aa.slurm) on a Slurm cluster, one array task per system:
#   bash cluster/aa/build.sh RECIPES OUT [KEY ...]      (default: every task of RECIPES)
# e.g. bash cluster/aa/build.sh benchmarks/aa_recipes.json runs/aa pdms
# Then run cluster/aa/apply_autopoly_patch.sh once per AutoPoly checkout, before the first build.
set -eo pipefail
source "$(cd "$(dirname "$0")/../.." && pwd)/cluster/env/slurm.sh"
REC=$(realpath "${1:?recipes file}"); mkdir -p "${2:?output dir}/logs"; OUT=$(realpath "$2"); shift 2
if [ $# -gt 0 ]; then KEYS=$(IFS=+; echo "$*"); N=$#
else KEYS=""; N=$(python -c "import json, sys; print(len(json.load(open(sys.argv[1]))['tasks']))" "$REC"); fi
${DRYRUN:+echo} sbatch --array=1-$N --qos=$AA_BUILD_QOS --partition=$AA_CPU_PARTITION -o "$OUT/logs/build-%A_%a.out" \
  --export=ALL,RECIPES=$REC,OUT=$OUT,KEYS="$KEYS" "$ASCG_ROOT/cluster/aa/build_aa.slurm"
