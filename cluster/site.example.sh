# Site configuration of a Slurm cluster. Copy to cluster/site.local.sh (not tracked) and replace the upper-case
# placeholders. Sourced by cluster/env/slurm.sh, and so by every cluster/aa script.

CLUSTER_WORK=/path/to/WORK                                   # your work (project / scratch) directory
CLUSTER_ROOT=$CLUSTER_WORK/ascg-agent                        # this repository's checkout on the cluster
CLUSTER_PYTHON_ENV=$CLUSTER_WORK/envs/ascg                   # Python environment; its bin/ goes first on PATH
CLUSTER_FASTMD_ROOT=$CLUSTER_WORK/software/fastMD            # fastMD main b35e30a or later
CLUSTER_GROMACS_MODULE=gromacs/VERSION                       # module loaded by aa/gmx_run.slurm
AA_REQUIRE_CPU_FLAG=                                         # e.g. avx512f if the GROMACS build needs it; empty: no check
AA_CPU_PARTITION=CPU_PARTITIONS                              # e.g. cpuA,cpuB
AA_GPU_PARTITION=GPU_PARTITION                               # GPU nodes the GROMACS build runs on
AA_GPU_QOS=(GPU_QOS)                                         # rotated over GPU jobs
AA_BUILD_QOS=CPU_QOS                                         # AutoPoly builds (aa/build_aa.slurm)

# site_account sets, for the submitting user:
#   RUNBASE  where runs are written
#   AA_QOS   one entry per concurrent 16-core AA job, rotated (to fill per-user CPU limits)
#   MAP_QOS  QoS of the mapping jobs
site_account() {
  RUNBASE=$CLUSTER_ROOT/runs; AA_QOS=(CPU_QOS); MAP_QOS=CPU_QOS
}
# The --account option a QoS needs (prints nothing when none is needed), e.g.
#   case "$1" in SPECIAL_QOS) echo --account=BILLING_ACCOUNT ;; esac
qos_account() {
  :
}
