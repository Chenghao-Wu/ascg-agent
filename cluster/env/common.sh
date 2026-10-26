# Shared part of the host environments (sourced by the files next to it): this checkout's sources on PYTHONPATH and,
# when FASTMD_ROOT is set, the fastMD build (command-line program and its numpy-only Python package).
export PYTHONPATH="$ASCG_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${FASTMD_ROOT:-}" ]; then
  export FASTMD_BINARY="$FASTMD_ROOT/build/fastmd"
  export PYTHONPATH="$PYTHONPATH:$FASTMD_ROOT/python"
fi
