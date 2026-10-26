#!/bin/bash
# Append the ASCG OPLS typing additions to the pinned AutoPoly clone (idempotent)
# and record the resulting file hash next to it.
set -eo pipefail
ROOT=${ASCG_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
F=$ROOT/external/AutoPoly/AutoPoly/extern/rdlt_data/opls_lt_2024.fdefn
ADD=$ROOT/patches/opls_lt_2024_ascg_additions.fdefn
# Start from the pinned upstream file every time, so edits to the additions apply cleanly.
git -C "$ROOT/external/AutoPoly" checkout -- AutoPoly/extern/rdlt_data/opls_lt_2024.fdefn
cat "$ADD" >> "$F"
sha256sum "$F" "$ADD" > "$ROOT/external/AutoPoly.ascg_patch.sha256"
git -C "$ROOT/external/AutoPoly" diff --stat
