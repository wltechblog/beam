#!/bin/sh
# GPU soak test — self-judging verification that renders run on GPU silicon,
# not on the CPU (and not on a CPU-hosted OpenCL runtime).
#
# Verdicts:
#   PASS   completed under the budget  -> really on the GPU
#   FAIL   still running at the timeout -> CPU speed: something fell back
#   ERROR  render error / wrong device   -> see the diagnostics output
#
# Usage: ./run-test.sh [budget]        (budget in seconds, default 60)
#
# How the budget is sized (measured on the A770, this exact scene):
#   GPU: ~9s  (1920x1080 @ samples=8 = 64 rays/px)
#   CPU: ~170s (all-core Go renderer) — 45s kills it ~4x short of finishing
# So a CPU fallback never gets mistaken for a slow GPU: it's killed and
# reported as FAIL within the budget. A CPU-hosted OpenCL ICD (POCL etc.)
# is equally doomed — the kernel itself would run at CPU speed.
#
# The render also prints its device line ("gpu device: <name> <precision>")
# on success, which doubles as an identity check.

set -u
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BEAM="$DIR/../beam"
SCENE="$DIR/scene.json"
BUDGET=${1:-60}

# regenerate scene + knot mesh if missing (gen_soak.py is authoritative)
[ -f "$SCENE" ] || { echo "-- generating scene --"; python3 "$DIR/gen_soak.py" || exit 2; }

echo "== beam GPU soak: $(basename "$SCENE") budget=${BUDGET}s =="
echo "-- device identity --"
timeout 30s "$BEAM" -q -gpu -o /tmp/beam-soak-id.png "$DIR/../gpu-smoke/scene.json" 2>&1 || {
    echo "FAIL: GPU not reachable (smoke render failed)"; exit 1; }

echo "-- soak render (${BUDGET}s budget) --"
T0=$(date +%s.%N)
timeout "${BUDGET}s" "$BEAM" -gpu "$SCENE" 2>&1
RC=$?
T1=$(date +%s.%N)
WALL=$(python3 -c "print(f'{$T1-$T0:.1f}')")

if [ $RC -eq 124 ]; then
    echo "FAIL: exceeded ${BUDGET}s budget (wall ${WALL}s) — render is running at CPU speed; \
a GPU would finish in ~9s. Check GPU access and no-fallback diagnostics."
    exit 1
elif [ $RC -ne 0 ]; then
    echo "ERROR: render exited $RC (wall ${WALL}s)"
    exit 2
fi
echo "PASS: wall ${WALL}s (budget ${BUDGET}s) — GPU silicon confirmed by timing."
exit 0
