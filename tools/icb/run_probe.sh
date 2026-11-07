#!/bin/sh
# Compile and run the indirect-command probe, then print the verdict.
#
#   ./tools/icb/run_probe.sh
#
# Exit code follows the probe: 0 = indirect compute commands are usable here,
# 1 = they are broken, 2 = the harness itself failed (nothing to conclude).
#
# This exists because the decode replay layer in ds4_metal.m (DS4_QWEN4_ICB=1)
# depends on the driver executing indirect commands correctly, and the answer
# has already changed once: the first conclusion ("reads its inputs as zero")
# was a missing setBarrier in the probe, not a driver defect.
set -u

here=$(cd "$(dirname "$0")" && pwd)
out="${TMPDIR:-/tmp}/icb_probe"

echo "building $out"
if ! clang -fobjc-arc -fmodules -framework Metal -framework Foundation \
        -o "$out" "$here/icb_probe.m"; then
    echo "build failed" >&2
    exit 2
fi

"$out"
status=$?
echo
case $status in
    0) echo "verdict: ICB WORKS - the replay layer can be tested with DS4_QWEN4_ICB=1" ;;
    1) echo "verdict: ICB BROKEN - keep DS4_QWEN4_ICB unset" ;;
    *) echo "verdict: HARNESS FAILED - the direct path failed, so ICB is not implicated" ;;
esac
exit $status
