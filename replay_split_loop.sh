#!/usr/bin/env bash
# Replays a split model over a list of segments in chain mode, one python process per segment, skipping finished ones.
# Usage: replay_split_loop.sh <model dir> <out dir> <segment list file>
D=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python}
mkdir -p "$2"
while read -r seg; do
  out="$2/$(basename "$seg").npz"
  [ -f "$out" ] && continue
  "$PYTHON" "$D/replay_split.py" "$seg" "$1" "$out" 0 --chain < /dev/null > /dev/null 2>&1 && echo "ok $(basename "$seg")" || echo "failed $(basename "$seg")"
done < "$3"
echo "loop done"
