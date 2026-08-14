#!/usr/bin/env bash
# One command for native correctness + Nsys + DCGM on the two rotating P0 cases.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=${TK_PYTHON:-$HOME/Repos/ThunderKittens/.venv/bin/python3}

exec "$python" "$repo/scripts/profile_tk_p0.py" \
  --cases TK002,TK026 "$@"
