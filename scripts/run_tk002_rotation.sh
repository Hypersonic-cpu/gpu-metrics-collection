#!/usr/bin/env bash
# Native correctness + Nsys + DCGM for rotating-buffer TK002 only.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=${TK_PYTHON:-$HOME/Repos/ThunderKittens/.venv/bin/python3}

exec "$python" "$repo/scripts/profile_tk_p0.py" --cases TK002 "$@"
