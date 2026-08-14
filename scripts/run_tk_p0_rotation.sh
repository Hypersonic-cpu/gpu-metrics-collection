#!/usr/bin/env bash
# Edit this whitespace-separated list to select the P0 cases to profile.
set -euo pipefail

P0_CASES=(
  TK002 TK005 TK006 TK007 TK022 TK025 TK026 TK034
  TK041 TK042 TK043 TK048 TK052 TK055 TK056 TK064
)

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=${TK_PYTHON:-$HOME/Repos/ThunderKittens/.venv/bin/python3}
rotation_config=${ROTATION_CONFIG:-$HOME/Repos/ThunderKittens-RotationSize.json}
case_csv=$(IFS=,; echo "${P0_CASES[*]}")

exec "$python" "$repo/scripts/profile_tk_p0.py" \
  --cases "$case_csv" --rotation-config "$rotation_config" "$@"
