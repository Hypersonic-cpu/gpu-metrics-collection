#!/usr/bin/env bash
# Supplement the failed/incomplete cases from tk_p1_rotation_20260814-092710.
set -euo pipefail

P1_CASES=(
  TK003 TK004 TK011 TK017 TK018 TK019 TK020
  TK050 TK053 TK054 TK057 TK058
)

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=${TK_PYTHON:-$HOME/Repos/ThunderKittens/.venv/bin/python3}
rotation_config=${ROTATION_CONFIG:-$HOME/Repos/ThunderKittens-RotationSize.json}
nsys_delay=${TK_NSYS_DELAY:-5}
case_csv=$(IFS=,; echo "${P1_CASES[*]}")

exec "$python" "$repo/scripts/profile_tk_p1.py" \
  --cases "$case_csv" --rotation-config "$rotation_config" \
  --nsys-delay "$nsys_delay" "$@"
