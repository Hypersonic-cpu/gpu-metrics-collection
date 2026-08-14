#!/usr/bin/env bash
# Edit this whitespace-separated list to select the P1 cases to profile.
set -euo pipefail

P1_CASES=(
  TK003 TK004 TK008 TK010 TK011 TK012 TK013 TK014 TK016 TK017 TK018 TK019 TK020  # Seed-OSS-36B
  # TK023 TK024 TK027 TK028 TK029 TK031 TK032 TK033 TK035 TK036 TK038 TK039 TK040  # Qwen3-30B-A3B
  TK044 TK045 TK046 TK047 TK049 TK050                                      # DeepSeek-V4-Flash
  TK053 TK054 TK057 TK058 TK059 TK061 TK062 TK063 TK065 TK066 TK068 TK069 TK070  # Qwen3-235B-A22B
)

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=${TK_PYTHON:-$HOME/Repos/ThunderKittens/.venv/bin/python3}
rotation_config=${ROTATION_CONFIG:-$HOME/Repos/ThunderKittens-RotationSize.json}
case_csv=$(IFS=,; echo "${P1_CASES[*]}")

exec "$python" "$repo/scripts/profile_tk_p1.py" \
  --cases "$case_csv" --rotation-config "$rotation_config" "$@"
