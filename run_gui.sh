#!/usr/bin/env bash
set -euo pipefail

# Stable launcher for electricUniform.
# Always runs the GUI and all builds from the project that contains this file.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing Python environment: $PROJECT_DIR/.venv" >&2
  echo "Create it once with: python3 -m venv .venv" >&2
  exit 1
fi

# Load ROOT when the project is opened from a desktop launcher/new terminal.
# Keep an already configured environment untouched.
if ! command -v root-config >/dev/null 2>&1; then
  for setup in \
    "$HOME/root-install/bin/thisroot.sh" \
    "/opt/root/bin/thisroot.sh"; do
    if [[ -f "$setup" ]]; then
      # shellcheck disable=SC1090
      source "$setup"
      break
    fi
  done
fi

# Activate the project Python environment.
# shellcheck disable=SC1091
source .venv/bin/activate

# Use the Qt libraries bundled with PySide6, while preserving ROOT/Garfield
# library paths already present in LD_LIBRARY_PATH.
PYSIDE6_DIR="$(
python3 - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("PySide6")
if spec is None or spec.origin is None:
    raise SystemExit("PySide6 is not installed in .venv")
print(Path(spec.origin).parent)
PY
)"

export LD_LIBRARY_PATH="$PYSIDE6_DIR/Qt/lib:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="$PYSIDE6_DIR/Qt/plugins"

exec python3 "$PROJECT_DIR/gui.py"
