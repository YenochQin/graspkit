#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_cmd="$project_root/../graspkit-tools/.venv/bin/python"
if [[ ! -x "$python_cmd" ]]; then
    echo "Set up the shared environment from graspkit-tools with uv sync --extra cpu." >&2
    exit 1
fi
exec "$python_cmd" "$project_root/build_package.py" "$@"
