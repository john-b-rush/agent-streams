#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install uv first: https://github.com/astral-sh/uv" >&2
  exit 1
fi

uv tool install --editable "$SCRIPT_DIR" --force

echo "Installed: agent-streams (uv tool)"
echo "Ensure your tool bin dir is on PATH (commonly: ~/.local/bin)"
