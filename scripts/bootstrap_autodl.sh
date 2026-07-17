#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/project/causal_ttl_project}"

python3 -m pip install --upgrade pip
python3 -m pip install -r "$PROJECT_ROOT/requirements.txt"

mkdir -p "$PROJECT_ROOT/outputs"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/data/processed"

echo "Bootstrap complete."
echo "Project root: $PROJECT_ROOT"
