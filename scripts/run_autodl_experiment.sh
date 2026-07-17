#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/project/causal_ttl_project}"
CONFIG_PATH="${2:-$PROJECT_ROOT/configs/digestive_mock.yaml}"

cd "$PROJECT_ROOT"
python3 experiment_runner.py --config "$CONFIG_PATH"
