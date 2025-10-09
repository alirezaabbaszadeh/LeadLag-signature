#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"

python -m leadlag_rl.training.train "$@"
