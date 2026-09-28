#!/usr/bin/env bash
# Pre-training validation workflow wrapper.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
exec python scripts/workflow.py pre-training "$@"
