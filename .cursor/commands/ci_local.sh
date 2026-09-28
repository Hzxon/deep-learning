#!/usr/bin/env bash
# Local CI workflow wrapper.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
exec python scripts/workflow.py ci-local "$@"
