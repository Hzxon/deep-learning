#!/usr/bin/env bash
# Pre-change / audit workflow wrapper.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
exec python scripts/workflow.py pre-change "$@"
