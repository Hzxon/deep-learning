#!/usr/bin/env bash
# Post-training audit workflow wrapper.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
exec python scripts/workflow.py post-training "$@"
