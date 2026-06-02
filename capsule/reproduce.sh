#!/usr/bin/env bash
# TensorGuard reproducibility capsule — the one command (Step 122).
#
# Regenerates every deterministic artifact from source, verifies it is
# byte-identical to what is committed, and re-audits every numeric claim in the
# README. Exits non-zero the moment anything fails to reproduce.
#
# Run locally (no Docker required):
#   bash capsule/reproduce.sh
# or inside the capsule image:
#   docker run --rm tensorguard-capsule
set -euo pipefail

# Resolve the repository root regardless of where the script is invoked from.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export PYTHONHASHSEED=0

echo "==> capsule manifest (environment satisfies the pinned lock?)"
python3 reproducibility/capsule_manifest.py --verify-env

echo "==> from-scratch reproduction + byte-identical determinism check"
python3 reproducibility/reproduce_all.py --check

echo "==> capsule reproduction PASS"
