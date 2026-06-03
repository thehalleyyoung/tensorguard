#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m reproducibility.reviewer_commands --check
python -m reproducibility.reviewer_commands --dry-run
