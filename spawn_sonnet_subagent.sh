#!/bin/bash
# Spawn a focused Sonnet-4.6 subagent. Usage:
#   ./spawn_sonnet_subagent.sh "do this task"
# Or with a prompt file:
#   ./spawn_sonnet_subagent.sh -p @prompt.md
set -euo pipefail
exec copilot --model claude-sonnet-4.6 --allow-all-tools "$@"
