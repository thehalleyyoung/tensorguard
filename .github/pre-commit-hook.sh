#!/bin/bash
# TensorGuard pre-commit hook: verify tensor shapes in staged .py files
STAGED=$(git diff --cached --name-only --diff-filter=ACM -- '*.py')
if [ -n "$STAGED" ]; then
    echo "🔍 TensorGuard: Checking tensor shapes..."
    python -m src.cli.main verify $STAGED --format=compact
    if [ $? -ne 0 ]; then
        echo "❌ TensorGuard found shape errors. Fix them before committing."
        exit 1
    fi
    echo "✅ TensorGuard: All shapes verified."
fi
