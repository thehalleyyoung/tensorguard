"""Step 68 — pre-commit hook entry point.

``pre-commit`` invokes a hook with the list of changed files as positional
arguments.  This entry point verifies the staged Python modules with TensorGuard
(honoring any ``tensorguard.toml``), prints a concise plain-text report, and
exits non-zero when a real bug is found so the commit is blocked.

The heavy lifting is the Step-66 :func:`src.github_action.run_action`; this
module only adds argument parsing and plain (non-annotation) rendering, both of
which are pure and tested.
"""

from __future__ import annotations

import argparse
from typing import List, Optional


def _format_report(result) -> str:
    """Plain-text summary of an ActionResult for a terminal / commit hook."""
    if result.total_issues == 0:
        return f"TensorGuard: verified {result.files_checked} file(s); no issues."
    lines: List[str] = [
        f"TensorGuard: {result.total_issues} issue(s) in "
        f"{result.files_with_issues} of {result.files_checked} file(s):",
    ]
    for ann in result.annotations:
        loc = f"{ann.file}:{ann.line}"
        if ann.col:
            loc += f":{ann.col}"
        lines.append(f"  {loc}: {ann.message}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorguard-precommit",
        description="Verify PyTorch modules in changed files before commit.",
    )
    parser.add_argument("files", nargs="*", help="Files to verify (from pre-commit).")
    parser.add_argument(
        "--soundness-mode",
        choices=["sound", "balanced", "heuristic"],
        default="balanced",
    )
    parser.add_argument(
        "--input-shapes",
        default="",
        help='Semicolon-separated specs, e.g. "x=batch,3,32,32".',
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    from src.github_action import run_action, _parse_shapes

    args = build_parser().parse_args(argv)
    paths = args.files or ["."]
    shapes = _parse_shapes(args.input_shapes)
    result = run_action(
        paths,
        soundness_mode=args.soundness_mode,
        input_shapes=shapes or None,
        fail_on="any",
    )
    print(_format_report(result))
    return 1 if result.failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
