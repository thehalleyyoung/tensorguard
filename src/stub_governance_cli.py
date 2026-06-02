"""CLI for community stub-registry governance (Step 178).

Used by ``.github/workflows/stub-registry.yml`` and locally::

    python -m src.stub_governance_cli --check community_stubs/

Exits non-zero if any manifest in the directory (or the given files) fails
validation, printing a per-manifest report.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from src.stub_governance import (
    ValidationReport,
    validate_directory,
    validate_manifest_file,
)


def _print_report(report: ValidationReport) -> None:
    where = report.source or report.class_name or "<manifest>"
    if report.ok:
        print(f"  ok   {where}  ({report.cases_checked} conformance case(s))")
    else:
        print(f"  FAIL {where}")
        for err in report.errors:
            print(f"         - {err}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate community shape-stub manifests.")
    parser.add_argument("--check", metavar="PATH", action="append", default=[],
                        help="A directory of *.json manifests or a single manifest file.")
    parser.add_argument("paths", nargs="*", help="Manifest files/directories to check.")
    args = parser.parse_args(argv)

    targets = list(args.check) + list(args.paths)
    if not targets:
        targets = ["community_stubs"]

    reports: List[ValidationReport] = []
    for target in targets:
        import os
        if os.path.isdir(target):
            reports.extend(validate_directory(target))
        else:
            reports.append(validate_manifest_file(target))

    if not reports:
        print("No manifests found to validate.")
        return 0

    print(f"Validating {len(reports)} community stub manifest(s):")
    for report in reports:
        _print_report(report)

    failed = [r for r in reports if not r.ok]
    print(f"\n{len(reports) - len(failed)}/{len(reports)} valid.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
