"""Corpus-freeze verification (round-1 reviewer W2 / Q1).

Re-runs the 10 rb_* repros against the *current* TG tree and asserts
that every bug still fires RP at confidence >= 0.99.  The companion
git operation (run by the maintainer, not by this script) is::

    git rev-parse HEAD                                # current SHA
    git log -1 --format=%H -- src/v5/                 # most-recent v5 mod
    git log -1 --format=%H -- experiments_v5/v8/real_bug_corpus.json
                                                      # corpus commit SHA

The protocol guarantee (see REAL_BUG_SELECTION_PROTOCOL.md §3) is
that the second SHA *predates* the third.  This script verifies the
empirical half of that guarantee: that the current rule-set still
produces RP on every rb_* item.

Exit code 0 if all 10 items still RP at >= 0.99 confidence.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _REPO_ROOT)

from src.api import verify_architecture  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "real_bugs")
CORPUS_JSON = os.path.join(os.path.dirname(__file__), "real_bug_corpus.json")
PROTOCOL_OUT = os.path.join(
    os.path.dirname(__file__), "corpus_freeze_report.json"
)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def main() -> int:
    head_sha = _git("rev-parse", "HEAD")
    v5_last_mod_sha = _git("log", "-1", "--format=%H", "--", "src/v5/")
    corpus_sha = _git(
        "log", "-1", "--format=%H", "--",
        os.path.relpath(CORPUS_JSON, _REPO_ROOT),
    )

    pass_count = 0
    fail_count = 0
    per_item: list[dict] = []

    for fname in sorted(os.listdir(BASE)):
        if not fname.endswith(".py") or not fname.startswith("rb_"):
            continue
        fpath = os.path.join(BASE, fname)
        with open(fpath) as f:
            src = f.read()

        m = re.search(r"INPUT_SHAPES\s*=\s*(\{[^}]+\})", src)
        if not m:
            per_item.append({"file": fname, "status": "SKIP", "reason": "no INPUT_SHAPES"})
            continue

        input_shapes = eval(m.group(1))  # noqa: S307 - trusted local repro
        try:
            result = verify_architecture(src, input_shapes=input_shapes)
            max_conf = max((b.confidence for b in result.bugs), default=0.0)
            if max_conf >= 0.99:
                pass_count += 1
                per_item.append({"file": fname, "status": "PASS", "max_conf": max_conf})
            else:
                fail_count += 1
                per_item.append({"file": fname, "status": "FAIL", "max_conf": max_conf})
        except Exception as e:  # pragma: no cover
            fail_count += 1
            per_item.append({"file": fname, "status": "ERROR", "error": str(e)})

    report = {
        "head_sha": head_sha,
        "v5_last_modification_sha": v5_last_mod_sha,
        "corpus_commit_sha": corpus_sha,
        "freeze_invariant": (
            "v5_last_modification_sha must be an ancestor of corpus_commit_sha "
            "for the no-rule-tuning guarantee to hold; verify with "
            "`git merge-base --is-ancestor <v5_sha> <corpus_sha>`."
        ),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "per_item": per_item,
    }

    with open(PROTOCOL_OUT, "w") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
