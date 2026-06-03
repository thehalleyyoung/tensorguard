#!/usr/bin/env python3
"""validate_entry.py — CI gate for community leaderboard submissions.

A pull request that drops a tool entry in
``benchmarks/leaderboard_entries/<tool>.json`` must pass this validator before
the leaderboard re-scores it.  The validator never trusts a submitter's
self-reported metrics — it only checks that the *raw per-case verdicts* are
well-formed against the frozen corpus, so the rescoring in ``leaderboard.py``
can recompute recall/precision/F1 honestly:

* the file is a JSON object with a non-empty ``tool`` name, a ``verdicts``
  mapping, and a detached SSH signature from a key in
  ``benchmarks/leaderboard_entries/allowed_signers``;
* every verdict value is one of ``SAFE`` / ``UNSAFE`` / ``UNKNOWN`` (case
  insensitive);
* every case id is a real id in the frozen ``real_benchmarks`` manifest
  (a missing id is allowed and treated as ``UNKNOWN``, but an *unknown* id is a
  typo and rejected);
* the entry carries **no** self-reported metric fields (``recall``,
  ``precision``, ``f1``, ``accuracy``, ``tp`` …): scoring is recomputed, never
  submitted.

Usage::

    python reproducibility/validate_entry.py benchmarks/leaderboard_entries/*.json

Exit code is non-zero if any file is invalid; every error is printed with its
file and offending field so a contributor can fix it from the CI log.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Tuple

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from real_benchmarks.load import load_items  # noqa: E402

VALID_VERDICTS = {"SAFE", "UNSAFE", "UNKNOWN"}
SIGNATURE_NAMESPACE = "tensorguard-leaderboard-v1"
ALLOWED_SIGNERS = REPO / "benchmarks" / "leaderboard_entries" / "allowed_signers"
# Metric fields a submitter must NOT provide — the leaderboard recomputes them.
FORBIDDEN_METRIC_KEYS = {
    "recall", "precision", "f1", "accuracy",
    "tp", "fp", "tn", "fn", "scorecard",
}


def valid_case_ids() -> set:
    """The set of real, hash-pinned case ids in the frozen corpus."""
    return {it["id"] for it in load_items(verify=True)}


def canonical_signed_payload(raw: dict) -> bytes:
    """Canonical bytes covered by a leaderboard entry's detached signature.

    The signature envelope itself is excluded. Verdict tokens are normalized to
    uppercase before signing so validation and scoring use the same semantics.
    """
    payload: dict[str, Any] = {
        str(k): v for k, v in raw.items()
        if k != "signature"
    }
    verdicts = payload.get("verdicts")
    if isinstance(verdicts, dict):
        payload["verdicts"] = {
            str(k): str(v).upper()
            for k, v in sorted(verdicts.items(), key=lambda item: str(item[0]))
        }
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def validate_signature(
    raw: dict,
    allowed_signers: Path = ALLOWED_SIGNERS,
) -> List[str]:
    """Verify the detached SSH signature against the maintainer trust anchor."""
    sig = raw.get("signature")
    if not isinstance(sig, dict):
        return ["missing 'signature' object"]

    identity = sig.get("identity")
    namespace = sig.get("namespace")
    value = sig.get("value")
    if not isinstance(identity, str) or not identity.strip():
        return ["signature.identity must be a non-empty string"]
    if namespace != SIGNATURE_NAMESPACE:
        return [f"signature.namespace must be {SIGNATURE_NAMESPACE!r}"]
    if not isinstance(value, str) or "BEGIN SSH SIGNATURE" not in value:
        return ["signature.value must be an armored SSH signature"]
    if not allowed_signers.exists():
        return [f"allowed signers file not found: {allowed_signers}"]
    ssh_keygen = shutil.which("ssh-keygen")
    if not ssh_keygen:
        return ["ssh-keygen is required to verify leaderboard signatures"]

    with tempfile.TemporaryDirectory() as td:
        payload_path = Path(td) / "payload.json"
        sig_path = Path(td) / "payload.sig"
        payload_path.write_bytes(canonical_signed_payload(raw))
        sig_path.write_text(value if value.endswith("\n") else value + "\n")
        proc = subprocess.run(
            [
                ssh_keygen,
                "-Y", "verify",
                "-f", str(allowed_signers),
                "-I", identity,
                "-n", SIGNATURE_NAMESPACE,
                "-s", str(sig_path),
            ],
            input=payload_path.read_bytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).decode("utf-8", errors="replace").strip()
        return [f"invalid leaderboard signature for {identity!r}: {detail}"]
    return []


def validate_entry(
    raw: dict,
    corpus_ids: set,
    allowed_signers: Path = ALLOWED_SIGNERS,
    require_signature: bool = True,
) -> List[str]:
    """Return a list of human-readable problems (empty == valid)."""
    problems: List[str] = []

    if not isinstance(raw, dict):
        return [f"top level must be a JSON object, got {type(raw).__name__}"]

    tool = raw.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        problems.append("missing/empty 'tool' (a non-empty string is required)")

    signature = raw.get("signature")
    if "signature" in raw and not isinstance(signature, dict):
        problems.append("'signature' must be an object")

    forbidden = FORBIDDEN_METRIC_KEYS & set(raw)
    if forbidden:
        problems.append(
            "self-reported metric field(s) not allowed (scoring is recomputed): "
            + ", ".join(sorted(forbidden))
        )

    verdicts = raw.get("verdicts")
    if not isinstance(verdicts, dict) or not verdicts:
        problems.append("missing/empty 'verdicts' object")
        return problems

    for cid, verdict in verdicts.items():
        if str(cid) not in corpus_ids:
            problems.append(f"unknown corpus case id: {cid!r}")
        token = str(verdict)
        if token != token.upper():
            problems.append(f"verdict for {cid!r} must be uppercase, got {verdict!r}")
        if token.upper() not in VALID_VERDICTS:
            problems.append(
                f"invalid verdict {verdict!r} for {cid!r} "
                f"(expected one of {sorted(VALID_VERDICTS)})"
            )
    if require_signature:
        problems.extend(validate_signature(raw, allowed_signers))
    return problems


def validate_path(
    path: Path,
    corpus_ids: set,
    allowed_signers: Path = ALLOWED_SIGNERS,
) -> List[str]:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"invalid JSON: {exc}"]
    return validate_entry(raw, corpus_ids, allowed_signers=allowed_signers)


def main(argv: List[str]) -> int:
    paths = [Path(p) for p in argv]
    if not paths:
        entries_dir = REPO / "benchmarks" / "leaderboard_entries"
        paths = sorted(entries_dir.glob("*.json"))
    if not paths:
        print("no leaderboard entries to validate")
        return 0

    corpus_ids = valid_case_ids()
    failures: List[Tuple[Path, List[str]]] = []
    for path in paths:
        problems = validate_path(path, corpus_ids)
        if problems:
            failures.append((path, problems))
        else:
            print(f"OK   {path}")

    for path, problems in failures:
        for p in problems:
            print(f"FAIL {path}: {p}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
