"""Reproducibility-capsule manifest + live-environment verifier (Step 122).

The capsule (`capsule/Dockerfile.reproduce`, `capsule/requirements.lock.txt`,
`capsule/reproduce.sh`) packages a one-command, from-scratch regeneration of
every deterministic artifact in the repository. This harness is the capsule's
self-description and its environment gate:

* As a **manifest** (default / ``--check``) it emits a deterministic JSON+MD
  record of the capsule -- the pinned wheel set parsed from the lock file, the
  base image and entrypoint, content hashes of the three capsule files, and the
  count of deterministic artifacts the capsule regenerates (read live from
  ``reproduce_all``) -- so the capsule's contract is itself a checkable artifact.

* As an **environment verifier** (``--verify-env``) it compares the *live*
  installed package versions against the pinned lock and exits non-zero on any
  mismatch. This is the "proven against real code" half: it runs against the
  actual interpreter the evidence pipeline will use, not a declared intention.

The deterministic manifest deliberately contains only static facts (pins,
hashes, structure), never live-resolved versions, so it is byte-identical across
machines; ``--check`` regenerates and diffs it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as ilm
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

LOCK = REPO / "capsule" / "requirements.lock.txt"
DOCKERFILE = REPO / "capsule" / "Dockerfile.reproduce"
REPRODUCE_SH = REPO / "capsule" / "reproduce.sh"

OUT_JSON = REPO / "reproducibility" / "capsule_manifest.json"
OUT_MD = REPO / "reproducibility" / "capsule_manifest.md"

BASE_IMAGE = "python:3.12-slim"
ENTRYPOINT = "bash capsule/reproduce.sh"
ONE_COMMAND = "docker run --rm tensorguard-capsule"

# pip metadata name -> import test (not all needed, kept for documentation).
_PIN_RE = re.compile(r"^([A-Za-z0-9_.\-]+)==([0-9][0-9A-Za-z.\-+]*)$")


def parse_lock() -> List[Tuple[str, str]]:
    pins: List[Tuple[str, str]] = []
    for raw in LOCK.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _PIN_RE.match(line)
        if m:
            pins.append((m.group(1), m.group(2)))
    return sorted(pins)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release_tuple(v: str) -> Tuple[int, ...]:
    # Drop any local/build suffix (e.g. +cpu) and non-numeric trailers.
    core = v.split("+", 1)[0]
    parts: List[int] = []
    for chunk in core.split("."):
        num = re.match(r"^\d+", chunk)
        parts.append(int(num.group(0)) if num else 0)
    return tuple(parts)


def satisfies(installed: str, pinned: str) -> bool:
    """True if `installed` is the pinned release (zero-padded equality)."""
    a, b = _release_tuple(installed), _release_tuple(pinned)
    n = max(len(a), len(b))
    a += (0,) * (n - len(a))
    b += (0,) * (n - len(b))
    return a == b


def build_manifest() -> Dict[str, object]:
    pins = parse_lock()
    import reproducibility.reproduce_all as ra

    deterministic = [
        p for p in ra.GENERATED_DETERMINISTIC
    ]
    return {
        "step": 122,
        "base_image": BASE_IMAGE,
        "entrypoint": ENTRYPOINT,
        "one_command": ONE_COMMAND,
        "torch_cpu_index": "https://download.pytorch.org/whl/cpu",
        "pinned_wheels": [{"name": n, "version": v} for n, v in pins],
        "n_pinned_wheels": len(pins),
        "capsule_file_sha256": {
            "requirements.lock.txt": _sha256(LOCK),
            "Dockerfile.reproduce": _sha256(DOCKERFILE),
            "reproduce.sh": _sha256(REPRODUCE_SH),
        },
        "n_deterministic_artifacts_regenerated": len(deterministic),
        "determinism_verified_by": "reproducibility/reproduce_all.py --check",
        "numeric_audit": "reproducibility/audit_numeric_claims.py",
    }


def verify_env() -> int:
    pins = parse_lock()
    print(f"capsule env check against {LOCK.name} ({len(pins)} pins)")
    all_ok = True
    for name, pinned in pins:
        try:
            installed: Optional[str] = ilm.version(name)
        except ilm.PackageNotFoundError:
            installed = None
        ok = installed is not None and satisfies(installed, pinned)
        all_ok = all_ok and ok
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {name}: pinned {pinned}, installed {installed}")
    if all_ok:
        print("capsule env: live environment satisfies every pin")
        return 0
    print("capsule env: MISMATCH — live environment does not satisfy the lock")
    return 1


def render_markdown(d: Dict[str, object]) -> str:
    lines = [
        "# Reproducibility capsule manifest (Step 122)",
        "",
        f"One command — `{d['one_command']}` — regenerates **"
        f"{d['n_deterministic_artifacts_regenerated']}** deterministic "
        "artifacts from source and verifies each is byte-identical to the "
        "committed tree, then re-audits every README numeric claim.",
        "",
        f"- base image: `{d['base_image']}`",
        f"- entrypoint: `{d['entrypoint']}`",
        f"- determinism gate: `{d['determinism_verified_by']}`",
        f"- numeric audit: `{d['numeric_audit']}`",
        "",
        "## Pinned wheels",
        "",
        "| package | version |",
        "| --- | --- |",
    ]
    for w in d["pinned_wheels"]:  # type: ignore[index]
        lines.append(f"| {w['name']} | {w['version']} |")
    lines += [
        "",
        "## Capsule file hashes (sha256)",
        "",
        "| file | sha256 |",
        "| --- | --- |",
    ]
    for f, h in d["capsule_file_sha256"].items():  # type: ignore[union-attr]
        lines.append(f"| `capsule/{f}` | `{h}` |")
    lines.append("")
    return "\n".join(lines)


def run(check: bool = False) -> int:
    d = build_manifest()
    js = json.dumps(d, indent=2, sort_keys=True) + "\n"
    md = render_markdown(d)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != js:
            print(f"MISMATCH: {OUT_JSON}")
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != md:
            print(f"MISMATCH: {OUT_MD}")
            ok = False
        if ok:
            print("capsule_manifest: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    if "--verify-env" in sys.argv:
        sys.exit(verify_env())
    sys.exit(run(check="--check" in sys.argv))
