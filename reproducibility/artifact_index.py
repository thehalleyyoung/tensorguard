"""Tamper-evident generated-artifact index.

This ledger records the SHA-256 digest and byte size of every artifact owned by
the reproducibility pipeline. It is derived from ``reproduce_all`` rather than a
hand-maintained list, so adding a generated artifact to the pipeline
automatically adds it to the hash index.

The index files themselves are listed with an explicit self-hash policy instead
of a normal digest to avoid a recursive fixed-point hash.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_JSON = REPO / "reproducibility" / "artifact_index.json"
OUT_MD = REPO / "reproducibility" / "artifact_index.md"
SELF_PATHS = {
    "reproducibility/artifact_index.json",
    "reproducibility/artifact_index.md",
}
EXTRA_GENERATED_ARTIFACTS = [
    "tool_paper.pdf",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generated_paths() -> List[str]:
    import reproducibility.reproduce_all as ra

    paths = (
        list(ra.GENERATED_DETERMINISTIC)
        + list(ra._corpus_repro_paths())
        + list(ra._corpus_extended_paths())
        + list(ra.VOLATILE_REGENERATED)
        + EXTRA_GENERATED_ARTIFACTS
    )
    return sorted(dict.fromkeys(paths))


def _category(rel: str) -> str:
    if rel in SELF_PATHS:
        return "self_index"
    if rel.startswith("real_benchmarks/") or rel.startswith("corpus_extended/"):
        return "generated_corpus"
    if rel.endswith(".pdf"):
        return "generated_pdf"
    if rel in {"SOUNDNESS_CONTRACT.md", "VERIFIABLE_FRAGMENT.md", "formal_soundness_appendix.tex"}:
        return "generated_documentation"
    if rel.endswith(".json") or rel.endswith(".md") or rel.endswith(".html"):
        return "generated_evidence"
    return "generated_artifact"


def _entry(rel: str) -> Dict[str, object]:
    path = REPO / rel
    present = path.exists()
    entry: Dict[str, object] = {
        "path": rel,
        "category": _category(rel),
        "present": present,
    }
    if rel in SELF_PATHS:
        entry["bytes"] = None
        entry["sha256"] = None
        entry["hash_policy"] = "self-referential index file; checked byte-identical by --check"
    else:
        entry["bytes"] = path.stat().st_size if present else None
        entry["sha256"] = _sha256(path) if present else None
        entry["hash_policy"] = "sha256"
    return entry


def _root_digest(entries: Iterable[Dict[str, object]]) -> str:
    rows = []
    for entry in entries:
        if entry["path"] in SELF_PATHS:
            continue
        rows.append(f"{entry['path']}\0{entry['sha256']}\0{entry['bytes']}")
    payload = "\n".join(sorted(rows)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_index() -> Dict[str, object]:
    entries = [_entry(rel) for rel in _generated_paths()]
    hashed = [e for e in entries if e["path"] not in SELF_PATHS]
    return {
        "step": 264,
        "purpose": "tamper-evident SHA-256 ledger for generated reproducibility artifacts",
        "source_of_truth": "reproducibility/reproduce_all.py",
        "check_command": "python reproducibility/artifact_index.py --check",
        "n_artifacts_indexed": len(entries),
        "n_hashed_artifacts": len(hashed),
        "n_missing_artifacts": sum(1 for e in entries if not e["present"]),
        "all_hashed_artifacts_present": all(e["present"] for e in hashed),
        "artifact_root_sha256": _root_digest(entries),
        "self_index_paths": sorted(SELF_PATHS),
        "artifacts": entries,
    }


def render_markdown(data: Dict[str, object]) -> str:
    lines = [
        "# Generated artifact hash index (Step 264)",
        "",
        "This is a tamper-evident ledger for every generated artifact owned by "
        "`reproducibility/reproduce_all.py`. Each non-index artifact is recorded "
        "with byte size and SHA-256; the aggregate root digest changes if any "
        "artifact path, size, or content hash changes.",
        "",
        f"- artifacts indexed: **{data['n_artifacts_indexed']}**",
        f"- hashed artifacts: **{data['n_hashed_artifacts']}**",
        f"- missing artifacts: **{data['n_missing_artifacts']}**",
        f"- artifact root sha256: `{data['artifact_root_sha256']}`",
        f"- check command: `{data['check_command']}`",
        "",
        "| path | category | bytes | sha256 |",
        "| --- | --- | ---: | --- |",
    ]
    for entry in data["artifacts"]:  # type: ignore[index]
        digest = entry["sha256"] or entry["hash_policy"]
        lines.append(
            f"| `{entry['path']}` | {entry['category']} | "
            f"{entry['bytes'] if entry['bytes'] is not None else '-'} | `{digest}` |"
        )
    lines.append("")
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = build_index()
    js = json.dumps(data, indent=2, sort_keys=True) + "\n"
    md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != js:
            print(f"MISMATCH: {OUT_JSON}")
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != md:
            print(f"MISMATCH: {OUT_MD}")
            ok = False
        if data["n_missing_artifacts"]:
            print(f"MISSING: {data['n_missing_artifacts']} indexed artifacts")
            ok = False
        if ok:
            print("artifact_index: byte-identical; hashes match current artifacts")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
