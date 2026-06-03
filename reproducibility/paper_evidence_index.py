"""Single-target paper-evidence index (Step 125).

`make paper-evidence` regenerates every table and figure the paper draws on and
then builds this index: a single catalogue of every paper-facing artifact, the
script that generates it, and whether it currently contains a rendered table.
The index is the one place a co-author (or a reviewer) can see, at a glance, that
every claim in the write-up is backed by a regenerable artifact.

The catalogue is derived from ``reproduce_all.GENERATED_DETERMINISTIC`` (the set
of byte-deterministic artifacts the from-scratch pipeline owns), so it cannot
silently drift out of sync with what the pipeline actually produces: a new
harness wired into the pipeline appears here automatically, and an artifact whose
generator script goes missing is flagged. ``--check`` regenerates and diffs it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_JSON = REPO / "reproducibility" / "paper_evidence_index.json"
OUT_MD = REPO / "reproducibility" / "paper_evidence_index.md"


def _has_markdown_table(path: Path) -> bool:
    if not path.exists():
        return False
    return "| --- |" in path.read_text() or "| ---" in path.read_text()


def _generator_for(stem: str) -> str | None:
    if stem == "task_packet":
        return "reproducibility/developer_study.py"
    # Artifacts follow reproducibility/<stem>.{json,md} <- reproducibility/<stem>.py
    cand = REPO / "reproducibility" / f"{stem}.py"
    if cand.exists():
        return f"reproducibility/{stem}.py"
    # A few artifacts are produced by evaluation/ or src/ scripts.
    for sub in ("evaluation", "src"):
        c = REPO / sub / f"{stem}.py"
        if c.exists():
            return f"{sub}/{stem}.py"
    return None


def measure() -> Dict[str, object]:
    import reproducibility.reproduce_all as ra

    # Group the deterministic artifacts by stem (a json/md pair share a stem).
    stems: Dict[str, Dict[str, str]] = {}
    for rel in ra.GENERATED_DETERMINISTIC:
        p = Path(rel)
        if p.suffix not in (".json", ".md"):
            continue  # VERSION files, html, etc. are not paper tables/figures
        stem = p.stem
        stems.setdefault(stem, {})
        if p.suffix == ".json":
            stems[stem]["json"] = rel
        elif p.suffix == ".md":
            stems[stem]["md"] = rel

    entries: List[dict] = []
    for stem in sorted(stems):
        files = stems[stem]
        md_rel = files.get("md")
        json_rel = files.get("json")
        gen = _generator_for(stem)
        has_table = _has_markdown_table(REPO / md_rel) if md_rel else False
        entries.append({
            "stem": stem,
            "json": json_rel,
            "md": md_rel,
            "generator": gen,
            "generator_present": gen is not None,
            "renders_table": has_table,
            "json_present": bool(json_rel) and (REPO / json_rel).exists(),
            "md_present": bool(md_rel) and (REPO / md_rel).exists(),
        })

    return {
        "step": 125,
        "make_target": "paper-evidence",
        "regenerates_via": "reproducibility/reproduce_all.py",
        "n_evidence_items": len(entries),
        "n_with_table": sum(1 for e in entries if e["renders_table"]),
        "n_missing_generator": sum(1 for e in entries
                                   if not e["generator_present"]),
        "all_generators_present": all(e["generator_present"] for e in entries),
        "all_artifacts_present": all(
            e["json_present"] or e["md_present"] for e in entries),
        "evidence": entries,
    }


def render_markdown(d: Dict[str, object]) -> str:
    lines = [
        "# Paper-evidence index (Step 125)",
        "",
        f"`make {d['make_target']}` regenerates every table and figure via "
        f"`{d['regenerates_via']}` and rebuilds this catalogue. "
        f"**{d['n_evidence_items']}** evidence items "
        f"({d['n_with_table']} render a table); every generator script is "
        f"present: **{d['all_generators_present']}**.",
        "",
        "| stem | generator | table? | json | md |",
        "| --- | --- | --- | --- | --- |",
    ]
    for e in d["evidence"]:  # type: ignore[index]
        gen = e["generator"] or "(missing)"
        lines.append(
            f"| {e['stem']} | `{gen}` | {e['renders_table']} "
            f"| {'y' if e['json_present'] else '-'} "
            f"| {'y' if e['md_present'] else '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(check: bool = False) -> int:
    d = measure()
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
            print("paper_evidence_index: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
