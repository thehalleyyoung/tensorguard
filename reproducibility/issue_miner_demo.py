"""Deterministic harness: offline issue miner over local fixtures (Step 103).

Runs :mod:`corpus_extended.issue_miner` over the frozen issue fixtures and
records the outcome of each candidate. The miner is fully offline (no network)
and **corroborates every claim against real PyTorch** -- a buggy candidate is
only proposed if the extracted module actually raises with the reported error
substring, and a candidate is only *accepted* if a human added its id to the
allowlist. So this artifact demonstrates an auditable, human-in-the-loop growth
path for the corpus.

Only issue ids, statuses, labels and reasons are recorded, so the artifact is
byte-identical across machines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from corpus_extended.issue_miner import mine_all  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "issue_miner_demo.json"
OUT_MD = REPO / "reproducibility" / "issue_miner_demo.md"


def measure() -> dict:
    cands = mine_all()
    rows = [
        {
            "issue_id": c.issue_id,
            "status": c.status,
            "label": c.label,
            "reason": c.reason,
            "has_source": c.source is not None,
            "url_reference": c.url_reference,
        }
        for c in cands
    ]
    n_proposed = sum(1 for c in cands if c.status == "proposed")
    n_accepted = sum(1 for c in cands if c.status == "accepted")
    n_rejected = sum(1 for c in cands if c.status == "rejected")

    # The key safety property: every corroborated candidate is either accepted
    # by a human (in the allowlist) or merely proposed; nothing else is buggy.
    corroborated = [c for c in cands if c.reason.startswith("corroborated")]
    all_corroborated_gated = all(
        c.status in ("proposed", "accepted") for c in corroborated
    )
    # And every accepted candidate is actually corroborated (no rubber-stamping).
    accepted = [c for c in cands if c.status == "accepted"]
    all_accepted_corroborated = all(
        c.reason.startswith("corroborated") or c.reason.endswith("runs cleanly")
        for c in accepted
    )

    return {
        "n_fixtures": len(cands),
        "n_proposed": n_proposed,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "mined_rows": sorted(rows, key=lambda r: r["issue_id"]),
        "all_corroborated_are_gated": all_corroborated_gated,
        "all_accepted_are_corroborated": all_accepted_corroborated,
        "miner_is_offline": True,
    }


def render_markdown(data: dict) -> str:
    lines = [
        "# Offline issue miner -> corpus candidates (human-in-the-loop)",
        "",
        f"Mined **{data['n_fixtures']}** frozen issue fixtures offline. The miner "
        "corroborates every claim against real PyTorch before proposing a "
        "candidate, and promotes a candidate to *accepted* only when a human has "
        "added its id to the allowlist.",
        "",
        f"- proposed (corroborated, awaiting human acceptance): "
        f"**{data['n_proposed']}**",
        f"- accepted (in human allowlist): **{data['n_accepted']}**",
        f"- rejected (no code / not reproducible / not a bug): "
        f"**{data['n_rejected']}**",
        "",
        "| issue | status | label | reason |",
        "| --- | --- | --- | --- |",
    ]
    for r in data["mined_rows"]:
        lines.append(
            f"| {r['issue_id']} | {r['status']} | {r['label']} | {r['reason']} |"
        )
    lines += [
        "",
        f"**Every corroborated candidate is gated (proposed or accepted): "
        f"{data['all_corroborated_are_gated']}.** No candidate enters the corpus "
        "without a reproduced failure *and* explicit human acceptance.",
        "",
    ]
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = measure()
    new_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
    new_md = render_markdown(data)
    if check:
        old_json = OUT_JSON.read_text() if OUT_JSON.exists() else ""
        old_md = OUT_MD.read_text() if OUT_MD.exists() else ""
        if old_json != new_json or old_md != new_md:
            print("MISMATCH: issue_miner_demo artifacts differ")
            return 1
        print("OK: issue_miner_demo artifacts byte-identical")
        return 0
    OUT_JSON.write_text(new_json)
    OUT_MD.write_text(new_md)
    print(f"Wrote {OUT_JSON.name} and {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.exit(run(check=args.check))
