#!/usr/bin/env python3
"""
leaderboard.py — an open, reproducible TensorGuard benchmark leaderboard
(100_STEPS.md Step 95).

Goal: let the community drive recall up. This harness scores tools on the
frozen, content-addressed ground-truth corpus in ``real_benchmarks/`` (each
case is a real PyTorch ``nn.Module`` labelled clean/buggy and pinned by
sha256, so the benchmark cannot silently drift). It:

  * runs the reference tool — TensorGuard itself — live over every corpus case
    and computes a standard detection scorecard (confusion matrix, recall,
    precision, F1, accuracy, abstention rate);
  * merges any community-submitted entries dropped in
    ``benchmarks/leaderboard_entries/*.json`` (schema documented in
    ``docs/leaderboard/CONTRIBUTING.md``); each submitted entry is re-scored
    here from its raw per-case verdicts, so a submitter cannot inflate their own
    metrics — only the verdicts are trusted, the scoring is recomputed;
  * emits a deterministic JSON + a ranked Markdown leaderboard.

The artifact records metrics and verdicts only (no wall-clock), so it is
byte-deterministic and checked by ``reproduce_all.py --check``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from real_benchmarks.load import load_items, read_source, verify_item  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "leaderboard.json"
OUT_MD = REPO / "reproducibility" / "leaderboard.md"
ENTRIES_DIR = REPO / "benchmarks" / "leaderboard_entries"


def _corpus_fingerprint(items: List[dict]) -> str:
    """Content hash of the (id, label, sha256) tuples — pins the corpus
    version a scorecard was produced against."""
    h = hashlib.sha256()
    for it in sorted(items, key=lambda i: i["id"]):
        h.update(it["id"].encode())
        h.update(it["label"].encode())
        h.update(it["sha256"].encode())
    return h.hexdigest()


def _score(verdicts: Dict[str, str], labels: Dict[str, str]) -> Dict:
    """Recompute a scorecard from raw per-case verdicts.

    A buggy case is a true positive iff the verdict is UNSAFE. A verdict of
    UNKNOWN (abstention) counts as 'not detected' for recall but is tracked
    separately and never counted as a false positive on clean cases.
    """
    tp = fp = tn = fn = 0
    abstain = 0
    for cid, label in labels.items():
        v = verdicts.get(cid, "UNKNOWN")
        if v == "UNKNOWN":
            abstain += 1
        if label == "buggy":
            if v == "UNSAFE":
                tp += 1
            else:                       # SAFE or UNKNOWN -> missed
                fn += 1
        else:                           # clean
            if v == "UNSAFE":
                fp += 1
            else:                       # SAFE or UNKNOWN -> not a false alarm
                tn += 1
    n = tp + fp + tn + fn
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    accuracy = (tp + tn) / n if n else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "abstain": abstain,
        "recall": round(recall, 6),
        "precision": round(precision, 6),
        "f1": round(f1, 6),
        "accuracy": round(accuracy, 6),
        "n": n,
    }


def _tensorguard_verdicts(items: List[dict]) -> Dict[str, str]:
    verdicts: Dict[str, str] = {}
    for it in items:
        res = verify_item(it)
        # three-valued: prefer the soundness-aware verdict when present.
        v = getattr(res, "verdict", None)
        if v not in ("SAFE", "UNSAFE", "UNKNOWN"):
            v = "UNSAFE" if res.bug_count > 0 else "SAFE"
        verdicts[it["id"]] = v
    return verdicts


def _load_community_entries(labels: Dict[str, str]) -> List[Dict]:
    entries: List[Dict] = []
    if not ENTRIES_DIR.exists():
        return entries
    for path in sorted(ENTRIES_DIR.glob("*.json")):
        raw = json.loads(path.read_text())
        name = raw.get("tool", path.stem)
        verdicts = {str(k): str(v).upper() for k, v in
                    raw.get("verdicts", {}).items()}
        entries.append({
            "name": name,
            "description": raw.get("description", ""),
            "url": raw.get("url", ""),
            "submitted_verdicts": verdicts,
            "scorecard": _score(verdicts, labels),
            "source": "community",
        })
    return entries


def measure() -> Dict:
    items = load_items(verify=True)
    labels = {it["id"]: it["label"] for it in items}
    fingerprint = _corpus_fingerprint(items)

    tg_verdicts = _tensorguard_verdicts(items)
    entries: List[Dict] = [{
        "name": "TensorGuard",
        "description": "Sound static shape/device/gradient verifier "
                       "(refinement types + Z3 + reduced-product domain).",
        "url": "https://github.com/thehalleyyoung/tensorguard",
        "submitted_verdicts": tg_verdicts,
        "scorecard": _score(tg_verdicts, labels),
        "source": "reference",
    }]
    entries.extend(_load_community_entries(labels))

    # Rank: F1 desc, then recall desc, then precision desc, then name asc.
    def _key(e):
        s = e["scorecard"]
        return (-s["f1"], -s["recall"], -s["precision"], e["name"])
    entries.sort(key=_key)
    for rank, e in enumerate(entries, start=1):
        e["rank"] = rank

    return {
        "corpus": {
            "name": "tensorguard-real-benchmarks",
            "total": len(items),
            "clean": sum(1 for v in labels.values() if v == "clean"),
            "buggy": sum(1 for v in labels.values() if v == "buggy"),
            "fingerprint_sha256": fingerprint,
        },
        "n_entries": len(entries),
        "entries": entries,
    }


def render_markdown(data: Dict) -> str:
    c = data["corpus"]
    L: List[str] = []
    L.append("# TensorGuard open benchmark leaderboard")
    L.append("")
    L.append("> Generated by `reproducibility/leaderboard.py`. Metrics and "
             "verdicts only, no timing — byte-deterministic, checked by "
             "`reproduce_all.py --check`.")
    L.append("")
    L.append(f"Corpus **{c['name']}**: {c['total']} content-addressed cases "
             f"({c['clean']} clean, {c['buggy']} buggy). Corpus fingerprint "
             f"`{c['fingerprint_sha256'][:16]}…`. Scores are recomputed here "
             f"from raw per-case verdicts, so a submission cannot self-report "
             f"inflated metrics.")
    L.append("")
    L.append("| Rank | Tool | Source | Recall | Precision | F1 | Accuracy | "
             "TP | FP | FN | Abstain |")
    L.append("|------|------|--------|--------|-----------|----|----------|"
             "----|----|----|---------|")
    for e in data["entries"]:
        s = e["scorecard"]
        L.append(f"| {e['rank']} | {e['name']} | {e['source']} | "
                 f"{s['recall']:.3f} | {s['precision']:.3f} | {s['f1']:.3f} | "
                 f"{s['accuracy']:.3f} | {s['tp']} | {s['fp']} | {s['fn']} | "
                 f"{s['abstain']} |")
    L.append("")
    L.append("## How to climb the leaderboard")
    L.append("")
    L.append("Two ways to contribute, both documented in "
             "[`docs/leaderboard/CONTRIBUTING.md`](../docs/leaderboard/CONTRIBUTING.md):")
    L.append("")
    L.append("1. **Add a benchmark case** — a real `nn.Module` (clean or "
             "buggy) the field should be able to verify. New cases are "
             "content-addressed and frozen into `real_benchmarks/manifest.json` "
             "so every tool is scored on the same fixed corpus.")
    L.append("2. **Submit a tool entry** — drop a results JSON in "
             "`benchmarks/leaderboard_entries/` giving your tool's verdict "
             "(`SAFE`/`UNSAFE`/`UNKNOWN`) for each case id. The harness "
             "re-scores it here; only the raw verdicts are trusted.")
    L.append("")
    L.append("Precision must stay at 1.000 to be taken seriously: a single "
             "false alarm on a clean module is disqualifying for a tool meant "
             "to ship inside a framework. The open challenge is to drive "
             "**recall** up on ever-harder real-world bugs without sacrificing "
             "that precision.")
    L.append("")
    return "\n".join(L)


def run(check: bool = False) -> int:
    data = measure()
    new_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
    new_md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != new_json:
            print("MISMATCH: leaderboard.json differs", file=sys.stderr)
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != new_md:
            print("MISMATCH: leaderboard.md differs", file=sys.stderr)
            ok = False
        print("leaderboard --check:", "OK" if ok else "FAILED")
        return 0 if ok else 1
    OUT_JSON.write_text(new_json)
    OUT_MD.write_text(new_md)
    ref = next(e for e in data["entries"] if e["source"] == "reference")
    s = ref["scorecard"]
    print(f"Wrote {OUT_JSON.relative_to(REPO)} and {OUT_MD.relative_to(REPO)}: "
          f"{data['n_entries']} entries; TensorGuard recall={s['recall']:.3f} "
          f"precision={s['precision']:.3f} f1={s['f1']:.3f}.")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
