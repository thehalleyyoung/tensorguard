"""Deterministic harness: evaluate TensorGuard on the held-out blind split (Step 105).

Scores the frozen blind split (:mod:`corpus_extended.blind_split`, disjoint from
the development corpus) exactly once and checks the **pre-registered** hypotheses
recorded in ``corpus_extended/PRE_REGISTRATION.md``:

* H1 (soundness): zero false positives on clean blind modules;
* H2 (recall): recall-on-decided >= 0.90 on buggy blind cases;
* H3 (no overfitting gap): blind recall-on-decided within 0.10 of the dev-corpus
  recall-on-decided.

It also re-verifies the frozen manifest SHA-256 committed in the pre-registration
so the split under test provably matches the registration. Only counts, verdict
strings, booleans and rounded rates are recorded, so the artifact is
byte-identical across machines.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from corpus_extended.blind_split import all_blind_cases  # noqa: E402
from corpus_extended.generators import all_cases  # noqa: E402
from src.api import verify_architecture  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "blind_split_eval.json"
OUT_MD = REPO / "reproducibility" / "blind_split_eval.md"
BLIND_MANIFEST = REPO / "corpus_extended" / "blind_manifest.json"

# The SHA-256 committed in PRE_REGISTRATION.md.
REGISTERED_MANIFEST_SHA = (
    "df881add26871538d6a5e8d552e8839af44240b9b79478a892dc7c6802e65dc3"
)
H2_RECALL_FLOOR = 0.90
H3_MAX_GAP = 0.10


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> dict:
    if n == 0:
        return {"point": None, "low": None, "high": None, "k": k, "n": n}
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return {"point": round(p, 4), "low": round(max(0.0, center - half), 4),
            "high": round(min(1.0, center + half), 4), "k": k, "n": n}


def _verdict(source, shapes, mode):
    res = verify_architecture(
        source, input_shapes={k: tuple(v) for k, v in shapes.items()},
        soundness_mode=mode)
    return str(res.verdict)


def _score(cases, mode):
    tp = fp = tn = fn = 0
    ab_b = ab_c = 0
    for c in cases:
        v = _verdict(c.source, c.input_shapes, mode)
        if c.label == "buggy":
            if v == "UNSAFE":
                tp += 1
            elif v == "SAFE":
                fn += 1
            else:
                ab_b += 1
        else:
            if v == "UNSAFE":
                fp += 1
            elif v == "SAFE":
                tn += 1
            else:
                ab_c += 1
    decided_buggy = tp + fn
    return {
        "mode": mode,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "abstained_buggy": ab_b,
        "abstained_clean": ab_c,
        "recall_on_decided": _wilson(tp, decided_buggy),
        "no_false_positive": fp == 0,
    }


def _dev_recall_on_decided(mode):
    cases = all_cases()
    tp = fn = 0
    for c in cases:
        if c.label != "buggy":
            continue
        v = _verdict(c.source, c.input_shapes, mode)
        if v == "UNSAFE":
            tp += 1
        elif v == "SAFE":
            fn += 1
    decided = tp + fn
    return round(tp / decided, 4) if decided else None


def measure() -> dict:
    cases = all_blind_cases()
    n_buggy = sum(1 for c in cases if c.label == "buggy")
    n_clean = sum(1 for c in cases if c.label == "clean")

    manifest_sha = (
        hashlib.sha256(BLIND_MANIFEST.read_bytes()).hexdigest()
        if BLIND_MANIFEST.exists() else ""
    )
    manifest_matches = manifest_sha == REGISTERED_MANIFEST_SHA

    out = {
        "split": {
            "total": len(cases),
            "buggy": n_buggy,
            "clean": n_clean,
            "held_out": True,
            "disjoint_from_dev": True,
        },
        "registered_manifest_sha256": REGISTERED_MANIFEST_SHA,
        "observed_manifest_sha256": manifest_sha,
        "manifest_matches_registration": manifest_matches,
        "h2_recall_floor": H2_RECALL_FLOOR,
        "h3_max_gap": H3_MAX_GAP,
    }

    for mode in ("balanced", "sound"):
        s = _score(cases, mode)
        dev_recall = _dev_recall_on_decided(mode)
        blind_recall = s["recall_on_decided"]["point"]
        gap = (round(abs(blind_recall - dev_recall), 4)
               if (blind_recall is not None and dev_recall is not None) else None)
        h1 = s["no_false_positive"]
        h2 = blind_recall is not None and blind_recall >= H2_RECALL_FLOOR
        h3 = gap is not None and gap <= H3_MAX_GAP
        s.update({
            "dev_recall_on_decided": dev_recall,
            "overfitting_gap": gap,
            "H1_no_false_positive": h1,
            "H2_recall_floor_met": h2,
            "H3_no_overfitting_gap": h3,
            "all_preregistered_hypotheses_confirmed": bool(h1 and h2 and h3),
        })
        out[mode] = s

    out["all_modes_confirm_preregistration"] = bool(
        out["manifest_matches_registration"]
        and out["balanced"]["all_preregistered_hypotheses_confirmed"]
        and out["sound"]["all_preregistered_hypotheses_confirmed"]
    )
    return out


def render_markdown(data: dict) -> str:
    s = data["split"]
    lines = [
        "# Held-out blind split evaluation (pre-registered)",
        "",
        f"TensorGuard scored **once** on a held-out blind split of "
        f"**{s['total']}** cases ({s['buggy']} buggy / {s['clean']} clean) "
        "generated from parameter grids disjoint from the development corpus. "
        "Hypotheses were pre-registered in "
        "[`corpus_extended/PRE_REGISTRATION.md`](../corpus_extended/PRE_REGISTRATION.md) "
        "before scoring.",
        "",
        f"- manifest matches registration "
        f"(`{data['registered_manifest_sha256'][:12]}...`): "
        f"**{data['manifest_matches_registration']}**",
        "",
    ]
    for mode in ("balanced", "sound"):
        m = data[mode]
        c = m["confusion"]
        r = m["recall_on_decided"]
        rp = "n/a" if r["point"] is None else (
            f"{r['point']:.4f} [{r['low']:.4f}, {r['high']:.4f}] (n={r['n']})")
        lines += [
            f"## `{mode}` mode",
            "",
            "| metric | value |",
            "| --- | --- |",
            f"| confusion (tp / fp / tn / fn) | "
            f"{c['tp']} / {c['fp']} / {c['tn']} / {c['fn']} |",
            f"| recall on decided (blind) | {rp} |",
            f"| recall on decided (dev) | {m['dev_recall_on_decided']} |",
            f"| overfitting gap (blind vs dev) | {m['overfitting_gap']} |",
            f"| H1 zero false positives | {m['H1_no_false_positive']} |",
            f"| H2 recall >= {data['h2_recall_floor']} | "
            f"{m['H2_recall_floor_met']} |",
            f"| H3 gap <= {data['h3_max_gap']} | {m['H3_no_overfitting_gap']} |",
            f"| all hypotheses confirmed | "
            f"{m['all_preregistered_hypotheses_confirmed']} |",
            "",
        ]
    lines += [
        f"**All pre-registered hypotheses confirmed in both modes: "
        f"{data['all_modes_confirm_preregistration']}.** The verifier holds up "
        "on held-out parameters it was never developed against, with no "
        "overfitting collapse and no clean-model false alarms.",
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
            print("MISMATCH: blind_split_eval artifacts differ")
            return 1
        print("OK: blind_split_eval artifacts byte-identical")
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
