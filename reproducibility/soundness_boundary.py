#!/usr/bin/env python3
"""
soundness_boundary.py — empirically validate TensorGuard's unsound/incomplete
boundary against the LIVE verifier (100_STEPS.md Step 94).

The soundness contract (``src/soundness_contract.py`` / ``SOUNDNESS_CONTRACT.md``)
states *what* TensorGuard guarantees and where it abstains or may miss a bug.
A contract is only trustworthy if its claims hold against real code. This
harness runs ``verify_architecture`` on a curated set of boundary probes — one
per region of the contract — under all three soundness modes and asserts the
observed verdict matches the documented behaviour:

  * Refutation soundness (no false alarm): an in-fragment shape bug is REFUTED
    (UNSAFE) in every mode.
  * Verification soundness (in-fragment, modeled scope): an in-fragment clean
    module is SAFE in every mode.
  * The verifiable-fragment boundary is *mode-dependent* (KNOWN_UNSOUNDNESS U1):
    an out-of-fragment construct (data-dependent control flow, tensor->scalar
    coercion) ABSTAINS (UNKNOWN) in ``sound`` mode but is reported SAFE in the
    permissive ``balanced``/``heuristic`` modes — the recall trade-off, surfaced
    rather than hidden.

The emitted artifact records verdicts only (no wall-clock), so it is
byte-deterministic and checked by ``reproduce_all.py --check``.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.api import verify_architecture  # noqa: E402
from src import soundness_contract as sc  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "soundness_boundary.json"
OUT_MD = REPO / "reproducibility" / "soundness_boundary.md"

MODES = ("sound", "balanced", "heuristic")

PRE = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"


@dataclass(frozen=True)
class Probe:
    name: str
    region: str          # which contract region this exercises
    source: str
    input_shapes: Dict[str, tuple]
    # expected verdict per mode
    expected: Dict[str, str]
    note: str


def _probes() -> List[Probe]:
    clean = PRE + (
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.a = nn.Linear(4, 8)\n"
        "        self.b = nn.Linear(8, 2)\n"
        "    def forward(self, x):\n"
        "        return self.b(self.a(x))\n"
    )
    shape_bug = PRE + (
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.a = nn.Linear(4, 8)\n"
        "        self.b = nn.Linear(16, 2)\n"   # expects 16, gets 8
        "    def forward(self, x):\n"
        "        return self.b(self.a(x))\n"
    )
    ddcf = PRE + (
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.a = nn.Linear(4, 8)\n"
        "        self.b = nn.Linear(8, 2)\n"
        "    def forward(self, x):\n"
        "        x = self.a(x)\n"
        "        if x.sum() > 0:\n"            # data-dependent control flow
        "            x = x * 2\n"
        "        return self.b(x)\n"
    )
    to_scalar = PRE + (
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.a = nn.Linear(4, 8)\n"
        "        self.b = nn.Linear(8, 2)\n"
        "    def forward(self, x):\n"
        "        x = self.a(x)\n"
        "        k = int(x.flatten()[0].item())\n"   # tensor -> python scalar
        "        return self.b(x) + k\n"
    )
    return [
        Probe(
            name="in_fragment_clean",
            region="verification soundness (in-fragment, modeled scope)",
            source=clean, input_shapes={"x": (1, 4)},
            expected={"sound": "SAFE", "balanced": "SAFE", "heuristic": "SAFE"},
            note="Clean MLP fully inside the verifiable fragment is SAFE in "
                 "every mode.",
        ),
        Probe(
            name="in_fragment_shape_bug",
            region="refutation soundness (no false alarm)",
            source=shape_bug, input_shapes={"x": (1, 4)},
            expected={"sound": "UNSAFE", "balanced": "UNSAFE",
                      "heuristic": "UNSAFE"},
            note="Real in-fragment Linear inner-dim mismatch is REFUTED in "
                 "every mode (Z3-discharged).",
        ),
        Probe(
            name="out_of_fragment_data_dependent_control_flow",
            region="fragment boundary / KNOWN_UNSOUNDNESS U1 (mode-dependent)",
            source=ddcf, input_shapes={"x": (1, 4)},
            expected={"sound": "UNKNOWN", "balanced": "SAFE",
                      "heuristic": "SAFE"},
            note="Data-dependent branch leaves the fragment: sound mode "
                 "abstains (UNKNOWN); balanced/heuristic report SAFE (the U1 "
                 "recall trade-off).",
        ),
        Probe(
            name="out_of_fragment_tensor_to_scalar",
            region="fragment boundary / KNOWN_UNSOUNDNESS U1 (mode-dependent)",
            source=to_scalar, input_shapes={"x": (1, 4)},
            expected={"sound": "UNKNOWN", "balanced": "SAFE",
                      "heuristic": "SAFE"},
            note="`.item()` coerces a tensor to a Python scalar (out of "
                 "fragment): sound mode abstains; permissive modes report SAFE.",
        ),
    ]


@dataclass
class ProbeResult:
    name: str
    region: str
    note: str
    observed: Dict[str, str] = field(default_factory=dict)
    expected: Dict[str, str] = field(default_factory=dict)
    match: bool = True


def measure() -> Dict:
    results: List[ProbeResult] = []
    all_match = True
    for p in _probes():
        pr = ProbeResult(name=p.name, region=p.region, note=p.note,
                         expected=dict(p.expected))
        for mode in MODES:
            r = verify_architecture(p.source, input_shapes=p.input_shapes,
                                    filename=f"<{p.name}>", soundness_mode=mode)
            pr.observed[mode] = r.verdict
        pr.match = pr.observed == pr.expected
        all_match = all_match and pr.match
        results.append(pr)

    contract = {
        "domain_clauses": len(sc.DOMAIN_CLAUSES),
        "under_approximated_bug_classes": len(sc.UNDER_APPROXIMATED_BUG_CLASSES),
        "out_of_fragment_clauses": len(sc.OUT_OF_FRAGMENT_CLAUSES),
        "known_unsoundness_gaps": [
            {"id": g.id, "affected_direction": g.affected_direction,
             "location": g.location}
            for g in sc.KNOWN_UNSOUNDNESS
        ],
    }
    return {
        "modes": list(MODES),
        "n_probes": len(results),
        "all_match": all_match,
        "probes": [
            {"name": r.name, "region": r.region, "note": r.note,
             "expected": r.expected, "observed": r.observed, "match": r.match}
            for r in results
        ],
        "contract": contract,
    }


def render_markdown(data: Dict) -> str:
    L: List[str] = []
    L.append("# Soundness / incompleteness boundary — validated against real code")
    L.append("")
    L.append("> Generated by `reproducibility/soundness_boundary.py`. Verdicts "
             "only, no timing — byte-deterministic, checked by "
             "`reproduce_all.py --check`.")
    L.append("")
    L.append(f"Probes: **{data['n_probes']}** · modes: "
             f"{', '.join('`'+m+'`' for m in data['modes'])} · "
             f"all observed verdicts match the documented contract: "
             f"**{str(data['all_match']).upper()}**")
    L.append("")
    L.append("| Probe | Contract region | `sound` | `balanced` | `heuristic` | "
             "matches contract |")
    L.append("|-------|-----------------|---------|------------|-------------|"
             "------------------|")
    for p in data["probes"]:
        o = p["observed"]
        L.append(f"| `{p['name']}` | {p['region']} | {o['sound']} | "
                 f"{o['balanced']} | {o['heuristic']} | "
                 f"{'yes' if p['match'] else 'NO'} |")
    L.append("")
    L.append("## What each probe proves")
    L.append("")
    for p in data["probes"]:
        L.append(f"- **`{p['name']}`** — {p['note']}")
    L.append("")
    c = data["contract"]
    L.append("## Contract coverage (from `src/soundness_contract.py`)")
    L.append("")
    L.append(f"- Domain clauses: **{c['domain_clauses']}**")
    L.append(f"- Under-approximated bug classes (may miss bugs, out of the "
             f"never-miss-pass guarantee): **{c['under_approximated_bug_classes']}**")
    L.append(f"- Out-of-fragment construct classes (detected by "
             f"`check_traceability`): **{c['out_of_fragment_clauses']}**")
    L.append(f"- Known unsoundness gaps surfaced (not hidden): "
             f"**{len(c['known_unsoundness_gaps'])}** "
             f"({', '.join(g['id'] for g in c['known_unsoundness_gaps'])})")
    L.append("")
    L.append("The full clause-by-clause contract is `SOUNDNESS_CONTRACT.md` "
             "(generated from the same module). This harness shows the "
             "*empirical* half: the documented behaviour reproduces on the live "
             "verifier.")
    L.append("")
    return "\n".join(L)


def run(check: bool = False) -> int:
    data = measure()
    new_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
    new_md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != new_json:
            print("MISMATCH: soundness_boundary.json differs", file=sys.stderr)
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != new_md:
            print("MISMATCH: soundness_boundary.md differs", file=sys.stderr)
            ok = False
        if not data["all_match"]:
            print("FAIL: observed verdicts diverge from documented contract",
                  file=sys.stderr)
            ok = False
        print("soundness_boundary --check:", "OK" if ok else "FAILED")
        return 0 if ok else 1
    OUT_JSON.write_text(new_json)
    OUT_MD.write_text(new_md)
    if not data["all_match"]:
        print("WARNING: observed verdicts diverge from contract!",
              file=sys.stderr)
        return 1
    print(f"Wrote {OUT_JSON.relative_to(REPO)} and "
          f"{OUT_MD.relative_to(REPO)} ({data['n_probes']} probes, "
          f"all_match={data['all_match']}).")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
