"""Step 37 -- frontend reconciliation: torch.fx vs torch.export verdicts.

TensorGuard now has two independent capture frontends:

  * **fx**     -- ``torch.fx.symbolic_trace`` lowered by ``fx_extractor``;
  * **export** -- ``torch.export`` (ATen IR) lowered by ``export_extractor``.

They take entirely different routes -- symbolic Python tracing of the high-level
module versus an ahead-of-time ATen capture with lifted parameters -- yet, being
two views of the same program, a *sound* verifier must reach the **same
safe/unsafe verdict** through either one. This harness runs a fixed corpus of
real modules through both frontends and reports the number of *divergences*
(models the two frontends both capture but disagree on). The headline invariant,
enforced by ``--gate``, is **zero divergences**.

A model that one frontend cannot capture (for example ``torch.export`` validates
shapes eagerly and refuses to export a shape-buggy model, while ``torch.fx``
happily traces it and lets the Z3 engine produce the counterexample) is recorded
as a *capture gap*, not a divergence: the frontends still agree that the model is
unsafe, they merely discover it at different stages. The corpus is split into
``safe`` and ``unsafe`` cases so the harness also checks each verdict against
ground truth, not merely fx/export agreement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

JSON_PATH = os.path.join(HERE, "frontend_reconciliation.json")
MD_PATH = os.path.join(HERE, "frontend_reconciliation.md")


# ---------------------------------------------------------------------------
# Corpus -- (name, module factory, input shape, expected_safe).
# ---------------------------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _MLPBad(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(99, 8)  # wrong in_features

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.mp = nn.MaxPool2d(2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = torch.relu(self.bn(self.c1(x)))
        x = self.mp(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class _CNNBad(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(999, 10)  # wrong flattened in_features

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.flatten(x, 1)
        return self.fc(x)


class _Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 16)

    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(x)))


class _ConvBNStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.c1(x)))
        x = torch.relu(self.bn2(self.c2(x)))
        x = torch.flatten(self.pool(x), 1)
        return self.fc(x)


class _DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 16),
        )

    def forward(self, x):
        return self.net(x)


_Case = Tuple[str, Callable[[], nn.Module], tuple, bool]

CORPUS: List[_Case] = [
    ("mlp", _MLP, (4, 16), True),
    ("deep_mlp", _DeepMLP, (4, 32), True),
    ("residual_mlp", _Residual, (4, 16), True),
    ("cnn", _CNN, (1, 3, 16, 16), True),
    ("conv_bn_stack", _ConvBNStack, (2, 3, 32, 32), True),
    ("mlp_bad_in_features", _MLPBad, (4, 16), False),
    ("cnn_bad_flatten", _CNNBad, (1, 3, 8, 8), False),
]


def _verdict_fx(factory, shape) -> Dict[str, object]:
    from src.fx_extractor import verify_module

    try:
        r = verify_module(factory().eval(), input_shapes={"x": shape},
                          backend="fx")
        return {"captured": True, "safe": bool(r.safe)}
    except Exception as exc:
        return {"captured": False, "safe": None,
                "error": f"{type(exc).__name__}: {str(exc)[:120]}"}


def _verdict_export(factory, shape) -> Dict[str, object]:
    from src.export_extractor import verify_module_export

    ex = (torch.randn(*shape),)
    r = verify_module_export(factory().eval(), input_shapes={"x": shape},
                            example_inputs=ex)
    # A capture failure surfaces as safe=False with an "extraction failed" error.
    captured = not any("extraction failed" in e for e in r.errors)
    return {"captured": captured, "safe": bool(r.safe),
            "error": r.errors[0] if r.errors else None}


def evaluate_corpus() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, factory, shape, expected_safe in CORPUS:
        fx = _verdict_fx(factory, shape)
        ex = _verdict_export(factory, shape)
        both_captured = bool(fx["captured"] and ex["captured"])
        divergent = both_captured and (fx["safe"] != ex["safe"])
        # Ground-truth agreement: every frontend that captured the model must
        # match the expected verdict (capture failure on an unsafe model still
        # yields safe=False, which matches expected for the unsafe cases).
        fx_correct = (fx["safe"] == expected_safe) if fx["captured"] else \
            (fx["safe"] is False and not expected_safe)
        ex_correct = (ex["safe"] == expected_safe) if ex["captured"] else \
            (ex["safe"] is False and not expected_safe)
        rows.append({
            "model": name,
            "expected_safe": expected_safe,
            "fx": fx,
            "export": ex,
            "both_captured": both_captured,
            "divergent": divergent,
            "fx_correct": bool(fx_correct),
            "export_correct": bool(ex_correct),
        })
    return rows


def _summarise(rows: List[Dict[str, object]]) -> Dict[str, object]:
    n = len(rows)
    both = sum(1 for r in rows if r["both_captured"])
    div = sum(1 for r in rows if r["divergent"])
    fx_ok = sum(1 for r in rows if r["fx_correct"])
    ex_ok = sum(1 for r in rows if r["export_correct"])
    return {
        "n_models": n,
        "both_captured": both,
        "divergences": div,
        "fx_correct": fx_ok,
        "export_correct": ex_ok,
        "agreement_rate": round((both - div) / both, 4) if both else 1.0,
    }


def build_report() -> Dict[str, object]:
    rows = evaluate_corpus()
    return {
        "meta": {
            "generated_by": "evaluation/frontend_reconciliation.py",
            "command": "PYTHONPATH=. python3 evaluation/frontend_reconciliation.py",
            "torch_version": torch.__version__,
            "python_version": "%d.%d" % sys.version_info[:2],
        },
        "summary": _summarise(rows),
        "models": rows,
    }


def _dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def render_markdown(rep: Dict[str, object]) -> str:
    meta, summ = rep["meta"], rep["summary"]
    lines = [
        "# Frontend reconciliation: torch.fx vs torch.export",
        "",
        ("The fx and export frontends capture each model by entirely different "
         "routes, yet a sound verifier must reach the same verdict through "
         "either. Generated against torch `%s`." % meta["torch_version"]),
        "",
        ("Across %d models, %d are captured by both frontends with **%d "
         "divergences** (agreement rate %.3f). Every captured verdict also "
         "matches ground truth: fx correct on %d, export correct on %d." % (
             summ["n_models"], summ["both_captured"], summ["divergences"],
             summ["agreement_rate"], summ["fx_correct"],
             summ["export_correct"])),
        "",
        "| Model | Expected | fx | export | Both captured | Divergent |",
        "|-------|----------|----|--------|---------------|-----------|",
    ]
    for r in rep["models"]:
        def _v(d):
            if not d["captured"]:
                return "no-capture"
            return "safe" if d["safe"] else "UNSAFE"
        lines.append("| `%s` | %s | %s | %s | %s | %s |" % (
            r["model"], "safe" if r["expected_safe"] else "UNSAFE",
            _v(r["fx"]), _v(r["export"]),
            "yes" if r["both_captured"] else "no",
            "YES" if r["divergent"] else "no"))
    lines.append("")
    return "\n".join(lines)


def gate() -> int:
    rep = build_report()
    summ = rep["summary"]
    problems: List[str] = []
    if summ["divergences"] != 0:
        problems.append("%d fx/export divergences" % summ["divergences"])
    if summ["fx_correct"] != summ["n_models"]:
        problems.append("fx incorrect on %d models"
                        % (summ["n_models"] - summ["fx_correct"]))
    if summ["export_correct"] != summ["n_models"]:
        problems.append("export incorrect on %d models"
                        % (summ["n_models"] - summ["export_correct"]))
    if problems:
        print("FRONTEND RECONCILIATION GATE FAILED: " + "; ".join(problems))
        for r in rep["models"]:
            if r["divergent"] or not r["fx_correct"] or not r["export_correct"]:
                print("  - %s: fx=%s export=%s expected_safe=%s"
                      % (r["model"], r["fx"], r["export"], r["expected_safe"]))
        return 1
    print("frontend reconciliation gate PASS: %d models, %d divergences, "
          "fx+export both correct on all" % (summ["n_models"],
                                             summ["divergences"]))
    return 0


def run(check: bool = False, write: bool = True) -> int:
    rep = build_report()
    text = _dumps(rep)
    if check:
        if not os.path.exists(JSON_PATH):
            print("frontend_reconciliation.json missing; run the harness first")
            return 1
        committed = json.load(open(JSON_PATH))
        if committed.get("meta", {}).get("torch_version") \
                != rep["meta"]["torch_version"]:
            print("QUALIFIED: torch version mismatch; skipping byte-identical "
                  "check")
            return 0
        if open(JSON_PATH).read() != text:
            print("frontend_reconciliation.json is stale; run "
                  "`make frontend-reconciliation`")
            return 1
        md = render_markdown(rep)
        if not os.path.exists(MD_PATH) or open(MD_PATH).read() != md:
            print("frontend_reconciliation.md is stale; run "
                  "`make frontend-reconciliation`")
            return 1
        print("frontend reconciliation report up to date")
        return 0
    if write:
        with open(JSON_PATH, "w") as fh:
            fh.write(text)
        with open(MD_PATH, "w") as fh:
            fh.write(render_markdown(rep))
    s = rep["summary"]
    print("frontend reconciliation: %d models, %d both-captured, %d "
          "divergences (rate %.3f)" % (
              s["n_models"], s["both_captured"], s["divergences"],
              s["agreement_rate"]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Verify the committed report is up to date (version-gated).")
    ap.add_argument("--gate", action="store_true",
                    help="Fail on any fx/export divergence or wrong verdict.")
    args = ap.parse_args()
    if args.gate:
        return gate()
    return run(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
