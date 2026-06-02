"""Step 45 -- AST-frontend parse-success SLA over a real-world model corpus.

A static verifier only matters if its *frontend* can ingest the architectures
people actually write.  Step 36 (`fx_trace_success.py`) measures the `torch.fx`
frontend; this harness measures the complementary **source/AST frontend**
(`extract_computation_graph`) that powers `verify_model` directly from Python
source -- the path used when the model object is not instantiable (missing
weights/configs) or when verifying from a file.

For every model in a curated, self-contained corpus
(`evaluation/frontend_corpus.py`) the harness records, never raising:

  * **parsed**   -- `extract_computation_graph(source)` returned a graph with a
    non-empty step list without crashing;
  * **steps**    -- number of extracted computation steps;
  * **isolated** -- statements the frontend had to isolate (Step 43);
  * **unsupported** -- operators soundly abstracted as `UNSUPPORTED` (Step 34).

The corpus is embedded (not live-cloned) and the AST frontend never imports
torch, so the published artifact is byte-reproducible across machines and torch
versions.  `--gate` fails the build if the parse-success rate regresses below
the published floor; `--check` enforces a byte-identical committed artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

JSON_PATH = os.path.join(HERE, "frontend_parse_sla.json")
MD_PATH = os.path.join(HERE, "frontend_parse_sla.md")

# The curated corpus is fully supported by the AST frontend, so the published
# floor is total parse success.  Any regression (a newly-introduced crash or an
# empty graph) trips the gate.
PARSE_FLOOR = 1.0


def _count_op(graph, op_name: str) -> int:
    from src.model_checker import OpKind

    target = getattr(OpKind, op_name, None)
    if target is None:
        return 0
    return sum(1 for s in graph.steps if getattr(s, "op", None) == target)


def _eval_one(name: str, source: str) -> Dict[str, object]:
    """Parse a single source through the AST frontend; never raises."""
    from src.model_checker import extract_computation_graph

    rec: Dict[str, object] = {
        "model": name, "parsed": False, "steps": 0,
        "isolated": 0, "unsupported": 0, "error": None,
    }
    try:
        graph = extract_computation_graph(source)
    except Exception as exc:
        rec["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
        return rec
    n_steps = len(graph.steps)
    rec["steps"] = n_steps
    rec["isolated"] = len(getattr(graph, "isolated_regions", []) or [])
    rec["unsupported"] = _count_op(graph, "UNSUPPORTED")
    # "Parsed" requires a non-empty graph: an empty step list means the frontend
    # silently produced nothing useful, which we count as a failure.
    rec["parsed"] = n_steps > 0
    return rec


def evaluate_corpus() -> List[Dict[str, object]]:
    from evaluation.frontend_corpus import CORPUS

    return [_eval_one(name, src) for name, src in sorted(CORPUS.items())]


def _summarise(records: List[Dict[str, object]]) -> Dict[str, object]:
    n = len(records)
    parsed = sum(1 for r in records if r["parsed"])
    total_steps = sum(r["steps"] for r in records)
    total_iso = sum(r["isolated"] for r in records)
    total_unsup = sum(r["unsupported"] for r in records)
    return {
        "n_models": n,
        "parsed": parsed,
        "parse_success_rate": round(parsed / n, 4) if n else 0.0,
        "total_steps": total_steps,
        "total_isolated": total_iso,
        "total_unsupported": total_unsup,
        "precise_step_fraction": round(
            (total_steps - total_unsup) / total_steps, 4)
        if total_steps else 0.0,
    }


def build_report() -> Dict[str, object]:
    records = evaluate_corpus()
    return {
        "meta": {
            "generated_by": "evaluation/frontend_parse_sla.py",
            "command": "PYTHONPATH=. python3 evaluation/frontend_parse_sla.py",
            "python_version": "%d.%d" % sys.version_info[:2],
            "corpus_size": len(records),
            "note": ("AST source frontend is torch-version-independent; this "
                     "artifact is byte-reproducible across environments."),
        },
        "summary": _summarise(records),
        "parse_floor": PARSE_FLOOR,
        "models": records,
    }


def _dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def render_markdown(rep: Dict[str, object]) -> str:
    summ = rep["summary"]
    lines = [
        "# AST-frontend parse-success SLA (real-world model corpus)",
        "",
        ("Static ingestion of %d self-contained, real-world-style PyTorch model "
         "sources through TensorGuard's source/AST frontend "
         "(`extract_computation_graph`)." % summ["n_models"]),
        "",
        ("Parse success: **%d of %d** models lowered to a non-empty computation "
         "graph without crashing (rate %.3f). Across the corpus, %d of %d "
         "extracted steps are operators reasoned about precisely; %d are soundly "
         "abstracted as unsupported (Step 34) and %d statements were isolated "
         "(Step 43)." % (
             summ["parsed"], summ["n_models"], summ["parse_success_rate"],
             summ["total_steps"] - summ["total_unsupported"],
             summ["total_steps"], summ["total_unsupported"],
             summ["total_isolated"])),
        "",
        "| Model | Parsed | Steps | Isolated | Unsupported |",
        "|-------|--------|-------|----------|-------------|",
    ]
    for r in rep["models"]:
        lines.append("| `%s` | %s | %d | %d | %d |" % (
            r["model"], "yes" if r["parsed"] else "NO",
            r["steps"], r["isolated"], r["unsupported"]))
    lines.append("")
    failures = [r for r in rep["models"] if not r["parsed"]]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for r in failures:
            lines.append("* `%s`: %s" % (r["model"], r["error"]))
        lines.append("")
    return "\n".join(lines)


def gate() -> int:
    rep = build_report()
    summ = rep["summary"]
    rate = summ["parse_success_rate"]
    if rate < PARSE_FLOOR:
        print("FRONTEND PARSE-SLA GATE FAILED: rate %.4f < floor %.4f"
              % (rate, PARSE_FLOOR))
        for r in rep["models"]:
            if not r["parsed"]:
                print("  - %s: %s" % (r["model"], r["error"]))
        return 1
    print("frontend parse-sla gate PASS: %d of %d models parsed (rate %.3f, "
          "floor %.3f)" % (summ["parsed"], summ["n_models"], rate, PARSE_FLOOR))
    return 0


def run(check: bool = False, write: bool = True) -> int:
    rep = build_report()
    text = _dumps(rep)

    if check:
        if not os.path.exists(JSON_PATH):
            print("frontend_parse_sla.json missing; run the harness first")
            return 1
        if open(JSON_PATH).read() != text:
            print("frontend_parse_sla.json is stale; run "
                  "`make frontend-parse-sla`")
            return 1
        md = render_markdown(rep)
        if not os.path.exists(MD_PATH) or open(MD_PATH).read() != md:
            print("frontend_parse_sla.md is stale; run "
                  "`make frontend-parse-sla`")
            return 1
        print("frontend parse-sla report up to date")
        return 0

    if write:
        with open(JSON_PATH, "w") as fh:
            fh.write(text)
        with open(MD_PATH, "w") as fh:
            fh.write(render_markdown(rep))
    s = rep["summary"]
    print("frontend parse-sla: %d of %d models parsed (rate %.3f); %d steps, "
          "%d isolated, %d unsupported" % (
              s["parsed"], s["n_models"], s["parse_success_rate"],
              s["total_steps"], s["total_isolated"], s["total_unsupported"]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Verify the committed report is byte-identical.")
    ap.add_argument("--gate", action="store_true",
                    help="Fail if parse-success rate regresses below the floor.")
    args = ap.parse_args()
    if args.gate:
        return gate()
    return run(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
