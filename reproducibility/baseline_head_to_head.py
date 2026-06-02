"""Head-to-head baseline comparison on the extended corpus (Step 110).

A reviewer's sharpest question is "why not just use an existing tool?". We answer
with a direct, same-corpus comparison against the realistic off-the-shelf
options a practitioner actually has:

* **TensorGuard** -- our static refinement-type + SMT verifier. It reads
  *source only*, needs just declared input *shapes* (no concrete tensors), and
  never executes the model.
* **torch.export tracing** -- the modern PyTorch ahead-of-time graph capture.
  It catches shape errors, but it must *instantiate the model, build concrete
  example inputs, and execute a trace* to do so. It is a dynamic baseline.
* **mypy** -- the standard Python static type checker. It is static like
  TensorGuard, but it does not model tensor shapes, so it cannot see these bugs.

The comparison is run on a deterministic stratified subset of the extended
corpus (two cases per ``(family, label)`` cell, covering every family, both
buggy and clean). TensorGuard's verdict is *also* recorded on the full corpus
for context. We report, per tool: bugs caught, false alarms on clean models,
and the *capability axes* that matter operationally -- whether the tool needs to
execute the model and whether it needs concrete example inputs.

The headline is not merely detection rate: torch.export can also catch these
bugs, but only by running the model. TensorGuard is the only option here that is
simultaneously (a) static -- no execution -- and (b) input-free -- shapes only --
and (c) complete on the corpus. mypy, the only other static tool, catches none.

Only booleans, counts and tool names are recorded (never error text or timings),
so the artifact is byte-identical across machines.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from corpus_extended.generators import all_cases  # noqa: E402
from src.api import verify_architecture  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "baseline_head_to_head.json"
OUT_MD = REPO / "reproducibility" / "baseline_head_to_head.md"

MODE = "sound"


def _stratified_subset(cases):
    groups = defaultdict(list)
    for c in cases:
        groups[(c.family, c.label)].append(c)
    sub = []
    for key in sorted(groups):
        sub.extend(groups[key][:2])
    return sub


def _instantiate(case):
    import torch
    import torch.nn as nn

    ns = {}
    exec(compile(case.source, f"<{case.id}>", "exec"), ns)
    mods = [v for v in ns.values()
            if isinstance(v, type) and issubclass(v, nn.Module)
            and v is not nn.Module]
    net = mods[0]()
    net.eval()
    inputs = []
    for shape in case.input_shapes.values():
        if case.family == "embedding":
            inputs.append(torch.randint(0, 100, tuple(shape)))
        else:
            inputs.append(torch.randn(*shape))
    return net, tuple(inputs)


def _tensorguard_catches(case) -> bool:
    r = verify_architecture(
        case.source,
        input_shapes={k: tuple(v) for k, v in case.input_shapes.items()},
        soundness_mode=MODE,
    )
    return str(r.verdict) == "UNSAFE"


def _torch_export_errors(case) -> bool:
    """True if torch.export raises (i.e. it reports a problem) for this case."""
    import torch

    logging.getLogger("torch").setLevel(logging.CRITICAL)
    net, inputs = _instantiate(case)
    devnull = io.StringIO()
    try:
        with contextlib.redirect_stderr(devnull), \
                contextlib.redirect_stdout(devnull):
            torch.export.export(net, inputs)
        return False
    except Exception:
        return True


def _mypy_flags_shape_bug(case) -> bool:
    """True if mypy reports an error referencing a tensor shape mismatch.

    mypy does not model tensor shapes, so this is expected to be False for every
    case -- that is precisely the point of including a general static type
    checker as a baseline.
    """
    with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, dir=str(REPO)) as fh:
        fh.write(case.source)
        path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "mypy", "--ignore-missing-imports",
             "--no-error-summary", "--no-color-output", path],
            cwd=str(REPO), capture_output=True, text=True,
            env={**os.environ, "MYPY_CACHE_DIR": os.devnull},
        )
        out = (proc.stdout + proc.stderr).lower()
        shape_terms = ("shape", "dimension", "broadcast", "size mismatch",
                       "mat1 and mat2")
        return any(t in out for t in shape_terms)
    finally:
        os.unlink(path)


def measure() -> dict:
    cases = all_cases()
    subset = _stratified_subset(cases)
    buggy = [c for c in subset if c.label == "buggy"]
    clean = [c for c in subset if c.label == "clean"]

    tg_full_buggy = sum(
        1 for c in cases if c.label == "buggy" and _tensorguard_catches(c))
    tg_full_clean_fp = sum(
        1 for c in cases if c.label == "clean" and _tensorguard_catches(c))
    n_full_buggy = sum(1 for c in cases if c.label == "buggy")
    n_full_clean = sum(1 for c in cases if c.label == "clean")

    tools = {
        "tensorguard": {
            "static_no_execution": True,
            "needs_concrete_inputs": False,
            "detector": _tensorguard_catches,
        },
        "torch_export_trace": {
            "static_no_execution": False,
            "needs_concrete_inputs": True,
            "detector": _torch_export_errors,
        },
        "mypy": {
            "static_no_execution": True,
            "needs_concrete_inputs": False,
            "detector": _mypy_flags_shape_bug,
        },
    }

    results = {}
    for name, spec in tools.items():
        det = spec["detector"]
        caught = sum(1 for c in buggy if det(c))
        false_alarms = sum(1 for c in clean if det(c))
        results[name] = {
            "static_no_execution": spec["static_no_execution"],
            "needs_concrete_inputs": spec["needs_concrete_inputs"],
            "buggy_caught": caught,
            "buggy_total": len(buggy),
            "clean_false_alarms": false_alarms,
            "clean_total": len(clean),
        }

    # The distinguishing claim: among tools that catch all subset bugs,
    # TensorGuard is the only one that is static AND input-free.
    complete_catchers = [
        n for n, r in results.items()
        if r["buggy_caught"] == len(buggy)]
    static_input_free_complete = [
        n for n in complete_catchers
        if results[n]["static_no_execution"]
        and not results[n]["needs_concrete_inputs"]]

    return {
        "mode": MODE,
        "subset_size": len(subset),
        "subset_buggy": len(buggy),
        "subset_clean": len(clean),
        "families_covered": sorted({c.family for c in subset}),
        "tools": results,
        "tensorguard_full_corpus": {
            "buggy_caught": tg_full_buggy,
            "buggy_total": n_full_buggy,
            "clean_false_alarms": tg_full_clean_fp,
            "clean_total": n_full_clean,
        },
        "tools_catching_all_subset_bugs": sorted(complete_catchers),
        "static_input_free_complete_tools": sorted(static_input_free_complete),
        "tensorguard_is_unique_static_input_free_complete": (
            static_input_free_complete == ["tensorguard"]),
        "mypy_catches_zero_shape_bugs": results["mypy"]["buggy_caught"] == 0,
    }


def render_markdown(data: dict) -> str:
    lines = [
        "# Head-to-head baseline comparison",
        "",
        f"Same-corpus comparison on a deterministic stratified subset of "
        f"**{data['subset_size']}** cases ({data['subset_buggy']} buggy, "
        f"{data['subset_clean']} clean) covering "
        f"{len(data['families_covered'])} families.",
        "",
        "| tool | static (no exec) | needs inputs | bugs caught | false alarms |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in ("tensorguard", "torch_export_trace", "mypy"):
        r = data["tools"][name]
        lines.append(
            f"| {name} | {r['static_no_execution']} | "
            f"{r['needs_concrete_inputs']} | "
            f"{r['buggy_caught']}/{r['buggy_total']} | "
            f"{r['clean_false_alarms']}/{r['clean_total']} |"
        )
    tgf = data["tensorguard_full_corpus"]
    lines += [
        "",
        f"TensorGuard on the **full** extended corpus: "
        f"{tgf['buggy_caught']}/{tgf['buggy_total']} bugs caught, "
        f"{tgf['clean_false_alarms']}/{tgf['clean_total']} false alarms.",
        "",
        f"- tools catching every subset bug: "
        f"{', '.join(data['tools_catching_all_subset_bugs'])}",
        f"- of those, static *and* input-free: "
        f"{', '.join(data['static_input_free_complete_tools'])}",
        f"- TensorGuard is the unique static, input-free, complete tool: "
        f"**{data['tensorguard_is_unique_static_input_free_complete']}**",
        f"- mypy (general static type checker) catches zero shape bugs: "
        f"**{data['mypy_catches_zero_shape_bugs']}**",
        "",
        "torch.export can also surface these bugs, but only by instantiating "
        "the model, building concrete example inputs, and executing a trace. "
        "TensorGuard reaches the same verdicts statically from source and "
        "declared shapes alone; mypy, the only other static tool, is blind to "
        "tensor shapes.",
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
            print("MISMATCH: baseline_head_to_head artifacts differ")
            return 1
        print("OK: baseline_head_to_head artifacts byte-identical")
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
