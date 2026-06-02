"""Deterministic harness: the proposed upstream verification hook vs real torch.

Proves the reference hook in :mod:`src.upstream_hook` against real PyTorch:

* a **buggy** module (chained Linears with mismatched features) -- real
  PyTorch raises a ``RuntimeError`` deep inside ``aten`` at forward time; with
  ``attach_verifier`` the same module is rejected at the *boundary* with a
  precise :class:`ShapeVerificationError` and a one-line diagnostic.
* a **clean** module -- the verifier proves it safe, the hook is transparent,
  and the real forward runs and returns the expected output shape.
* the **@verifiable** decorator -- attaching at construction reproduces the
  same accept/reject behaviour.

Only booleans / verdict strings / shape ints are recorded so the artifact is
byte-identical across machines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.upstream_hook import (  # noqa: E402
    ShapeVerificationError,
    attach_verifier,
    verifiable,
    verify_nn_module,
)

OUT_JSON = REPO / "reproducibility" / "upstream_hook_demo.json"
OUT_MD = REPO / "reproducibility" / "upstream_hook_demo.md"


def _make_buggy():
    import torch.nn as nn

    class Buggy(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 4)
            self.b = nn.Linear(5, 2)  # expects 5, gets 4 -> mismatch

        def forward(self, x):
            return self.b(self.a(x))

    return Buggy


def _make_clean():
    import torch.nn as nn

    class Clean(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 4)
            self.b = nn.Linear(4, 2)

        def forward(self, x):
            return self.b(self.a(x))

    return Clean


def _buggy_real_forward_errors() -> bool:
    import torch

    Buggy = _make_buggy()
    m = Buggy()
    try:
        m(torch.randn(2, 8))
        return False
    except RuntimeError:
        return True


def _hook_rejects_buggy() -> bool:
    Buggy = _make_buggy()
    import torch

    m = Buggy()
    attach_verifier(m, input_shapes={"x": (2, 8)}, soundness_mode="sound")
    try:
        m(torch.randn(2, 8))
        return False
    except ShapeVerificationError:
        return True


def _hook_transparent_on_clean() -> dict:
    import torch

    Clean = _make_clean()
    m = Clean()
    attach_verifier(m, input_shapes={"x": (2, 8)}, soundness_mode="sound")
    y = m(torch.randn(2, 8))
    return {"forward_ran": True, "out_shape": list(y.shape)}


def _decorator_accepts_clean_rejects_buggy() -> dict:
    import torch
    import torch.nn as nn

    @verifiable(input_shapes={"x": (2, 8)}, soundness_mode="sound")
    class CleanD(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 4)
            self.b = nn.Linear(4, 2)

        def forward(self, x):
            return self.b(self.a(x))

    @verifiable(input_shapes={"x": (2, 8)}, soundness_mode="sound")
    class BuggyD(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 4)
            self.b = nn.Linear(5, 2)

        def forward(self, x):
            return self.b(self.a(x))

    clean_ok = False
    try:
        CleanD()(torch.randn(2, 8))
        clean_ok = True
    except ShapeVerificationError:
        clean_ok = False

    buggy_rejected = False
    try:
        BuggyD()(torch.randn(2, 8))
    except ShapeVerificationError:
        buggy_rejected = True

    return {"clean_accepted": clean_ok, "buggy_rejected": buggy_rejected}


def measure() -> dict:
    Buggy = _make_buggy()
    Clean = _make_clean()

    buggy_verdict = verify_nn_module(
        Buggy(), input_shapes={"x": (2, 8)}, soundness_mode="sound"
    ).verdict
    clean_verdict = verify_nn_module(
        Clean(), input_shapes={"x": (2, 8)}, soundness_mode="sound"
    ).verdict

    real_raises = _buggy_real_forward_errors()
    hook_rejects = _hook_rejects_buggy()
    clean_hook = _hook_transparent_on_clean()
    deco = _decorator_accepts_clean_rejects_buggy()

    return {
        "buggy_static_verdict": buggy_verdict,
        "clean_static_verdict": clean_verdict,
        "buggy_real_forward_errors": real_raises,
        "hook_rejects_buggy_before_forward": hook_rejects,
        "hook_transparent_on_clean": clean_hook,
        "decorator_behavior": deco,
        # the headline equivalence: static rejection iff real runtime failure
        "static_matches_live": (
            (buggy_verdict == "UNSAFE") == real_raises
            and hook_rejects == real_raises
        ),
        "all_consistent": (
            buggy_verdict == "UNSAFE"
            and clean_verdict == "SAFE"
            and real_raises is True
            and hook_rejects is True
            and clean_hook["forward_ran"] is True
            and deco["clean_accepted"] is True
            and deco["buggy_rejected"] is True
        ),
    }


def render_markdown(data: dict) -> str:
    ch = data["hook_transparent_on_clean"]
    deco = data["decorator_behavior"]
    lines = [
        "# Proposed upstream verification hook vs real PyTorch",
        "",
        "Reference implementation (`src/upstream_hook.py`) for the upstream "
        "proposal in [`docs/upstream/pytorch_proposal.md`](../docs/upstream/pytorch_proposal.md). "
        "It shows how PyTorch could let any `nn.Module` be statically verified "
        "with zero changes to model code.",
        "",
        "| property | value |",
        "| --- | --- |",
        f"| buggy module static verdict | `{data['buggy_static_verdict']}` |",
        f"| clean module static verdict | `{data['clean_static_verdict']}` |",
        f"| buggy real `forward` raises at runtime | "
        f"{data['buggy_real_forward_errors']} |",
        f"| attached hook rejects buggy *before* forward | "
        f"{data['hook_rejects_buggy_before_forward']} |",
        f"| hook transparent on clean (forward ran) | {ch['forward_ran']} |",
        f"| clean forward output shape | {ch['out_shape']} |",
        f"| @verifiable accepts clean | {deco['clean_accepted']} |",
        f"| @verifiable rejects buggy | {deco['buggy_rejected']} |",
        f"| static rejection iff runtime failure | "
        f"{data['static_matches_live']} |",
        "",
        f"**All consistent: {data['all_consistent']}.** The hook turns a deep "
        "`aten`-level runtime stack trace into a precise diagnostic raised at "
        "the module boundary, while remaining completely transparent (and "
        "non-breaking) for modules it proves safe.",
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
            print("MISMATCH: upstream_hook_demo artifacts differ")
            return 1
        print("OK: upstream_hook_demo artifacts byte-identical")
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
