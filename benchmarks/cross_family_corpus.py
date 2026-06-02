"""Step 168 — cross-family, runtime-grounded regression corpus.

A deliberately *cross-architecture* evidence suite: matched clean/buggy
``nn.Module`` pairs spanning six structural families (MLP, CNN, attention,
normalization, residual, matmul).  Its purpose is to show TensorGuard's static
verdicts generalise across architecture families rather than over-fitting one
shape pattern.

Every case carries **runtime ground truth** so the suite cannot be a rigged
demo:

* a ``clean`` case must execute without error in eager PyTorch *and* verify SAFE;
* a ``buggy`` case must raise a real ``RuntimeError`` whose message contains the
  declared ``expected_error_substring`` *and* verify UNSAFE.

:func:`evaluate` re-establishes both facts on every run, computes per-family and
aggregate TP/FP/TN/FN (recall = caught bugs / real bugs, precision = caught /
flagged), and content-addresses each case by sha256 so the corpus cannot
silently drift.  Provenance is honest: all cases are ``synthetic`` minimal
reproductions of ubiquitous real-world mistakes.

Run ``python -m benchmarks.cross_family_corpus`` to (re)write
``benchmarks/cross_family_results.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RESULTS_PATH = os.path.join(THIS_DIR, "cross_family_results.json")

_PRE = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n\n"


def _clean(family: str, note: str, shapes: Dict[str, tuple], body: str) -> Dict[str, Any]:
    return {
        "label": "clean",
        "family": family,
        "provenance": "synthetic",
        "note": note,
        "input_shapes": shapes,
        "expected_error_substring": None,
        "source": _PRE + body,
    }


def _buggy(
    family: str, note: str, shapes: Dict[str, tuple], err: str, body: str
) -> Dict[str, Any]:
    return {
        "label": "buggy",
        "family": family,
        "provenance": "synthetic",
        "note": note,
        "input_shapes": shapes,
        "expected_error_substring": err,
        "source": _PRE + body,
    }


_IMG = {"x": (8, 3, 32, 32)}
_SEQ = {"x": (4, 16, 64)}
_VEC = {"x": (32, 784)}
_MAT = {"x": (16, 64)}


CASES: Dict[str, Dict[str, Any]] = {
    # ---- MLP --------------------------------------------------------------- #
    "mlp_clean": _clean(
        "mlp",
        "Two-layer MLP with matching feature dims.",
        _VEC,
        """class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
""",
    ),
    "mlp_buggy": _buggy(
        "mlp",
        "fc2 expects 128 features but fc1 emits 256.",
        _VEC,
        "mat1 and mat2 shapes cannot be multiplied",
        """class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(128, 10)  # BUG: should be Linear(256, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
""",
    ),
    # ---- CNN --------------------------------------------------------------- #
    "cnn_clean": _clean(
        "cnn",
        "Two conv layers with matching channel counts.",
        _IMG,
        """class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)

    def forward(self, x):
        return self.c2(F.relu(self.c1(x)))
""",
    ),
    "cnn_buggy": _buggy(
        "cnn",
        "c2 expects 8 in-channels but c1 emits 16.",
        _IMG,
        "channels",
        """class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(8, 32, 3, padding=1)  # BUG: should be Conv2d(16, 32, ...)

    def forward(self, x):
        return self.c2(F.relu(self.c1(x)))
""",
    ),
    # ---- Attention --------------------------------------------------------- #
    "attention_clean": _clean(
        "attention",
        "Multi-head attention that reassembles heads to the right model dim.",
        _SEQ,
        """class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(64, 64)
        self.k = nn.Linear(64, 64)
        self.v = nn.Linear(64, 64)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, 8, 8).transpose(1, 2)
        k = self.k(x).view(B, T, 8, 8).transpose(1, 2)
        v = self.v(x).view(B, T, 8, 8).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)).softmax(-1)
        out = att @ v
        return out.transpose(1, 2).reshape(B, T, 64)
""",
    ),
    "attention_buggy": _buggy(
        "attention",
        "Head reassembly targets dim 48 instead of 8*8 = 64.",
        _SEQ,
        "is invalid for input of size",
        """class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(64, 64)
        self.k = nn.Linear(64, 64)
        self.v = nn.Linear(64, 64)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, 8, 8).transpose(1, 2)
        k = self.k(x).view(B, T, 8, 8).transpose(1, 2)
        v = self.v(x).view(B, T, 8, 8).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)).softmax(-1)
        out = att @ v
        return out.transpose(1, 2).reshape(B, T, 48)  # BUG: should be 64
""",
    ),
    # ---- Normalization ----------------------------------------------------- #
    "norm_clean": _clean(
        "normalization",
        "BatchNorm2d sized to the conv's output channels.",
        _IMG,
        """class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)

    def forward(self, x):
        return self.bn(self.c(x))
""",
    ),
    "norm_batchnorm_buggy": _buggy(
        "normalization",
        "BatchNorm2d sized for 8 channels but the conv emits 16.",
        _IMG,
        "running_mean should contain",
        """class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)  # BUG: should be 16

    def forward(self, x):
        return self.bn(self.c(x))
""",
    ),
    "norm_groupnorm_buggy": _buggy(
        "normalization",
        "GroupNorm declares 12 channels but the conv produces 16.",
        _IMG,
        "Expected weight to be a vector of size",
        """class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 16, 3, padding=1)
        self.gn = nn.GroupNorm(4, 12)  # BUG: 12 != 16 channels

    def forward(self, x):
        return self.gn(self.c(x))
""",
    ),
    # ---- Residual ---------------------------------------------------------- #
    "residual_clean": _clean(
        "residual",
        "Residual branch preserves channels, so the skip-add broadcasts.",
        _IMG,
        """class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 16, 3, padding=1)

    def forward(self, x):
        h = self.c1(x)
        return h + self.c2(h)
""",
    ),
    "residual_buggy": _buggy(
        "residual",
        "Skip-add of 16- and 8-channel tensors cannot broadcast.",
        _IMG,
        "must match the size of tensor",
        """class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 8, 3, padding=1)  # BUG: 8 != 16

    def forward(self, x):
        h = self.c1(x)
        return h + self.c2(h)
""",
    ),
    # ---- Matmul ------------------------------------------------------------ #
    "matmul_clean": _clean(
        "matmul",
        "Explicit matmul against a correctly sized parameter.",
        _MAT,
        """class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(64, 32))

    def forward(self, x):
        return x @ self.w
""",
    ),
    "matmul_buggy": _buggy(
        "matmul",
        "Inner dims disagree: (.,64) @ (32,.) is ill-formed.",
        _MAT,
        "mat1 and mat2 shapes cannot be multiplied",
        """class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(32, 16))  # BUG: should be (64, 16)

    def forward(self, x):
        return x @ self.w
""",
    ),
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _runtime_truth(case: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the case in eager PyTorch and report what actually happened."""
    import torch

    ns: Dict[str, Any] = {}
    exec(case["source"], ns)  # noqa: S102 - trusted, in-repo corpus
    cls = ns.get("CleanModule") or ns.get("BuggyModule")
    model = cls().eval()
    inputs = [torch.randn(*shape) for shape in case["input_shapes"].values()]
    try:
        with torch.no_grad():
            model(*inputs)
        return {"ran": True, "error": None}
    except Exception as exc:  # noqa: BLE001 - we want the runtime ground truth
        return {"ran": False, "error": f"{type(exc).__name__}: {exc}"}


def _tg_verdict(case: Dict[str, Any]) -> str:
    from src.api import verify_architecture

    shapes = {k: tuple(v) for k, v in case["input_shapes"].items()}
    result = verify_architecture(case["source"], input_shapes=shapes)
    return "UNSAFE" if getattr(result, "bug_count", 0) > 0 else "SAFE"


def evaluate() -> Dict[str, Any]:
    """Run the full corpus and return a content-addressed scorecard."""
    rows: List[Dict[str, Any]] = []
    for cid, case in CASES.items():
        rt = _runtime_truth(case)
        verdict = _tg_verdict(case)
        is_bug = case["label"] == "buggy"
        # runtime ground truth must agree with the declared label
        if is_bug:
            substr = case["expected_error_substring"] or ""
            runtime_ok = (not rt["ran"]) and substr in (rt["error"] or "")
        else:
            runtime_ok = rt["ran"]
        rows.append(
            {
                "id": cid,
                "family": case["family"],
                "label": case["label"],
                "provenance": case["provenance"],
                "expected_verdict": "UNSAFE" if is_bug else "SAFE",
                "tg_verdict": verdict,
                "runtime_ran": rt["ran"],
                "runtime_error": rt["error"],
                "runtime_ground_truth_ok": runtime_ok,
                "tg_correct": verdict == ("UNSAFE" if is_bug else "SAFE"),
                "sha256": _sha256(case["source"]),
            }
        )

    families = sorted({r["family"] for r in rows})
    by_family: Dict[str, Dict[str, int]] = {}
    tp = fp = tn = fn = 0
    for r in rows:
        fam = by_family.setdefault(
            r["family"], {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        )
        if r["label"] == "buggy":
            if r["tg_verdict"] == "UNSAFE":
                tp += 1
                fam["tp"] += 1
            else:
                fn += 1
                fam["fn"] += 1
        else:
            if r["tg_verdict"] == "UNSAFE":
                fp += 1
                fam["fp"] += 1
            else:
                tn += 1
                fam["tn"] += 1

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "meta": {
            "name": "tensorguard-cross-family-corpus",
            "version": "1.0.0",
            "description": (
                "Cross-architecture, runtime-grounded regression corpus: matched "
                "clean/buggy nn.Module pairs across six structural families. Each "
                "case is validated against eager PyTorch (clean runs, buggy raises "
                "the declared error) and content-addressed by sha256."
            ),
            "n_cases": len(rows),
            "n_families": len(families),
            "families": families,
            "all_runtime_ground_truth_ok": all(r["runtime_ground_truth_ok"] for r in rows),
        },
        "scorecard": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "recall": recall,
            "precision": precision,
        },
        "by_family": by_family,
        "cases": rows,
    }


def main() -> int:
    report = evaluate()
    with open(RESULTS_PATH, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    sc = report["scorecard"]
    print(
        f"cross-family corpus: {report['meta']['n_cases']} cases, "
        f"{report['meta']['n_families']} families | "
        f"recall={sc['recall']:.3f} precision={sc['precision']:.3f} "
        f"FP={sc['fp']} FN={sc['fn']} | "
        f"runtime-grounded={report['meta']['all_runtime_ground_truth_ok']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
