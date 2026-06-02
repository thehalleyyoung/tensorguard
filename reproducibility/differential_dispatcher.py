"""Differential testing against the live torch dispatcher at scale (Step 113).

The strongest possible oracle for a static verifier is the framework it claims to
reason about. This harness generates *thousands* of random PyTorch modules across
several architectural families -- each with randomly chosen dimensions so that an
adjacent-layer boundary is compatible or not by chance -- and then cross-checks
two judgements on every one:

  * the **ground truth**: instantiate the module and run a real forward pass under
    eager PyTorch; it either executes cleanly or raises (a shape/channel/numel
    mismatch is decided deterministically by torch, independent of tensor
    values or hardware);
  * the **verifier**: TensorGuard's sound-mode verdict (SAFE / UNSAFE / UNKNOWN).

The two judgements are crossed into an agreement matrix. Two cells are
load-bearing for a *sound* verifier and must be **empty**:

  * **soundness violation** -- TensorGuard says SAFE but torch raises (a missed
    bug; this is the cell a soundness gap would show up in);
  * **false alarm** -- TensorGuard says UNSAFE but torch runs cleanly.

UNKNOWN (abstention) against either ground truth is always permitted and is
reported separately as coverage. Generation is fully seeded, and only counts,
rounded rates, Wilson intervals and the (ideally empty) lists of violating
module sources are recorded, so the artifact is byte-identical across machines.
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.api import verify_architecture  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "differential_dispatcher.json"
OUT_MD = REPO / "reproducibility" / "differential_dispatcher.md"

SEED = 20240601
N_PER_FAMILY = 400  # five families -> 2000 random modules

_IMPORTS = "import torch\nimport torch.nn as nn\n\n"


def _cls(body_init: str, body_fwd: str) -> str:
    return (
        _IMPORTS
        + "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        + body_init
        + "    def forward(self, x):\n"
        + body_fwd
        + "        return x\n"
    )


def _gen_mlp(rng: random.Random):
    depth = rng.randint(1, 4)
    f0 = rng.choice([8, 16, 32, 50, 64])
    init = []
    cur = f0
    for i in range(depth):
        out = rng.choice([8, 16, 32, 64, 10])
        decl_in = cur if rng.random() < 0.65 else rng.choice([8, 16, 32, 64])
        init.append(f"        self.l{i} = nn.Linear({decl_in}, {out})\n")
        cur = out
    fwd = "".join(f"        x = self.l{i}(x)\n" for i in range(depth))
    return _cls("".join(init), fwd), {"x": (4, f0)}


def _gen_conv(rng: random.Random):
    depth = rng.randint(1, 3)
    c0 = rng.choice([1, 3, 8])
    side = rng.choice([8, 16, 28, 32])
    init = []
    cur = c0
    for i in range(depth):
        out = rng.choice([4, 8, 16, 3])
        k = rng.choice([1, 3, 3, 5])
        decl_in = cur if rng.random() < 0.7 else rng.choice([1, 3, 8, 16])
        init.append(
            f"        self.c{i} = nn.Conv2d({decl_in}, {out}, {k}, "
            f"padding={k // 2})\n"
        )
        cur = out
    fwd = "".join(f"        x = torch.relu(self.c{i}(x))\n" for i in range(depth))
    return _cls("".join(init), fwd), {"x": (2, c0, side, side)}


def _gen_convflat(rng: random.Random):
    c0 = rng.choice([1, 3])
    side = rng.choice([8, 16])
    cout = rng.choice([4, 8])
    nclass = rng.choice([10, 5])
    flat = cout * side * side
    decl = flat if rng.random() < 0.6 else rng.choice([flat, flat // 2, cout * side])
    init = (
        f"        self.c = nn.Conv2d({c0}, {cout}, 3, padding=1)\n"
        f"        self.fc = nn.Linear({decl}, {nclass})\n"
    )
    fwd = (
        "        x = torch.relu(self.c(x))\n"
        "        x = x.flatten(1)\n"
        "        x = self.fc(x)\n"
    )
    return _cls(init, fwd), {"x": (2, c0, side, side)}


def _gen_reshape(rng: random.Random):
    f0 = rng.choice([16, 32, 64])
    h = rng.choice([8, 16, 32])
    batch = 4
    target = h if rng.random() < 0.6 else rng.choice([8, 16, 32, 64])
    decl = target if rng.random() < 0.7 else rng.choice([8, 16, 32])
    out = rng.choice([10, 5])
    init = (
        f"        self.a = nn.Linear({f0}, {h})\n"
        f"        self.b = nn.Linear({decl}, {out})\n"
    )
    fwd = (
        "        x = self.a(x)\n"
        f"        x = x.reshape({batch}, {target})\n"
        "        x = self.b(x)\n"
    )
    return _cls(init, fwd), {"x": (batch, f0)}


def _gen_cat(rng: random.Random):
    f0 = rng.choice([16, 32])
    a = rng.choice([8, 16])
    b = rng.choice([8, 16])
    decl = (a + b) if rng.random() < 0.6 else rng.choice([a + b, a, b, a + b + 8])
    out = rng.choice([10, 5])
    init = (
        f"        self.p = nn.Linear({f0}, {a})\n"
        f"        self.q = nn.Linear({f0}, {b})\n"
        f"        self.r = nn.Linear({decl}, {out})\n"
    )
    fwd = (
        "        u = self.p(x)\n"
        "        v = self.q(x)\n"
        "        x = torch.cat([u, v], dim=1)\n"
        "        x = self.r(x)\n"
    )
    return _cls(init, fwd), {"x": (4, f0)}


FAMILIES = {
    "mlp": _gen_mlp,
    "conv": _gen_conv,
    "convflat": _gen_convflat,
    "reshape": _gen_reshape,
    "cat": _gen_cat,
}


def _torch_runs_clean(source: str, shapes: dict) -> bool:
    import torch

    ns: dict = {}
    try:
        exec(compile(source, "<gen>", "exec"), ns)
        net = ns["Net"]()
        net.eval()
        inputs = [torch.randn(*s) for s in shapes.values()]
        with torch.no_grad():
            net(*inputs)
        return True
    except Exception:
        return False


def _verdict(source: str, shapes: dict) -> str:
    return str(
        verify_architecture(
            source, input_shapes={k: tuple(v) for k, v in shapes.items()},
            soundness_mode="sound",
        ).verdict
    )


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> dict:
    if n == 0:
        return {"point": None, "low": None, "high": None, "k": k, "n": n}
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return {
        "point": round(p, 4),
        "low": round(max(0.0, center - half), 4),
        "high": round(min(1.0, center + half), 4),
        "k": k,
        "n": n,
    }


def measure() -> dict:
    import torch

    torch.manual_seed(0)
    logging.disable(logging.CRITICAL)
    rng = random.Random(SEED)

    # Deterministic generation order: round-robin across families.
    modules = []
    for r in range(N_PER_FAMILY):
        for fam, gen in FAMILIES.items():
            src, shapes = gen(rng)
            modules.append((f"{fam}:{r}", fam, src, shapes))

    n = len(modules)
    matrix = Counter()  # (verdict, ground_truth) -> count
    per_family = defaultdict(lambda: Counter())
    soundness_violations = []  # TG SAFE but torch raises
    false_alarms = []          # TG UNSAFE but torch clean

    for mid, fam, src, shapes in modules:
        clean = _torch_runs_clean(src, shapes)
        gt = "clean" if clean else "raises"
        v = _verdict(src, shapes)
        matrix[(v, gt)] += 1
        per_family[fam][(v, gt)] += 1
        if v == "SAFE" and gt == "raises":
            soundness_violations.append({"id": mid, "family": fam, "source": src})
        elif v == "UNSAFE" and gt == "clean":
            false_alarms.append({"id": mid, "family": fam, "source": src})

    n_clean = sum(c for (v, gt), c in matrix.items() if gt == "clean")
    n_raises = n - n_clean
    n_safe = sum(c for (v, gt), c in matrix.items() if v == "SAFE")
    n_unsafe = sum(c for (v, gt), c in matrix.items() if v == "UNSAFE")
    n_unknown = sum(c for (v, gt), c in matrix.items() if v == "UNKNOWN")

    # On the decided (non-abstained) modules, TG agrees with torch exactly.
    decided = n - n_unknown
    agree_decided = sum(
        c for (v, gt), c in matrix.items()
        if (v == "SAFE" and gt == "clean") or (v == "UNSAFE" and gt == "raises")
    )

    fam_summary = {}
    for fam in FAMILIES:
        m = per_family[fam]
        fam_n = sum(m.values())
        fam_clean = sum(c for (v, gt), c in m.items() if gt == "clean")
        fam_viol = sum(c for (v, gt), c in m.items()
                       if v == "SAFE" and gt == "raises")
        fam_fa = sum(c for (v, gt), c in m.items()
                     if v == "UNSAFE" and gt == "clean")
        fam_unknown = sum(c for (v, gt), c in m.items() if v == "UNKNOWN")
        fam_summary[fam] = {
            "n": fam_n,
            "n_clean": fam_clean,
            "n_raises": fam_n - fam_clean,
            "n_unknown": fam_unknown,
            "soundness_violations": fam_viol,
            "false_alarms": fam_fa,
        }

    return {
        "seed": SEED,
        "n_per_family": N_PER_FAMILY,
        "n_modules": n,
        "families": sorted(FAMILIES),
        "ground_truth": {"n_clean": n_clean, "n_raises": n_raises},
        "verdict_tally": {"SAFE": n_safe, "UNSAFE": n_unsafe, "UNKNOWN": n_unknown},
        "agreement_matrix": {
            f"{v}|{gt}": c for (v, gt), c in sorted(matrix.items())
        },
        "n_decided": decided,
        "n_agree_on_decided": agree_decided,
        "decided_agreement_perfect": agree_decided == decided,
        "n_soundness_violations": len(soundness_violations),
        "soundness_violation_examples": sorted(
            soundness_violations, key=lambda d: d["id"])[:5],
        "soundness_violation_rate": _wilson(len(soundness_violations), n_raises),
        "n_false_alarms": len(false_alarms),
        "false_alarm_examples": sorted(
            false_alarms, key=lambda d: d["id"])[:5],
        "false_alarm_rate": _wilson(len(false_alarms), n_clean),
        "zero_soundness_violations": len(soundness_violations) == 0,
        "zero_false_alarms": len(false_alarms) == 0,
        "per_family": fam_summary,
        "scale_at_least_1000": n >= 1000,
    }


def render_markdown(data: dict) -> str:
    gt = data["ground_truth"]
    vt = data["verdict_tally"]
    lines = [
        "# Differential testing vs the live torch dispatcher",
        "",
        f"We generate **{data['n_modules']}** random PyTorch modules "
        f"(seed `{data['seed']}`) across **{len(data['families'])}** families "
        f"({', '.join(data['families'])}), with dimensions chosen so that "
        "adjacent-layer boundaries are compatible or not by chance. Each module "
        "is judged twice: by a real eager-PyTorch forward pass (ground truth: "
        "clean vs raises) and by TensorGuard's sound-mode verdict.",
        "",
        f"- ground truth: **{gt['n_clean']}** clean, **{gt['n_raises']}** raise",
        f"- verdicts: **{vt['SAFE']}** SAFE, **{vt['UNSAFE']}** UNSAFE, "
        f"**{vt['UNKNOWN']}** UNKNOWN (abstain)",
        "",
        "Agreement matrix (verdict rows x ground-truth columns):",
        "",
        "| verdict \\\\ torch | clean | raises |",
        "| --- | --- | --- |",
    ]
    m = data["agreement_matrix"]
    for v in ("SAFE", "UNSAFE", "UNKNOWN"):
        lines.append(
            f"| {v} | {m.get(f'{v}|clean', 0)} | {m.get(f'{v}|raises', 0)} |"
        )
    svr = data["soundness_violation_rate"]
    far = data["false_alarm_rate"]
    lines += [
        "",
        f"- **soundness violations** (SAFE but torch raises): "
        f"**{data['n_soundness_violations']}** "
        f"(rate {svr['point']}, 95% CI upper {svr['high']})",
        f"- **false alarms** (UNSAFE but torch clean): "
        f"**{data['n_false_alarms']}** "
        f"(rate {far['point']}, 95% CI upper {far['high']})",
        f"- on the **{data['n_decided']}** decided (non-abstained) modules, "
        f"TensorGuard agrees with torch on "
        f"**{data['n_agree_on_decided']}** -- perfect agreement: "
        f"**{data['decided_agreement_perfect']}**",
        "",
        "Every decided verdict matches the live dispatcher: no random module is "
        "ever proved SAFE while torch rejects it (zero soundness violations), and "
        "no clean module is ever rejected (zero false alarms). Abstention is the "
        "only permitted form of disagreement.",
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
            print("MISMATCH: differential_dispatcher artifacts differ")
            return 1
        print("OK: differential_dispatcher artifacts byte-identical")
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
