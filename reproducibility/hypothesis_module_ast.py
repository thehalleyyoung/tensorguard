"""Property-based (Hypothesis-style) full-module-AST testing with shrinking (Step 114).

Step 113 swept random *source strings* against the live torch dispatcher. Step 114
lifts that to a structured, *compositional* module algebra (see
``corpus_extended/module_ast.py``) and adds the second thing a reviewer-grade
property-based campaign needs: **shrinking to a minimal counterexample**.

This harness records three deterministic things:

  1. **A soundness sweep over structured module ASTs.** A seeded enumerator draws
     hundreds of full module ASTs (Linear/Conv/ReLU/Flatten chains across the
     2D-vector and 4D-image regimes, with adjacent-boundary compatibility decided
     by chance). Each AST is judged by the live eager-torch oracle (clean vs
     raises) and by TensorGuard's sound verdict, and the two load-bearing cells --
     a SAFE-but-raises soundness violation and an UNSAFE-but-clean false alarm --
     must stay empty.

  2. **A shrinking demonstration.** A deliberately large buggy module is reduced
     by a deterministic delta-debugging shrinker to a *locally minimal*
     counterexample under the predicate "eager torch raises" (which is exactly
     the cell an always-``SAFE`` broken verifier would miss). The shrinker drops
     layers and shrinks dimensions until no single reduction preserves the
     failure, yielding a two-line witness instead of a forty-line one.

  3. **Confirmation that the real verifier catches the shrunk witness.** The
     minimal counterexample produced by shrinking is fed to TensorGuard's actual
     sound verdict, which reports it ``UNSAFE`` -- so the machinery that *would*
     surface a soundness bug also confirms the real verifier is not buggy here.

Only counts, rounded rates, Wilson intervals, and the (small, fixed) minimal
counterexample source are recorded, so the artifact is byte-identical across
machines. The Hypothesis ``module_asts()`` strategy itself is exercised as a real
property-based test in ``tests/test_hypothesis_module_ast.py``.
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from corpus_extended.module_ast import (  # noqa: E402
    Conv2d,
    Flatten,
    Linear,
    ModuleAST,
    ReLU,
    random_module_ast,
    render,
    shrink_to_minimal,
    size,
    torch_runs_clean,
)
from src.api import verify_architecture  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "hypothesis_module_ast.json"
OUT_MD = REPO / "reproducibility" / "hypothesis_module_ast.md"

SEED = 20240602
N_GENERATED = 800


def _verdict(ast: ModuleAST) -> str:
    source, shapes = render(ast)
    return str(
        verify_architecture(
            source,
            input_shapes={k: tuple(v) for k, v in shapes.items()},
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


# A fixed, deliberately large buggy module used to demonstrate shrinking. Its
# very first Linear layer declares an in-dim (7) that does not match the input
# feature count (16), so eager torch raises; everything after it is irrelevant
# padding that a good shrinker must strip away.
_BUGGY_BIG = ModuleAST(
    regime="vec",
    input_shape=(4, 16),
    layers=(
        Linear(7, 8),
        ReLU(),
        Linear(8, 8),
        ReLU(),
        Linear(8, 16),
        ReLU(),
        Linear(16, 5),
    ),
)


def measure() -> dict:
    logging.disable(logging.CRITICAL)
    try:
        rng = random.Random(SEED)

        n_clean = 0
        n_raise = 0
        verdict_counts: Counter = Counter()
        regime_counts: Counter = Counter()
        soundness_violations: list = []
        false_alarms: list = []
        n_unknown = 0
        n_agree_decided = 0
        n_decided = 0

        for _ in range(N_GENERATED):
            ast = random_module_ast(rng)
            regime_counts[ast.regime] += 1
            clean = torch_runs_clean(ast)
            if clean:
                n_clean += 1
            else:
                n_raise += 1
            verdict = _verdict(ast)
            verdict_counts[verdict] += 1
            if verdict == "UNKNOWN":
                n_unknown += 1
                continue
            n_decided += 1
            safe = verdict == "SAFE"
            if safe and not clean:
                src, _ = render(ast)
                soundness_violations.append(src)
            elif (not safe) and clean:
                src, _ = render(ast)
                false_alarms.append(src)
            else:
                n_agree_decided += 1

        # --- shrinking demonstration -------------------------------------
        # Predicate models an always-SAFE *broken* verifier: a counterexample is
        # any module the real torch dispatcher rejects. Shrinking finds the
        # minimal such module.
        assert torch_runs_clean(_BUGGY_BIG) is False
        start_layers, start_dimsum = size(_BUGGY_BIG)
        minimal = shrink_to_minimal(_BUGGY_BIG, lambda a: not torch_runs_clean(a))
        min_layers, min_dimsum = size(minimal)
        min_source, _ = render(minimal)
        # The real verifier must catch the minimal witness.
        tg_verdict_on_minimal = _verdict(minimal)
        # Minimality proof: no single further reduction preserves the failure.
        from corpus_extended.module_ast import (
            _input_reductions,
            _layer_dim_reductions,
        )
        from dataclasses import replace as _replace

        further_reducible = False
        for i in range(len(minimal.layers)):
            cand = _replace(
                minimal, layers=minimal.layers[:i] + minimal.layers[i + 1 :]
            )
            if not torch_runs_clean(cand):
                further_reducible = True
        if not further_reducible:
            for i, layer in enumerate(minimal.layers):
                for repl in _layer_dim_reductions(layer):
                    cand = _replace(
                        minimal,
                        layers=minimal.layers[:i] + (repl,) + minimal.layers[i + 1 :],
                    )
                    if not torch_runs_clean(cand):
                        further_reducible = True
                        break
        if not further_reducible:
            for cand in _input_reductions(minimal):
                if not torch_runs_clean(cand):
                    further_reducible = True
                    break

        data = {
            "step": 114,
            "seed": SEED,
            "n_generated": N_GENERATED,
            "regime_counts": dict(sorted(regime_counts.items())),
            "oracle": {"n_clean": n_clean, "n_raise": n_raise},
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "soundness": {
                "n_decided": n_decided,
                "n_unknown": n_unknown,
                "n_soundness_violations": len(soundness_violations),
                "n_false_alarms": len(false_alarms),
                "n_agree_decided": n_agree_decided,
                "perfect_decided_agreement": (
                    n_decided > 0 and n_agree_decided == n_decided
                ),
                "soundness_violation_sources": sorted(soundness_violations),
                "false_alarm_sources": sorted(false_alarms),
            },
            "agreement_wilson": _wilson(n_agree_decided, n_decided),
            "shrinking_demo": {
                "start_n_layers": start_layers,
                "start_dim_sum": start_dimsum,
                "minimal_n_layers": min_layers,
                "minimal_dim_sum": min_dimsum,
                "layer_reduction_factor": (
                    round(start_layers / min_layers, 4) if min_layers else None
                ),
                "minimal_counterexample_source": min_source,
                "minimal_is_locally_minimal": not further_reducible,
                "real_verifier_catches_minimal": tg_verdict_on_minimal == "UNSAFE",
                "real_verifier_verdict_on_minimal": tg_verdict_on_minimal,
            },
        }
        return data
    finally:
        logging.disable(logging.NOTSET)


def render_markdown(data: dict) -> str:
    o = data["oracle"]
    s = data["soundness"]
    sh = data["shrinking_demo"]
    w = data["agreement_wilson"]
    lines = [
        "# Property-based full-module-AST testing with shrinking (Step 114)",
        "",
        f"Seed `{data['seed']}` — **{data['n_generated']}** structured module "
        "ASTs drawn from a compositional algebra (Linear / Conv2d / ReLU / "
        "Flatten across the 2D-vector and 4D-image regimes).",
        "",
        "## Soundness sweep vs the live torch dispatcher",
        "",
        f"- regimes generated: `{data['regime_counts']}`",
        f"- eager-torch oracle: **{o['n_clean']}** clean, **{o['n_raise']}** raise",
        f"- TensorGuard verdicts: `{data['verdict_counts']}`",
        f"- decided verdicts: **{s['n_decided']}** (abstentions: "
        f"{s['n_unknown']})",
        f"- soundness violations (SAFE but torch raises): "
        f"**{s['n_soundness_violations']}**",
        f"- false alarms (UNSAFE but torch clean): **{s['n_false_alarms']}**",
        f"- perfect agreement on all decided verdicts: "
        f"**{s['perfect_decided_agreement']}** "
        f"(Wilson {w['low']}–{w['high']})",
        "",
        "## Shrinking to a minimal counterexample",
        "",
        "A deliberately large buggy module is reduced by a deterministic "
        "delta-debugging shrinker under the predicate *eager torch raises* "
        "(the cell an always-`SAFE` broken verifier would miss):",
        "",
        f"- start: **{sh['start_n_layers']}** layers (dim-sum "
        f"{sh['start_dim_sum']})",
        f"- minimal: **{sh['minimal_n_layers']}** layer(s) (dim-sum "
        f"{sh['minimal_dim_sum']}); "
        f"layer reduction factor **{sh['layer_reduction_factor']}×**",
        f"- locally minimal (no single further reduction preserves the bug): "
        f"**{sh['minimal_is_locally_minimal']}**",
        f"- the *real* TensorGuard verifier catches the shrunk witness: "
        f"**{sh['real_verifier_catches_minimal']}** "
        f"(verdict `{sh['real_verifier_verdict_on_minimal']}`)",
        "",
        "Minimal counterexample source:",
        "",
        "```python",
        sh["minimal_counterexample_source"].rstrip("\n"),
        "```",
        "",
    ]
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = measure()
    js = json.dumps(data, indent=2, sort_keys=True) + "\n"
    md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != js:
            print(f"MISMATCH: {OUT_JSON}")
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != md:
            print(f"MISMATCH: {OUT_MD}")
            ok = False
        if ok:
            print("hypothesis_module_ast: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
