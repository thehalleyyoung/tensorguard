"""Deterministic harness: mutation testing of clean models (Step 112).

Step 111 proved the verifier raises no false alarms on a large clean corpus. The
dual question is *sensitivity*: if we inject a single, genuine bug into an
otherwise-clean model, does the verifier catch it? This harness answers that with
a classical *mutation-testing* protocol:

  1. Take every clean model in the stress + natural corpora (each is known to run
     under eager PyTorch).
  2. Apply each local mutation operator (``corpus_extended/model_mutators.py``):
     bump a Linear/Conv in/out width by one, or cast the forward input to an
     integer dtype.
  3. **Validate** the mutant is a *genuine runtime bug*: instantiate it and run a
     forward pass under real PyTorch; keep it only if (a) the parent model runs
     cleanly and (b) the mutant raises. Operator applications that still execute
     (e.g. bumping the out-features of a final layer) are discarded -- they are
     not bugs, so a SAFE verdict on them would be *correct*.
  4. Automatically minimize each admitted mutant with deterministic
     delta-debugging over source lines. A deletion is accepted only if the
     minimized mutant still instantiates, still raises under real PyTorch, and
     preserves the original exception class plus message prefix. This gives each
     admitted mutant a compact, certified reproducer instead of an uninspected
     synthetic edit.
  5. Score the verifier on each minimized genuine-bug mutant. A mutant is
     **killed** if the
     verifier returns ``UNSAFE`` and **survived** otherwise. A *survived* mutant
     that the verifier calls ``SAFE`` is the dangerous kind (a missed bug); one
     it calls ``UNKNOWN`` merely declined.

We report the overall kill rate (Wilson interval), the kill rate per operator and
per target domain, and -- honestly -- the explicit ids of any survivors. Only
counts, rounded rates, intervals and sorted id lists are written, so the artifact
is byte-identical across machines.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from corpus_extended.fp_stress import all_models as stress_models  # noqa: E402
from corpus_extended.natural_models import all_models as natural_models  # noqa: E402
from corpus_extended.model_mutators import (  # noqa: E402
    OPERATORS,
    OPERATOR_DOMAIN,
)
from src.api import verify_architecture  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "mutation_clean_models.json"
OUT_MD = REPO / "reproducibility" / "mutation_clean_models.md"

MODES = ["sound", "balanced", "heuristic"]


def _logical_lines(source: str) -> int:
    return sum(1 for line in source.splitlines() if line.strip())


def _runtime_error_signature(source: str, input_shapes: dict) -> tuple[str, str] | None:
    """Return the PyTorch runtime failure signature, or None if forward succeeds."""
    import torch as _t

    ns: dict = {}
    try:
        exec(compile(source, "<model>", "exec"), ns)
        net = ns["Net"]()
        net.eval()
    except Exception:
        return None  # construction errors are not admitted mutation bugs
    inputs = _make_inputs(source, input_shapes)
    try:
        with _t.no_grad():
            net(*inputs.values())
        return None
    except Exception as exc:
        message = " ".join(str(exc).split())
        # Keep enough text to distinguish dtype/shape/channel failures while
        # avoiding machine-specific tensor repr details.
        return (type(exc).__name__, message[:160])


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


def _make_inputs(source: str, input_shapes: dict):
    import torch

    torch.manual_seed(0)
    tensors = {}
    use_int = "Embedding" in source
    for name, shape in input_shapes.items():
        shape = tuple(shape)
        if use_int:
            tensors[name] = torch.randint(0, 2, shape)
        else:
            tensors[name] = torch.randn(*shape)
    return tensors


def _runs_clean(source: str, input_shapes: dict) -> bool:
    """True iff the model instantiates and forward-passes without raising."""
    import torch  # noqa: F401

    ns: dict = {}
    try:
        exec(compile(source, "<model>", "exec"), ns)
        net = ns["Net"]()
        net.eval()
        inputs = _make_inputs(source, input_shapes)
        import torch as _t

        with _t.no_grad():
            net(*inputs.values())
        return True
    except Exception:
        return False


def _raises(source: str, input_shapes: dict) -> bool:
    """True iff the model instantiates fine but forward raises (a genuine bug)."""
    return _runtime_error_signature(source, input_shapes) is not None


def _minimize_source(source: str, input_shapes: dict) -> tuple[str, dict]:
    """Delta-debug source lines while preserving the real runtime failure.

    This is intentionally syntax-agnostic: candidate deletions must compile,
    instantiate ``Net``, and reproduce the same PyTorch exception signature before
    they are accepted. A final single-line pass certifies 1-minimality under this
    deletion relation.
    """
    target = _runtime_error_signature(source, input_shapes)
    if target is None:
        raise ValueError("cannot minimize a mutant that does not fail at runtime")

    attempts = 0

    def preserves(candidate_lines: list[str]) -> bool:
        nonlocal attempts
        attempts += 1
        if not any(line.startswith("class Net") for line in candidate_lines):
            return False
        candidate = "\n".join(candidate_lines).rstrip() + "\n"
        return _runtime_error_signature(candidate, input_shapes) == target

    lines = source.splitlines()
    n = 2
    while len(lines) >= 2:
        chunk = max(1, len(lines) // n)
        ranges = [(i, min(i + chunk, len(lines))) for i in range(0, len(lines), chunk)]
        reduced = False
        for start, end in ranges:
            candidate = lines[:start] + lines[end:]
            if candidate and preserves(candidate):
                lines = candidate
                n = max(2, n - 1)
                reduced = True
                break
        if not reduced:
            if n >= len(lines):
                break
            n = min(len(lines), n * 2)

    # Certify local minimality after chunk shrinking.
    locally_minimal = True
    for i in range(len(lines)):
        if preserves(lines[:i] + lines[i + 1:]):
            locally_minimal = False
            break

    minimized = "\n".join(lines).rstrip() + "\n"
    signature = _runtime_error_signature(minimized, input_shapes)
    original_lines = _logical_lines(source)
    minimized_lines = _logical_lines(minimized)
    return minimized, {
        "original_logical_lines": original_lines,
        "minimized_logical_lines": minimized_lines,
        "removed_logical_lines": original_lines - minimized_lines,
        "attempts": attempts,
        "exception_type": target[0],
        "exception_prefix": target[1],
        "same_failure_signature": signature == target,
        "one_line_minimal": locally_minimal,
        "minimized_sha256": hashlib.sha256(minimized.encode()).hexdigest(),
    }


def _verdict(source: str, input_shapes: dict, mode: str) -> str:
    r = verify_architecture(
        source,
        input_shapes={k: tuple(v) for k, v in input_shapes.items()},
        soundness_mode=mode,
    )
    return str(r.verdict)


def _iter_clean_models():
    for m in stress_models():
        yield ("stress", m.id, m.family, m.source, dict(m.input_shapes))
    for m in natural_models():
        yield ("natural", m.id, m.family, m.source, dict(m.input_shapes))


def measure() -> dict:
    # Build the genuine-bug mutant set deterministically.
    mutants = []  # list of dicts
    parents_validated = set()
    parents_clean = set()
    for corpus, mid, family, source, shapes in _iter_clean_models():
        key = (corpus, mid)
        if key not in parents_validated:
            parents_validated.add(key)
            if _runs_clean(source, shapes):
                parents_clean.add(key)
        if key not in parents_clean:
            continue  # only mutate models we confirmed clean
        for op_name, op in OPERATORS.items():
            mutated = op(source)
            if mutated is None or mutated == source:
                continue
            if not _raises(mutated, shapes):
                continue  # operator produced a non-bug; discard
            minimized, minimization = _minimize_source(mutated, shapes)
            mutants.append({
                "corpus": corpus,
                "parent": mid,
                "family": family,
                "operator": op_name,
                "domain": OPERATOR_DOMAIN[op_name],
                "source": minimized,
                "shapes": shapes,
                "mutant_id": f"{corpus}:{mid}:{op_name}",
                "minimization": minimization,
            })

    mutants.sort(key=lambda d: d["mutant_id"])
    n = len(mutants)

    per_mode = {}
    for mode in MODES:
        killed = []
        survived_safe = []
        survived_unknown = []
        per_op_killed = defaultdict(int)
        per_op_total = defaultdict(int)
        per_domain_killed = defaultdict(int)
        per_domain_total = defaultdict(int)
        for mut in mutants:
            v = _verdict(mut["source"], mut["shapes"], mode)
            op = mut["operator"]
            dom = mut["domain"]
            per_op_total[op] += 1
            per_domain_total[dom] += 1
            if v == "UNSAFE":
                killed.append(mut["mutant_id"])
                per_op_killed[op] += 1
                per_domain_killed[dom] += 1
            elif v == "SAFE":
                survived_safe.append(mut["mutant_id"])
            else:
                survived_unknown.append(mut["mutant_id"])

        per_operator = {}
        for op in OPERATORS:
            tot = per_op_total.get(op, 0)
            per_operator[op] = {
                "n_genuine_bugs": tot,
                "n_killed": per_op_killed.get(op, 0),
                "kill_rate": _wilson(per_op_killed.get(op, 0), tot),
                "domain": OPERATOR_DOMAIN[op],
            }
        per_domain = {}
        for dom in sorted(set(OPERATOR_DOMAIN.values())):
            tot = per_domain_total.get(dom, 0)
            per_domain[dom] = {
                "n_genuine_bugs": tot,
                "n_killed": per_domain_killed.get(dom, 0),
                "kill_rate": _wilson(per_domain_killed.get(dom, 0), tot),
            }

        per_mode[mode] = {
            "n_killed": len(killed),
            "n_survived": len(survived_safe) + len(survived_unknown),
            "n_survived_safe": len(survived_safe),
            "n_survived_unknown": len(survived_unknown),
            "kill_rate": _wilson(len(killed), n),
            "survived_safe_ids": sorted(survived_safe),
            "survived_unknown_ids": sorted(survived_unknown),
            "per_operator": per_operator,
            "per_domain": per_domain,
        }

    minimizations = {m["mutant_id"]: m["minimization"] for m in mutants}
    n_shrunk = sum(1 for m in minimizations.values()
                   if m["removed_logical_lines"] > 0)
    total_removed = sum(m["removed_logical_lines"] for m in minimizations.values())
    minimization_by_operator = {}
    for op in OPERATORS:
        rows = [m["minimization"] for m in mutants if m["operator"] == op]
        minimization_by_operator[op] = {
            "n": len(rows),
            "n_shrunk": sum(1 for row in rows if row["removed_logical_lines"] > 0),
            "removed_logical_lines": sum(row["removed_logical_lines"] for row in rows),
            "all_same_failure_signature": all(
                row["same_failure_signature"] for row in rows
            ),
            "all_one_line_minimal": all(row["one_line_minimal"] for row in rows),
        }

    return {
        "n_clean_parents_examined": len(parents_validated),
        "n_clean_parents": len(parents_clean),
        "n_operators": len(OPERATORS),
        "operators": sorted(OPERATORS),
        "n_genuine_bug_mutants": n,
        "modes": list(MODES),
        "per_mode": per_mode,
        "minimization": {
            "algorithm": "deterministic source-line ddmin preserving PyTorch exception signature",
            "n_minimized": n,
            "n_shrunk": n_shrunk,
            "total_removed_logical_lines": total_removed,
            "all_same_failure_signature": all(
                m["same_failure_signature"] for m in minimizations.values()
            ),
            "all_one_line_minimal": all(
                m["one_line_minimal"] for m in minimizations.values()
            ),
            "per_operator": minimization_by_operator,
            "per_mutant": minimizations,
        },
        # Headline: sound mode kills every genuine bug it does not abstain on,
        # and -- crucially -- never calls a genuine bug SAFE.
        "sound_mode_zero_false_safe": per_mode["sound"]["n_survived_safe"] == 0,
        "sound_mode_kill_rate_point": per_mode["sound"]["kill_rate"]["point"],
    }


def render_markdown(data: dict) -> str:
    sm = data["per_mode"]["sound"]
    lines = [
        "# Mutation testing of clean models",
        "",
        f"We inject single, local bugs into the **{data['n_clean_parents']}** "
        "validated-clean models of the stress and natural corpora using "
        f"**{data['n_operators']}** mutation operators (Linear/Conv width bumps "
        "and an integer-dtype cast), keep only the "
        f"**{data['n_genuine_bug_mutants']}** mutants that genuinely raise under "
        "eager PyTorch, automatically minimize each admitted mutant while "
        "preserving its real PyTorch exception signature, and measure how many "
        "the verifier kills (returns UNSAFE).",
        "",
        "| mode | killed | survived (SAFE) | survived (UNKNOWN) | kill rate [95% CI] |",
        "| --- | --- | --- | --- | --- |",
    ]
    for mode in data["modes"]:
        d = data["per_mode"][mode]
        kr = d["kill_rate"]
        lines.append(
            f"| {mode} | {d['n_killed']} | {d['n_survived_safe']} | "
            f"{d['n_survived_unknown']} | "
            f"{kr['point']} [{kr['low']}, {kr['high']}] |"
        )
    lines += [
        "",
        "Automatic minimization (all admitted mutants):",
        "",
        "| minimized | shrunk | removed logical lines | preserves failure signature | 1-line minimal |",
        "| --- | --- | --- | --- | --- |",
    ]
    mz = data["minimization"]
    lines.append(
        f"| {mz['n_minimized']} | {mz['n_shrunk']} | "
        f"{mz['total_removed_logical_lines']} | "
        f"{mz['all_same_failure_signature']} | {mz['all_one_line_minimal']} |"
    )
    lines += [
        "",
        "Per-operator kill rate (sound mode):",
        "",
        "| operator | domain | genuine bugs | killed | kill rate [95% CI] |",
        "| --- | --- | --- | --- | --- |",
    ]
    for op in sorted(sm["per_operator"]):
        d = sm["per_operator"][op]
        kr = d["kill_rate"]
        lines.append(
            f"| {op} | {d['domain']} | {d['n_genuine_bugs']} | {d['n_killed']} | "
            f"{kr['point']} [{kr['low']}, {kr['high']}] |"
        )
    lines += [
        "",
        f"- sound mode never calls a genuine bug SAFE: "
        f"**{data['sound_mode_zero_false_safe']}**",
        f"- sound-mode kill rate (point): **{data['sound_mode_kill_rate_point']}**",
        "",
    ]
    if sm["survived_safe_ids"]:
        lines += [
            "Survivors reported SAFE in sound mode (missed bugs):",
            "",
        ] + [f"- `{i}`" for i in sm["survived_safe_ids"]] + [""]
    else:
        lines += [
            "No genuine-bug mutant is reported SAFE in sound mode: every injected "
            "bug is either killed (UNSAFE) or explicitly abstained (UNKNOWN), so "
            "the verifier never silently passes an injected bug.",
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
            print("MISMATCH: mutation_clean_models artifacts differ")
            return 1
        print("OK: mutation_clean_models artifacts byte-identical")
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
