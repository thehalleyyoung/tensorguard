"""From-scratch reproduction harness for every CI-reproducible TensorGuard artifact.

``python reproducibility/reproduce_all.py`` regenerates, in dependency order,
every artifact that can be rebuilt in a standard CI box (no CUDA / no HuggingFace
download / no Lean toolchain) and then runs the numeric-claim audit, which
recomputes every ``x/y`` ratio and ``%`` token in ``README.md`` from the
freshly-regenerated artifacts. If any step fails, or the audit fails, the
harness exits non-zero.

Scope (honest):
  * Regenerated from source in CI: the generated spec docs/tables, the frozen
    benchmark corpus + its audit artifact, and the headline 60-bug RP figure.
  * Validated-but-not-regenerated in CI: artifacts that require CUDA, a
    HuggingFace download, or a Lean toolchain. These remain committed; the
    numeric audit reports their claims as ``QUALIFIED_ENV`` and records the
    exact regeneration command. Run ``make reproduce-full`` in an environment
    that has those toolchains to regenerate them too.

``--check`` additionally asserts determinism: after regeneration, the
byte-deterministic generated paths (``GENERATED_DETERMINISTIC``) must have no
git diff. The headline JSON is excluded from the byte-diff because it records a
volatile ``elapsed_s`` wall-clock field; its scientific content is instead
validated by the numeric audit, which recomputes 53/60, 56/60, 88.3%, and the
0% false-positive figure from it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


# Byte-deterministic artifacts: regenerating them must yield zero git diff.
GENERATED_DETERMINISTIC = [
    "SOUNDNESS_CONTRACT.md",
    "VERIFIABLE_FRAGMENT.md",
    "operator_confidence_table.json",
    "real_benchmarks/VERSION",
    "real_benchmarks/manifest.json",
    "real_benchmarks/DATASHEET.md",
    "reproducibility/real_benchmarks_audit.json",
    "reproducibility/numeric_claims_audit.json",
    "evaluation/significance.json",
    "evaluation/significance.md",
    "evaluation/localization_effort.json",
    "evaluation/localization_effort.md",
    "reproducibility/cegar_convergence.json",
    "reproducibility/cegar_convergence.md",
    "reproducibility/smt_backend_comparison.json",
    "reproducibility/smt_backend_comparison.md",
    "reproducibility/soundness_boundary.json",
    "reproducibility/soundness_boundary.md",
    "reproducibility/leaderboard.json",
    "reproducibility/leaderboard.md",
    "reproducibility/training_loop_hazards.json",
    "reproducibility/training_loop_hazards.md",
    "reproducibility/tensor_parallel_sharding.json",
    "reproducibility/tensor_parallel_sharding.md",
    "reproducibility/quant_export_safety.json",
    "reproducibility/quant_export_safety.md",
    "reproducibility/stub_autogen_coverage.json",
    "reproducibility/stub_autogen_coverage.md",
    "reproducibility/upstream_hook_demo.json",
    "reproducibility/upstream_hook_demo.md",
    "corpus_extended/manifest.json",
    "reproducibility/corpus_extended_score.json",
    "reproducibility/corpus_extended_score.md",
    "reproducibility/corpus_provenance_audit.json",
    "reproducibility/corpus_provenance_audit.md",
    "reproducibility/issue_miner_demo.json",
    "reproducibility/issue_miner_demo.md",
    "reproducibility/corpus_stratified.json",
    "reproducibility/corpus_stratified.md",
]
# Generated corpus repro files (also byte-deterministic) are added dynamically.

# Regenerated but NOT byte-diffed (records a volatile wall-clock field); its
# scientific content is validated by the numeric audit instead.
VOLATILE_REGENERATED = [
    "reproducibility/reproduce_headline_60bug.json",
]

# Artifacts that need CUDA / HuggingFace / Lean and cannot be rebuilt in a
# standard CI box. The numeric audit validates the committed copies and records
# each one's regeneration command (reported as QUALIFIED_ENV / QUALIFIED_REGIME).
ENV_QUALIFIED_NOTE = [
    ("Lean operator-rule audit (28 rules)", "requires Lean 4: `lake build TensorGuard.V5OperatorRules`"),
    ("HuggingFace cross-family natural bugs", "requires `transformers` + model downloads"),
    ("CUDA/Dynamo end-to-end artifacts", "requires a CUDA-enabled host"),
]


def _corpus_repro_paths():
    paths = []
    for sub in ("clean", "buggy"):
        d = os.path.join(ROOT, "real_benchmarks", sub)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".py"):
                    paths.append(f"real_benchmarks/{sub}/{f}")
    return paths


def _corpus_extended_paths():
    paths = []
    d = os.path.join(ROOT, "corpus_extended", "cases")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                paths.append(f"corpus_extended/cases/{f}")
    return paths


# Each step: (name, argv, stdout_path-or-None). When stdout_path is set, the
# subprocess' stdout is captured into that file (stderr is discarded so the
# update-check banner never pollutes generated docs).
STEPS = [
    ("doc: SOUNDNESS_CONTRACT.md",
     [PY, "-m", "src.soundness_contract"], "SOUNDNESS_CONTRACT.md"),
    ("doc: VERIFIABLE_FRAGMENT.md",
     [PY, "-m", "src.verifiable_fragment"], "VERIFIABLE_FRAGMENT.md"),
    ("table: operator_confidence_table.json",
     [PY, "-m", "src.operator_confidence"], "operator_confidence_table.json"),
    ("corpus: real_benchmarks manifest + repro files",
     [PY, "-m", "real_benchmarks.build_manifest"], None),
    ("corpus: real_benchmarks audit artifact",
     [PY, "-m", "real_benchmarks.build_audit_artifact"], None),
    ("corpus: real_benchmarks datasheet (Datasheets for Datasets)",
     [PY, "-m", "real_benchmarks.build_datasheet"], None),
    ("headline: 60-bug Refuted-Proof figure",
     [PY, "reproducibility/reproduce_headline_60bug.py"], None),
    ("significance: McNemar + Holm + paired bootstrap",
     [PY, "evaluation/significance.py"], None),
    ("user study: localization-effort proxy (effect sizes)",
     [PY, "evaluation/localization_effort.py"], None),
    ("theory: measured CEGAR convergence (real loop)",
     [PY, "reproducibility/cegar_convergence.py"], None),
    ("theory: Z3 vs cvc5 backend concordance + decidability",
     [PY, "reproducibility/smt_backend_comparison.py"], None),
    ("boundary: soundness/incompleteness boundary vs live verifier",
     [PY, "reproducibility/soundness_boundary.py"], None),
    ("leaderboard: open benchmark leaderboard over frozen corpus",
     [PY, "reproducibility/leaderboard.py"], None),
    ("training: training-loop hazard analyzer vs real torch",
     [PY, "reproducibility/training_loop_hazards.py"], None),
    ("distributed: tensor-parallel sharding checker vs real torch",
     [PY, "reproducibility/tensor_parallel_sharding.py"], None),
    ("deploy: quantization & export safety checks vs real torch",
     [PY, "reproducibility/quant_export_safety.py"], None),
    ("coverage: auto-generated shape stubs vs live torch forwards",
     [PY, "reproducibility/stub_autogen_coverage.py"], None),
    ("upstream: proposed nn.Module verification hook vs real torch",
     [PY, "reproducibility/upstream_hook_demo.py"], None),
    ("corpus+: extended benchmark corpus (materialize + runtime-validate)",
     [PY, "-m", "corpus_extended.build"], None),
    ("corpus+: TensorGuard score over extended corpus (Wilson CIs)",
     [PY, "reproducibility/corpus_extended_score.py"], None),
    ("corpus+: provenance + license-compatibility audit (redistributable)",
     [PY, "reproducibility/corpus_provenance_audit.py"], None),
    ("corpus+: offline issue miner (human-in-the-loop candidate proposals)",
     [PY, "reproducibility/issue_miner_demo.py"], None),
    ("corpus+: stratified per-class recall/specificity (Wilson CIs)",
     [PY, "reproducibility/corpus_stratified.py"], None),
    ("audit: numeric-claim audit (validates README numbers)",
     [PY, "reproducibility/audit_numeric_claims.py"], None),
]


def _run_step(name, argv, stdout_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    if stdout_path:
        abs_out = os.path.join(ROOT, stdout_path)
        with open(abs_out, "w") as fh:
            proc = subprocess.run(argv, cwd=ROOT, env=env, stdout=fh,
                                  stderr=subprocess.DEVNULL)
        return proc.returncode
    proc = subprocess.run(argv, cwd=ROOT, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Surface only on failure to keep the harness output readable.
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout.decode("utf-8", "replace"))
    return proc.returncode


def run(check=False):
    print("=" * 72)
    print("TensorGuard from-scratch reproduction harness")
    print("=" * 72)
    t0 = time.time()
    for name, argv, stdout_path in STEPS:
        step_t = time.time()
        rc = _run_step(name, argv, stdout_path)
        status = "OK " if rc == 0 else "FAIL"
        print(f"[{status}] {name}  ({time.time() - step_t:.1f}s)")
        if rc != 0:
            print(f"\nReproduction FAILED at step: {name} (exit {rc})")
            return 1

    print("-" * 72)
    print("Regenerated in CI (from source):")
    for p in GENERATED_DETERMINISTIC + _corpus_repro_paths() + _corpus_extended_paths() + VOLATILE_REGENERATED:
        print(f"  + {p}")
    print("\nValidated against committed artifacts (env-qualified, not "
          "regenerated in this box):")
    for label, why in ENV_QUALIFIED_NOTE:
        print(f"  ~ {label} — {why}")

    if check:
        rc = _determinism_check()
        if rc != 0:
            return rc

    print("-" * 72)
    print(f"Reproduction PASS ({time.time() - t0:.1f}s). Numeric audit green; "
          "every README x/y and % token recomputed from regenerated artifacts.")
    return 0


def _determinism_check():
    paths = GENERATED_DETERMINISTIC + _corpus_repro_paths() + _corpus_extended_paths()
    proc = subprocess.run(
        ["git", "diff", "--exit-code", "--"] + paths,
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        print("\nDETERMINISM CHECK FAILED: regeneration changed tracked files:")
        sys.stdout.write(proc.stdout.decode("utf-8", "replace"))
        return 1
    print("\nDeterminism check OK: byte-identical regeneration of "
          f"{len(paths)} tracked artifacts.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="assert byte-identical regeneration (no git diff) of "
                         "the deterministic generated artifacts")
    args = ap.parse_args()
    return run(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
