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
    "formal_soundness_appendix.tex",
    "real_benchmarks/VERSION",
    "real_benchmarks/manifest.json",
    "real_benchmarks/DATASHEET.md",
    "reproducibility/real_benchmarks_audit.json",
    "reproducibility/numeric_claims_audit.json",
    "evaluation/significance.json",
    "evaluation/significance.md",
    "evaluation/stratified_precision_recall.json",
    "evaluation/stratified_precision_recall.md",
    "evaluation/localization_effort.json",
    "evaluation/localization_effort.md",
    "reproducibility/developer_study.json",
    "reproducibility/developer_study.md",
    "docs/user_study/task_packet.json",
    "docs/user_study/task_packet.md",
    "reproducibility/cegar_convergence.json",
    "reproducibility/cegar_convergence.md",
    "reproducibility/smt_backend_comparison.json",
    "reproducibility/smt_backend_comparison.md",
    "reproducibility/soundness_boundary.json",
    "reproducibility/soundness_boundary.md",
    "reproducibility/leaderboard.json",
    "reproducibility/leaderboard.md",
    "reproducibility/model_zoo_certification.json",
    "reproducibility/model_zoo_certification.md",
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
    "corpus_extended/blind_manifest.json",
    "reproducibility/blind_split_eval.json",
    "reproducibility/blind_split_eval.md",
    "reproducibility/cross_version_stability.json",
    "reproducibility/cross_version_stability.md",
    "reproducibility/cross_python_determinism.json",
    "reproducibility/cross_python_determinism.md",
    "reproducibility/natural_distribution_study.json",
    "reproducibility/natural_distribution_study.md",
    "reproducibility/scaling_study.json",
    "reproducibility/scaling_study.md",
    "reproducibility/baseline_head_to_head.json",
    "reproducibility/baseline_head_to_head.md",
    "reproducibility/fp_stress_eval.json",
    "reproducibility/fp_stress_eval.md",
    "reproducibility/mutation_clean_models.json",
    "reproducibility/mutation_clean_models.md",
    "reproducibility/silent_bug_benchmark.json",
    "reproducibility/silent_bug_benchmark.md",
    "reproducibility/differential_dispatcher.json",
    "reproducibility/differential_dispatcher.md",
    "reproducibility/hypothesis_module_ast.json",
    "reproducibility/hypothesis_module_ast.md",
    "docs/dashboard/data.json",
    "docs/dashboard/index.html",
    "reproducibility/time_to_detect.json",
    "reproducibility/time_to_detect.md",
    "reproducibility/domain_ablation.json",
    "reproducibility/domain_ablation.md",
    "reproducibility/reduced_product_ablation.json",
    "reproducibility/reduced_product_ablation.md",
    "reproducibility/cegar_depth_ablation.json",
    "reproducibility/cegar_depth_ablation.md",
    "reproducibility/stagewise_ablation.json",
    "reproducibility/stagewise_ablation.md",
    "reproducibility/pr_history_survival.json",
    "reproducibility/pr_history_survival.md",
    "reproducibility/statistical_power.json",
    "reproducibility/statistical_power.md",
    "reproducibility/effect_sizes.json",
    "reproducibility/effect_sizes.md",
    "reproducibility/statistical_meta_analysis.json",
    "reproducibility/statistical_meta_analysis.md",
    "evaluation/negative_controls.json",
    "evaluation/negative_controls.md",
    "reproducibility/capsule_manifest.json",
    "reproducibility/capsule_manifest.md",
    "reproducibility/artifact_badges.json",
    "reproducibility/artifact_badges.md",
    "reproducibility/threats_to_validity.json",
    "reproducibility/threats_to_validity.md",
    "reproducibility/evaluation_protocol.json",
    "reproducibility/evaluation_protocol.md",
    "reproducibility/head_to_head_step252.json",
    "reproducibility/head_to_head_step252.md",
    "reproducibility/verified_reduction_diff.json",
    "reproducibility/verified_reduction_diff.md",
    "reproducibility/paper_evidence_index.json",
    "reproducibility/paper_evidence_index.md",
    "reproducibility/reviewer_commands.json",
    "reproducibility/reviewer_commands.md",
    "reproducibility/artifact_index.json",
    "reproducibility/artifact_index.md",
    "evaluation/real_model_operator_coverage.json",
    "evaluation/real_model_operator_coverage.md",
    "evaluation/deployment_gallery.json",
    "evaluation/deployment_gallery.md",
    "evaluation/pareto_curves.json",
    "evaluation/pareto_curves.md",
    "reproducibility/artifact_package.json",
    "reproducibility/artifact_package.md",
    "reproducibility/camera_ready_paper.json",
    "reproducibility/camera_ready_paper.md",
    "docs/governance/third_party_acceptance.md",
    "docs/site/index.html",
    "docs/site/concepts/soundness.html",
    "docs/site/concepts/verifiable-fragment.html",
    "docs/site/concepts/evidence.html",
    "docs/site/operators/index.html",
    "docs/site/proof-footprint/index.html",
    "docs/site/migration/runtime-assertions.html",
    "docs/site/migration/pytea.html",
    "docs/site/site_manifest.json",
    "reproducibility/governance_acceptance_audit.json",
    "reproducibility/governance_acceptance_audit.md",
    "docs/artifact/PACKAGING.md",
]
# Generated corpus repro files (also byte-deterministic) are added dynamically.

# Regenerated but NOT byte-diffed (records a volatile wall-clock field); its
# scientific content is validated by the numeric audit instead.
VOLATILE_REGENERATED = [
    "reproducibility/reproduce_headline_60bug.json",
    "reproducibility/scaling_walltime.json",
    "reproducibility/cegar_depth_walltime.json",
    "evaluation/pareto_latency.json",
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
    d2 = os.path.join(ROOT, "corpus_extended", "blind_cases")
    if os.path.isdir(d2):
        for f in sorted(os.listdir(d2)):
            if f.endswith(".py"):
                paths.append(f"corpus_extended/blind_cases/{f}")
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
    ("doc: formal_soundness_appendix.tex",
     [PY, "-m", "src.formal_soundness_appendix"], "formal_soundness_appendix.tex"),
    ("docs: public static site",
     [PY, "reproducibility/docs_site.py"], None),
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
    ("stratification: precision/recall by family/source with Wilson gates",
     [PY, "evaluation/stratified_precision_recall.py"], None),
    ("user study: localization-effort proxy (effect sizes)",
     [PY, "evaluation/localization_effort.py"], None),
    ("user study: controlled developer-study packet (Step 259)",
     [PY, "reproducibility/developer_study.py"], None),
    ("theory: measured CEGAR convergence (real loop)",
     [PY, "reproducibility/cegar_convergence.py"], None),
    ("theory: Z3 vs cvc5 backend concordance + decidability",
     [PY, "reproducibility/smt_backend_comparison.py"], None),
    ("boundary: soundness/incompleteness boundary vs live verifier",
     [PY, "reproducibility/soundness_boundary.py"], None),
    ("leaderboard: open benchmark leaderboard over frozen corpus",
     [PY, "reproducibility/leaderboard.py"], None),
    ("model zoo: public certification queue with freshness badges",
     [PY, "reproducibility/model_zoo_certification.py"], None),
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
    ("blind: held-out blind split (materialize + runtime-validate)",
     [PY, "-m", "corpus_extended.blind_build"], None),
    ("protocol: pre-specified evaluation registry (splits/metrics/scripts)",
     [PY, "reproducibility/evaluation_protocol.py"], None),
    ("baselines: Step 252 broad same-case benchmark vs PyTea/Pyright/runtime/LLM",
     [PY, "reproducibility/head_to_head_step252.py", "--check"], None),
    ("corpus+: TensorGuard score over extended corpus (Wilson CIs)",
     [PY, "reproducibility/corpus_extended_score.py"], None),
    ("corpus+: provenance + license-compatibility audit (redistributable)",
     [PY, "reproducibility/corpus_provenance_audit.py"], None),
    ("corpus+: offline issue miner (human-in-the-loop candidate proposals)",
     [PY, "reproducibility/issue_miner_demo.py"], None),
    ("corpus+: stratified per-class recall/specificity (Wilson CIs)",
     [PY, "reproducibility/corpus_stratified.py"], None),
    ("blind: pre-registered blind-split evaluation (no overfitting)",
     [PY, "reproducibility/blind_split_eval.py"], None),
    ("robustness: cross-Python determinism proof (hash-seed invariance)",
     [PY, "reproducibility/cross_python_determinism.py"], None),
    ("robustness: Step 257 multi-axis verdict-stability matrix",
     [PY, "reproducibility/cross_version_stability.py"], None),
    ("coverage: natural-distribution study (abstention + false-alarm rate)",
     [PY, "reproducibility/natural_distribution_study.py"], None),
    ("scaling: analysis-work vs model-size regression (+ volatile wall-clock)",
     [PY, "reproducibility/scaling_study.py"], None),
    ("baselines: head-to-head vs torch.export + mypy on the same corpus",
     [PY, "reproducibility/baseline_head_to_head.py"], None),
    ("soundness: 100+ clean-model false-positive stress test",
     [PY, "reproducibility/fp_stress_eval.py"], None),
    ("mutation testing: inject bugs into clean models, measure kill rate",
     [PY, "reproducibility/mutation_clean_models.py"], None),
    ("silent bugs: non-raising semantic-failure benchmark",
     [PY, "reproducibility/silent_bug_benchmark.py"], None),
    ("differential testing: 2000 random modules vs the live torch dispatcher",
     [PY, "reproducibility/differential_dispatcher.py"], None),
    ("property-based: full-module-AST soundness sweep + shrinking to minimal",
     [PY, "reproducibility/hypothesis_module_ast.py"], None),
    ("time-to-detect: static vs first failing forward op across buggy corpus",
     [PY, "reproducibility/time_to_detect.py"], None),
    ("per-domain ablation: leave-one-domain-out recall + orthogonality matrix",
     [PY, "reproducibility/domain_ablation.py"], None),
    ("reduced-product ablation: reductions-on vs -off precision gain (CPython oracle)",
     [PY, "reproducibility/reduced_product_ablation.py"], None),
    ("CEGAR-depth ablation: refinement depth vs diagnostic precision + work",
     [PY, "reproducibility/cegar_depth_ablation.py"], None),
    ("stage-wise ablation: extraction/domains/reductions/CEGAR/stubs/proof policy",
     [PY, "reproducibility/stagewise_ablation.py"], None),
    ("PR history: fix-linked category-level survival study + CI-cost proxy",
     [PY, "reproducibility/pr_history_survival.py"], None),
    ("statistical power: exact-binomial sample-size justification per headline claim",
     [PY, "reproducibility/statistical_power.py"], None),
    ("effect sizes: paired effect sizes + Holm/BH correction per comparison",
     [PY, "reproducibility/effect_sizes.py"], None),
    ("meta-analysis: distribution-stratified suite bootstrap across corpora",
     [PY, "reproducibility/statistical_meta_analysis.py"], None),
    ("negative controls: value-domain cases where runtime checks win",
     [PY, "evaluation/negative_controls.py"], None),
    ("capsule manifest: pinned-wheel reproducibility-capsule self-description",
     [PY, "reproducibility/capsule_manifest.py"], None),
    ("artifact badges: ACM/USENIX badge-to-evidence map (existence-checked)",
     [PY, "reproducibility/artifact_badges.py"], None),
    ("threats-to-validity: generated from abstention + false-positive data",
     [PY, "reproducibility/threats_to_validity.py"], None),
    ("verified reduction diff: Python reduced-product vs Lean-extracted checker",
     [PY, "reproducibility/verified_reduction_diff.py"], None),
    ("coverage: Step 208 real-model operator coverage over torchvision/timm/HF",
     [PY, "evaluation/real_model_operator_coverage.py"], None),
    ("deploy: real-model gallery with FX/export gates",
     [PY, "evaluation/deployment_gallery.py"], None),
    ("cost/latency: Step 256 hardware-normalized Pareto curves",
     [PY, "evaluation/pareto_curves.py"], None),
    ("artifact package: Step 267 Docker/conda/source fresh-machine modes",
     [PY, "reproducibility/artifact_package.py"], None),
    ("paper-evidence index: single catalogue of every regenerable table/figure",
     [PY, "reproducibility/paper_evidence_index.py"], None),
    ("camera-ready paper: generated claim ledger and page gate",
     [PY, "reproducibility/camera_ready_paper.py"], None),
    ("governance: third-party acceptance policy + review-gate audit",
     [PY, "reproducibility/governance_acceptance.py"], None),
    ("reviewer commands: one-command scripts for every main table/figure",
     [PY, "reproducibility/reviewer_commands.py"], None),
    ("artifact index: SHA-256 ledger for generated artifacts",
     [PY, "reproducibility/artifact_index.py"], None),
    ("dashboard: rebuild the static evidence site from committed artifacts",
     [PY, "reproducibility/build_dashboard.py"], None),
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
