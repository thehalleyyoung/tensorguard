# 100 Steps to Ship TensorGuard With PyTorch

**North star:** TensorGuard becomes good enough that the PyTorch project would
ship it (or bless it) as the default static safety net for `nn.Module`
architectures — catching shape, device, dtype, and phase bugs before a single
forward pass runs, with zero false positives on real code and negligible
friction.

To get there we must close the gap between today's research prototype and a
production-grade, sound, broadly-compatible, well-governed tool. The steps
below are ordered roughly by dependency, grouped into ten phases. Each step is
a concrete, checkable unit of work.

---

## Phase 1 — Honesty & ground truth (close the prototype→product credibility gap)

The README already lists places where the shipped behaviour is narrower than
the headline. Shipping with PyTorch is impossible until every claim is true,
measured, and reproducible.

1. [x] Promote CEGAR-discovered unsatisfiable predicates from metadata into real
   `Bug(category=cegar_refined_contract)` objects so `--cegar-iterations N`
   actually changes results (closes the documented "metadata only" gap).
   *Done: `src/api.py` `_cegar_refined_contract_bugs()` detects jointly
   unsatisfiable refined contracts (union of `iteration_log` predicates,
   Z3-discharged via `ShapeRefinement.check_feasibility`) and emits real Bugs;
   proven on a 768-vs-512 in_features conflict where CEGAR=0 → 0 bugs and
   CEGAR=10 → 1 sound bug. Regression tests in
   `tests/test_cegar_refined_contract.py`.*
2. [x] Make `--no-phase-check`/`--no-device-check`/`--no-grad-check` gate the
   *solver*, not just post-hoc filter the verdict, so timing and witnesses
   reflect the real per-domain cost.
   *Done: threaded `check_devices/phases/gradients` into `ConstraintVerifier`;
   the per-step encoders `_encode_{device,phase,gradient}_safety` return no
   constraints when their domain is disabled, and the device/phase theory
   solver checks are skipped. Proven that disabled-domain encoders emit zero
   constraints (no solver work / witnesses) while enabled ones do; post-hoc
   filter kept only as a defensive net. Tests in
   `tests/test_solver_domain_gating.py`.*
3. [x] Replace the flat `feature_ablation.json` line with a corpus where each
   domain (Shape/Device/Phase/Stride/Permutation) demonstrably contributes at
   least one bug the L1 shape view misses; if a domain never contributes,
   document it as "diagnostic-only" rather than "verification".
   *Done: built a curated per-domain corpus (`experiments_v5/domain_corpus/`
   + `domain_corpus_manifest.json`) and a marginal-contribution runner
   (`experiments_v5/run_domain_contribution.py` → `domain_contribution.json`).
   Real-code result: shape (base) refutes `shape_01`; **device** adds 2 bugs
   (`device_01`, `device_02`) the shape view misses; **gradient** adds 2 bugs
   (`grad_01`, `grad_02`); **phase** adds 0 → classified diagnostic-only.
   Stride/permutation are sub-aspects of the shape domain (no separate solver
   encoder), so they are covered by the shape view. The old flat
   `feature_ablation.json` ladder is superseded (notes added in its runner).
   Pinned by `tests/test_domain_contribution.py` (11 tests).*
4. [x] Audit every numeric claim in `README.md`, `neurips.tex`, and the workshop
   paper against a single regeneration script; delete or qualify any number
   that cannot be rerun from the current tree.
   *Done: `reproducibility/audit_numeric_claims.py` is a single harness with a
   registry of every headline numeric claim. For each it (a) checks the number
   is still literally present at its cited source and (b) recomputes it from the
   committed regeneration artifact(s), supporting ratios, percentages, and
   p-values with tolerance. Classifies VERIFIED / MISMATCH / QUALIFIED_REGIME /
   QUALIFIED_ENV / ORPHAN / SOURCE_MISSING and a README scanner fails on any
   uncovered `x/y` or `%` token (script-catalogue rows exempt). Live result:
   10 VERIFIED, 1 QUALIFIED_REGIME (Pytea 25/34 fragment-fair vs. 22/34 stricter
   regime — pinned to the correct artifact so it is not a false mismatch),
   1 QUALIFIED_ENV (Lean 28 rules). Resolved that the paper's `9/9` HF claim is
   the union of two artifacts (7 + Gemma-2 2) and that `25/34`/`32/34`/McNemar
   `p=0.0156` come from `pytea_fragment_fair.json`. Honest scope documented:
   audits committed artifacts (+ optional `--regenerate`), not a from-scratch
   rerun. Output `reproducibility/numeric_claims_audit.json`; pinned by
   `tests/test_numeric_claims_audit.py` (6 tests, incl. negative MISMATCH /
   ORPHAN / SOURCE_MISSING cases).*
5. [x] Define and publish a precise **soundness contract**: exactly which programs
   TensorGuard guarantees to never miss-pass, and which constructs are
   over-approximated, under-approximated, or skipped.
   *Done: `src/soundness_contract.py` is the single importable source of truth.
   It states both directions explicitly — Refutation soundness (no false alarm,
   Z3-discharged) and Verification soundness (never miss-pass, scoped to the
   verifiable fragment + modeled bug classes + SOUND operators) — and
   classifies every construct as `sound` / `over_approximated` /
   `under_approximated` / `skipped`, with a clause per `UnsupportedCategory`.
   Crucially it surfaces (not hides) the known unsoundness gaps: U1 —
   `verify_architecture` does not yet gate on the fragment, so an out-of-fragment
   module currently gets a silent SAFE (fix = Step 8); U2 — `shape_cegar`
   SAFE-on-infeasible. Published to `SOUNDNESS_CONTRACT.md` (generated;
   referenced from README). Empirical claims pinned against real code in
   `tests/test_soundness_contract.py` (7 tests): fragment boundary
   (clean in / data-dependent-CF out), the U1 silent-SAFE gap, a refutation
   probe, and doc/module sync.*
6. [x] Tag every operator transfer function as `sound`, `complete`, or `heuristic`
   in a machine-readable table; surface the tag in output so users know the
   confidence of each inference.
   *Done. `src/operator_confidence.py` is the single source of truth: a
   `ConfidenceTag` enum (COMPLETE/SOUND/HEURISTIC) + principled classifier
   covering all 117 registered transfer functions (75 complete, 36 sound,
   6 heuristic). complete = shape-preserving pointwise (activations,
   elementwise unary, comparisons) → exact (sound & complete); sound =
   exact structural rules enforced soundly (matmul family, reductions,
   gather/scatter, sort/topk, FFTs, static-shape sampling); heuristic =
   data-dependent output or approximated rule (unique, multinomial, einsum,
   linalg.*). Unknown/unregistered ops default to `heuristic` (honest, never
   over-claims). Surfaced three ways: machine-readable `operator_confidence_table.json`
   (committed, sync-tested), a new `tensorguard operator-confidence [ops...] [--json]`
   CLI command, and `annotate_registry()` which stamps `TransferFunction.confidence`
   so the tag travels with each transfer function. Pinned by
   `tests/test_operator_confidence.py` (10 tests: full-registry coverage,
   defensible spot-checks, heuristic default, table↔code sync, CLI).*
7. Add a `--soundness-mode {sound,balanced,heuristic}` flag and make `sound`
   the contract that PyTorch could rely on (no false negatives within the
   declared fragment).
8. Write `verifiable_fragment.py` into a formal spec doc: grammar of supported
   `nn.Module`/`forward` constructs, with an explicit "unsupported → reported
   as `unknown`" fallback rather than silent pass.
9. Establish a frozen, versioned **ground-truth bug corpus** (real models,
   labeled buggy/clean) checked into `real_benchmarks/` with provenance.
10. Stand up a reproducibility harness (`make reproduce`) that regenerates
    every paper table and README number from scratch in CI.

## Phase 2 — Correctness core: precision and recall on real PyTorch

11. Mine 500+ real shape/device bugs from public GitHub history (CI failures,
    "RuntimeError: size mismatch" commits) into a labeled dataset.
12. Measure precision/recall against PyTea, runtime tools (e.g. shape
    assertions), and a no-op baseline; commit the confusion matrices.
13. Drive false-positive rate to **0%** in `sound` mode on the clean half of
    the corpus — a single false positive kills trust for a shipped tool.
14. Drive recall on the buggy half above the strongest baseline; track per-bug
    misses with root-cause tags.
15. Add differential fuzzing: generate random valid `nn.Module`s, run them
    once, and assert TensorGuard never reports a bug on a model that executed
    cleanly (false-positive hunting).
16. Add negative fuzzing: inject shape/device faults into valid models and
    assert TensorGuard catches them (false-negative hunting).
17. Build a minimization tool that shrinks any disagreement (TensorGuard vs.
    runtime) to a minimal reproducing `nn.Module`.
18. Triage and fix the top 50 minimized disagreements; convert each into a
    regression test.
19. Add property-based tests (Hypothesis) over the shape-algebra transfer
    functions (e.g. `view ∘ reshape` invariants, `permute` involutions).
20. Establish a precision/recall dashboard that blocks merges on regression.

## Phase 3 — Operator & semantics coverage (match real PyTorch surface area)

21. Enumerate the full public `torch` + `torch.nn` + `torch.nn.functional`
    operator surface; build a coverage matrix vs. the 142 implemented transfer
    functions.
22. Prioritize operators by frequency in real model corpora (transformers,
    torchvision, timm) and implement the long tail in priority order.
23. Implement precise `einsum` shape inference (parse the equation, not a
    heuristic).
24. Implement `reshape`/`view`/`flatten` with symbolic product reasoning and
    `-1` inference backed by Z3 divisibility constraints.
25. Implement broadcasting semantics exactly (NumPy/PyTorch rules) including
    `expand`, `broadcast_to`, and implicit op broadcasting.
26. Implement advanced indexing, `gather`/`scatter`, `index_select`, masked
    ops, and boolean indexing shape effects.
27. Implement attention/transformer building blocks end-to-end
    (`scaled_dot_product_attention`, MHA, rotary/positional reshapes).
28. Implement convolution family precisely: stride/padding/dilation/groups,
    transposed conv, 1d/2d/3d, and `output_padding`.
29. Implement normalization layers' phase semantics (`BatchNorm`, `LayerNorm`,
    `GroupNorm`, `Dropout`) including train/eval-dependent behaviour.
30. Implement dtype inference & promotion rules (a second algebra alongside
    shape) so `float16/bfloat16/int` mismatches are caught.
31. Implement device propagation across `.to()`, `.cuda()`, `.cpu()`,
    `pin_memory`, and module-level `device` parameters.
32. Implement RNG/seed-independent reasoning so stochastic ops don't produce
    spurious "unknown".
33. Add a **conformance oracle**: every transfer function is cross-checked
    against a real `torch` execution on sampled concrete shapes in CI.
34. Auto-detect operator coverage gaps at analysis time and emit a precise
    "unsupported op: `torch.foo`" diagnostic instead of guessing.
35. Track operator coverage as a published percentage; gate releases on it.

## Phase 4 — Frontend robustness (survive real codebases)

36. Harden the `torch.fx` frontend to trace real models without crashing;
    measure trace success rate over torchvision/timm/transformers.
37. Add a `torch.export`/`dynamo` frontend and reconcile it with the fx path
    (`dynamo_gap_analysis.py` should converge to zero gaps).
38. Support dynamic control flow in `forward` (data-dependent branches, loops,
    `for` over `ModuleList`) via path-sensitive analysis.
39. Support symbolic/dynamic batch and sequence dimensions as first-class
    symbols, not concrete guesses.
40. Handle subclassing, `super().forward()`, mixins, and functional/`nn`
    style models uniformly.
41. Handle third-party layers (HuggingFace, timm custom blocks) via a
    pluggable shape-stub registry.
42. Infer input specs automatically from `forward` type hints, docstrings,
    `example_inputs`, and config files (reduce the `-s` annotation burden).
43. Gracefully degrade: when a region is unanalyzable, isolate it and continue
    verifying the rest of the model.
44. Add interprocedural analysis across helper functions and nested modules
    with a sound call-summary cache.
45. Stress-test the frontend on the top 100 most-starred PyTorch repos; fix
    every crash and record a parse-success SLA.

## Phase 5 — Performance & scale (sub-second on huge models)

46. Profile end-to-end latency on small/medium/large models; set budgets
    (e.g. <1s small, <10s for a 70B-config transformer graph).
47. Cache and reuse Z3 contexts; batch constraints; avoid solver calls when a
    fast syntactic check suffices.
48. Add incremental analysis: re-verify only modules touched by a diff
    (`src/_experimental/incremental` → productionized).
49. Add a constraint-simplification pass before Z3 (constant folding, interval
    domain pre-pass) to shrink SMT load.
50. Parallelize per-module verification across cores; make it deterministic.
51. Add memoization of operator transfer results keyed on symbolic input
    shapes.
52. Add a timeout/anytime mode that returns sound partial results under a
    budget instead of hanging.
53. Benchmark against `pip install`-time, import-time, and analysis-time costs;
    keep import time negligible for the PyTorch dependency story.
54. Add a regression benchmark suite (Criterion-style) that fails CI on >10%
    latency regressions.
55. Optimize the worst-case `reshape`/`einsum` divisibility constraints that
    blow up Z3.

## Phase 6 — Developer experience (zero-friction adoption)

56. Make `tensorguard verify <path>` work with **no flags** on the common case
    by auto-inferring input specs.
57. Produce world-class diagnostics: source-mapped, colored, with the offending
    op, the inferred vs. expected shape, and a suggested fix.
58. Add a "why" explainer (`--explain`) that prints the inference chain leading
    to a reported bug (leverage `contrastive_explanation.py`).
59. Add autofix suggestions (`--fix`) for mechanical cases (e.g. wrong
    `nn.Linear` in-features computed from upstream shapes).
60. Ship a `tensorguard watch` mode for live feedback during development.
61. Polish the VSCode extension: inline squiggles, hover shapes, quick-fixes;
    publish to the marketplace.
62. Add Jupyter/IPython integration that checks a model cell on definition.
63. Add a `@tensorguard.checked` decorator for opt-in per-module enforcement.
64. Provide a config file (`tensorguard.toml`) for per-repo rules, ignores, and
    soundness mode.
65. Write a 5-minute "Getting Started" doc and an honest "What it can't do yet"
    page; keep both tested.

## Phase 7 — CI / ecosystem integration (where it earns its stars)

66. Ship a production GitHub Action that runs TensorGuard on PRs and annotates
    diffs.
67. Finalize SARIF 2.1.0 output and verify it renders in GitHub Code Scanning
    and Advanced Security.
68. Add `pre-commit` hook, `nox`/`tox` integration, and a pytest plugin
    (`pytest --tensorguard`).
69. Publish to PyPI with reproducible wheels and a pinned `z3-solver` range;
    fix the `pyproject` URLs to the canonical repo.
70. Add a `conda-forge` recipe.
71. Provide a Docker image and a zero-install `pipx run tensorguard` path.
72. Add baseline/suppression support so teams can adopt incrementally on legacy
    repos without a wall of warnings.
73. Add JSON, JUnit-XML, and GitHub-annotation reporters alongside SARIF.
74. Integrate with `torch.compile`/`torch.export` so verification can run as an
    optional pre-pass in the compile pipeline.
75. Provide an API for framework authors (Lightning, HF Trainer) to call
    TensorGuard before training starts.

## Phase 8 — Trust, governance & PyTorch-readiness

76. Relicense/confirm license compatibility with PyTorch (BSD-3) and clear all
    vendored code (e.g. the bundled PyTea source) for redistribution.
77. Remove or quarantine `experiments_v5/_pytea_src` and any third-party trees
    from the shipped package; keep them only as dev-time references.
78. Establish semantic versioning, a deprecation policy, and a stability
    guarantee for the public API and CLI flags.
79. Write a security policy and threat model (untrusted model files are parsed —
    ensure analysis never executes arbitrary code; analyze statically, never
    `import` untrusted modules).
80. Replace any dynamic-import/`exec` model loading with a safe AST/fx-only path
    so verifying a malicious file is harmless.
81. Add a comprehensive, documented public API with type stubs and `py.typed`.
82. Reach >90% test coverage on `src/` with a coverage gate in CI.
83. Set up multi-OS, multi-Python (3.9–3.13), multi-`torch`-version CI
    (including nightly torch).
84. Write a CONTRIBUTING guide, code of conduct, issue/PR templates, and a
    maintainer rotation.
85. Draft a PyTorch RFC / governance proposal for inclusion as an official
    `torch` companion tool or `pytorch-labs` project, with a maintenance plan.

## Phase 9 — Research validation (the conference paper that backs the tool)

86. Write the tool paper targeting PLDI/OOPSLA/ISSTA: formalize the refinement
    type system and the 5-theory product domain.
87. Prove (on paper + Lean where feasible) soundness of the core transfer
    functions for the declared fragment; reconcile with `lean/`.
88. Run the empirical study: precision/recall/latency vs. PyTea and runtime
    baselines on the public corpus, with significance tests
    (`statistical_rigor.py`).
89. Submit a reproducible artifact and target an Artifact-Evaluation badge.
90. Publish the labeled real-world bug benchmark as a standalone contribution
    others can cite.
91. Run a user study: do developers fix bugs faster with TensorGuard? Report
    effect sizes.
92. Document the CEGAR predicate-discovery loop and its convergence theory
    (`cegar_convergence_theory.py`) with measured iteration counts.
93. Compare SMT backends (Z3 vs. cvc5) and report decidability/performance
    trade-offs (`decidability.py`).
94. Write up the limitations and the precise unsound/incomplete boundary as a
    first-class section reviewers can trust.
95. Open-source the benchmark leaderboard so the community can drive recall up.

## Phase 10 — Long-term: become indispensable

96. Extend beyond architecture to training-loop checks (optimizer/param device
    mismatches, AMP dtype hazards, gradient-flow breaks).
97. Add distributed/parallel checks (DDP/FSDP/tensor-parallel sharding shape
    consistency) — a pain point with no good static tool today.
98. Add quantization & export safety checks (QAT dtype/observer placement,
    ONNX export shape soundness).
99. Continuously expand operator and library coverage with a community stub
    registry; auto-generate stubs from `torch` signatures where possible.
100. Propose upstream: contribute TensorGuard-style shape annotations or a
    verification hook into PyTorch itself, so every `nn.Module` can be checked
    by default and entire classes of runtime errors disappear from the
    ecosystem.

---

### How to use this list
- Track each step as an issue/milestone; Phases 1–2 are the credibility
  prerequisites and should land before any "ship with PyTorch" conversation.
- The two hard gates for upstream adoption are **0% false positives in `sound`
  mode** (Phase 2) and **safe static analysis of untrusted models** (Phase 8,
  steps 79–80). Treat them as release blockers.
- Phases 3–7 are what convert the tool from "interesting" to "1000-star
  default", and Phase 9 is what makes it citable and defensible.
