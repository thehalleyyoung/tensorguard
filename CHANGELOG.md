# Changelog

All notable changes to TensorGuard are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and TensorGuard
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
stability guarantees for the public API, the integration modules, and the CLI
subcommands are defined in [`DEPRECATION_POLICY.md`](DEPRECATION_POLICY.md) and
enforced by `tests/test_api_stability.py`.

## [Unreleased]

### Added
- **Data plane (merged from DataRefine).** TensorGuard now analyzes the
  deep-learning *data* layer — *which numbers reach the model and what they mean*
  — alongside its native model-plane (shape/dtype) verification. New self-contained
  subpackage `src/dataplane/` (an 18-module closure; no new dependency — `z3-solver`
  is already core) built around an **abstract-interpretation engine**: a refinement
  product lattice (value-domain × information-flow × split-origin × role ×
  provenance) with operator transfer functions that infers a typed contract for
  every data value and emits z3-discharged obligations at sinks across **seven bug
  axes**. New public entry points (lazily imported): `analyze_data_plane`,
  `analyze_data_plane_file`, `analyze_data_plane_tree`, plus seven additive
  `BugCategory` members (`DATA_VALUE_DOMAIN`, `DATA_LEAKAGE`,
  `DATA_TEMPORAL_LEAKAGE`, `DATA_GROUP_LEAKAGE`, `DATA_JOIN_CARDINALITY`,
  `DATA_SAMPLING_DETERMINISM`, `DATA_SPLIT_CONTRACT`). Data-plane findings lower
  into first-class `Bug` objects, so one run can surface a shape mismatch *and* a
  data-leakage violation side by side.
  - **New axes (5):** loss applied outside its required value domain (e.g. `BCELoss`
    on logits), temporal lookahead (`shift(-k)`, centered `rolling`), group leakage
    (a group straddling the train/test split), join-cardinality fan-out before a
    split, and overlapping/ill-formed split contracts. These are outside any prior
    analyzer's vocabulary.
  - **Generalises the earlier `src/interface_layer/torch_data_misuse.py`
    (PromptABI) slice:** its worker-RNG-duplication and fit-before-split-leakage
    checks are re-expressed here as obligations over the refinement lattice (so they
    compose with the five new axes and export as proof packets). `torch_data_misuse`
    remains the home of its unique drop-last-on-eval check and the `scan-torch-data`
    CLI; prefer `analyze_data_plane` for the unified seven-axis sweep.
  - **Kept separate by design:** the data-plane structural SMT certifier
    (`src/dataplane/certification`, `smt_backend`) is distinct from `src/smt` because
    the two encode different theories (shape/broadcast algebra vs. value domains, set
    disjointness, index causality, cardinality) — each the cheapest sound mechanism
    for its own question.

- **Interface layer (merged from PromptABI).** TensorGuard now verifies the
  *discrete text/token interface* of LLM apps alongside its tensor plane. New
  subpackage `src/interface_layer/` (a self-contained 15-module closure; no new
  dependency — `z3-solver` is already core) with four new CLI provers:
  `scan-torch-data` (silent PyTorch data-pipeline bugs: worker-RNG duplication,
  `drop_last` on eval loaders, fit-before-split leakage — complements
  `training_loop_checks`), `prove-surface-ban` (does an id-level
  `bad_words`/`suppress_tokens` ban prevent a forbidden surface? product-automaton,
  unbounded), `prove-streaming-stop` (can a server truncate exactly at a stop
  string? overshoot/split-stop, unbounded), and `prove-decoding-feasibility`
  (does a guided-decoding grammar admit a tokenization? greatest-fixpoint +
  bounded SMT). The template/tokenizer forgery analyzers
  (`smt_tokenizer_forgery`, `role_separator_forgery`, `control_token_injection`,
  `normalization_confusables`) are available as a library. Redundant areas
  (shape/loss/training-numeric/axis-role checks) were deliberately not duplicated;
  see `docs/MERGE_PROMPTABI.md`. Tests: `tests/interface_layer/` (39).
- **Importable top-level package.** `import tensorguard` now resolves to a real
  public surface that re-exports the stability-guaranteed API
  (`verify_architecture`, `analyze`, `AnalysisResult`, …) and the integration
  entry points (`guarded_compile`, `make_tensorguard_backend`, `verify_module`,
  `TensorGuardViolation`). Proven by building the wheel and importing it from a
  clean install (`tests/test_pip_install.py`).
- **Changed-files mode for the GitHub Action.** The Action can verify only the
  models changed in a pull request (`changed-only: "true"`, with configurable
  `base-ref`/`head-ref`), using `git diff --diff-filter=ACMRT`. Deleted and
  non-Python files are ignored; a missing base ref falls back to a full scan.
- **`pytest --tensorguard` verifies the modules under test.** The plugin now
  discovers the project modules actually imported by the test session and
  verifies those, not just the rootdir — so the gate covers the code the suite
  exercises (`--tensorguard-under-test`, default on).
- **Language Server + VS Code extension.** A real LSP server
  (`python -m src.lsp_server`, JSON-RPC over stdio) publishes shape/device/phase
  diagnostics, hover shapes, and quick-fixes from unsaved editor buffers; a VS
  Code extension scaffold under `editors/vscode/` wires it as a language client.
- **Verified Dynamo backend.** `make_tensorguard_backend` / `guarded_compile`
  are proven end-to-end against a real `torch.compile` run on torch 2.x.
- **Hugging Face `from_pretrained` gate.** `guarded_from_pretrained` loads a
  checkpoint and verifies the returned `PreTrainedModel` before handing it back,
  so a misbuilt model never escapes the loader; a gated clean model then trains
  under a real `transformers.Trainer` (`src/integrations/hf_hook.py`).
- **Lightning adoption walkthrough.** Runnable
  `examples/lightning_guarded_training.py` guards a real CNN `Trainer.fit`;
  the buggy variant is blocked at fit-start before any optimizer step.
- **ONNX export gate hardening.** `guarded_onnx_export` now runs an
  `onnx.checker.check_model` post-export assertion (default on) and documents the
  opt-in `dynamo=True` exporter path.
- **Export / AOTInductor packaging gates.** `verify_exported_program`
  (verify-before-`torch.export.export`, shapes inferred from example args) and
  `guarded_aot_package` (verify-before-`aoti_compile_and_package`) bring the
  export/packaging paths to parity with the ONNX gate.
- **Public-leaderboard CI.** `reproducibility/validate_entry.py` +
  `.github/workflows/leaderboard.yml` validate external tool submissions and
  re-score the board from raw per-case verdicts (self-reported metrics rejected).
- **Proposed-upstream shim.** `src.upstream_hook.install()` grafts the proposed
  `torch.nn.utils.verify_module` / `attach_verifier` / `torch.nn.verifiable`
  surface onto the real namespace with no core changes (idempotent, reversible).

## [0.1.0]

### Added
- First public release: sound static verification of PyTorch `nn.Module`s for
  shape, broadcast, device, dtype, phase, and gradient bugs via guard-harvesting
  abstract interpretation and an SMT (Z3) backend.
- CLI (`tensorguard`), pre-commit hook, GitHub Action, SARIF/JUnit/JSON
  reporters, baseline-based incremental adoption, and a PEP 561 `py.typed`
  marker.
- Machine-checked soundness fragment in Lean (sorry-free axiom audit).

[Unreleased]: https://github.com/thehalleyyoung/tensorguard/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/thehalleyyoung/tensorguard/releases/tag/v0.1.0
