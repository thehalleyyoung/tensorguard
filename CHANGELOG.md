# Changelog

All notable changes to TensorGuard are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and TensorGuard
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
stability guarantees for the public API, the integration modules, and the CLI
subcommands are defined in [`DEPRECATION_POLICY.md`](DEPRECATION_POLICY.md) and
enforced by `tests/test_api_stability.py`.

## [Unreleased]

### Added
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
