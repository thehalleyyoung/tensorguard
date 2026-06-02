# Versioning, Deprecation & Stability Policy

TensorGuard follows [Semantic Versioning 2.0.0](https://semver.org/)
(`MAJOR.MINOR.PATCH`). The version is declared once in `pyproject.toml` and
mirrored by `src.__version__`.

## What "public API" means

The supported, stability-guaranteed surface is:

* The symbols exported from `src/__init__.py` (`analyze`, `analyze_file`,
  `analyze_directory`, `analyze_function`, `quick_check`, `AnalysisResult`,
  `Bug`, `BugCategory`, `SourceLocation`, `checked`, `__version__`).
* `src.api.verify_architecture` and its documented keyword arguments
  (`source`, `input_shapes`, `check_devices`, `check_phases`, `check_gradients`,
  `filename`, `soundness_mode`, `infer_inputs`).
* The ecosystem entry points added in Phase 7: `src.torch_integration`
  (`verify_module`, `guarded_compile`, `make_tensorguard_backend`,
  `TensorGuardViolation`), `src.framework_hooks` (`TensorGuardCallback`,
  `TensorGuardTrainerCallback`, `verify_before_training`), `src.reporters`
  (`render`, `write_report`, `to_json`, `to_junit_xml`), and `src.baseline`.
* The CLI subcommands: `analyze`, `analyze-package`, `verify`, `watch`,
  `ci-check`, `init`, `report`, `export`, `diff`, `server`, `version`, `config`,
  `operator-confidence`.

Anything under `src/_experimental/`, `src/v5/`, or names prefixed with `_` is
**not** part of the public API and may change without notice.

## Compatibility guarantee

* **PATCH** (`0.1.0 → 0.1.1`): bug fixes only; fully backward compatible.
* **MINOR** (`0.1.x → 0.2.0`): new features; backward compatible. Because the
  project is pre-1.0, the MINOR component is treated as the breaking-change axis
  per the common SemVer pre-1.0 convention, and breaking changes are confined to
  a MINOR bump until 1.0.
* **MAJOR** (`0.x → 1.0`): may contain breaking changes, each documented in the
  changelog and preceded by a deprecation cycle.

## Deprecation process

Before any public symbol or CLI flag is removed:

1. It is marked with `src.deprecation.deprecated` (or `deprecated_alias` /
   `warn_deprecated_parameter`), emitting a `DeprecationWarning` that names the
   replacement and the version in which it will be removed.
2. The deprecation ships for **at least one MINOR release** before removal.
3. The deprecation and its removal target are recorded in the changelog.

`tests/test_api_stability.py` pins the public surface so an accidental break
fails CI, and `tests/test_deprecation.py` proves the deprecation helpers emit the
correct warnings.
