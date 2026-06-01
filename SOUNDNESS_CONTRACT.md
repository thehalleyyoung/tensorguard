# TensorGuard Soundness Contract

> Generated from `src/soundness_contract.py` — the single source of truth. Do not edit by hand; run `python -m src.soundness_contract > SOUNDNESS_CONTRACT.md` and it is pinned by `tests/test_soundness_contract.py`.

## The guarantee

For a module M that lies inside the verifiable fragment V_TG (src/verifiable_fragment.py: FX-traceable, only supported layers / functions / methods, no out-of-fragment constructs) and is analysed with input shapes provided, TensorGuard guarantees:
  (R) Refutation soundness: every reported shape/device/gradient Refuted-Proof is Z3-discharged and corresponds to a real conflict that makes M unrunnable / mistrained (no false alarm).
  (V) Verification soundness (modeled scope): if M is reported SAFE for an enabled domain, then over the SOUND operator transfer functions and the modeled bug classes, no violating execution exists. This guarantee does NOT extend to UNDER_APPROXIMATED operators/bug-classes, nor to modules outside V_TG (see SKIPPED + KNOWN_UNSOUNDNESS).

## Domains

| Construct | Class | Direction | Rationale | Evidence |
|-----------|-------|-----------|-----------|----------|
| Shape domain (refutation) | `sound` | refutation (no false alarm) | Shape conflicts are encoded as Z3 constraints and a bug is emitted only when the solver proves unsatisfiability. | `src/model_checker.py:_encode_shape_safety + _z3_check_safety` |
| Shape domain (verification, in-fragment, shapes given) | `sound` | verification (never miss-pass) | A SAFE verdict means Z3 found no shape-violating model over the modeled (sound) shape transfer functions. | `src/model_checker.py ConstraintVerifier.verify` |
| Device domain (requires check_devices) | `sound` | refutation (no false alarm) | Device-mismatch refutations are Z3-discharged; contributes real bugs the shape view misses. | `experiments_v5/run_domain_contribution.py; tests/test_domain_contribution.py` |
| Gradient domain (requires check_gradients) | `sound` | refutation (no false alarm) | Gradient-flow refutations (e.g. detach on the trainable path) are Z3-discharged. | `src/model_checker.py:_encode_gradient_safety; tests/test_domain_contribution.py` |
| Phase domain (train/eval) | `over_approximated` | verification (never miss-pass) | DIAGNOSTIC-ONLY: registers well-formedness constraints for BatchNorm/Dropout but does not refute, so it never produces a false alarm and never claims to verify a phase property. | `experiments_v5/domain_corpus/phase_01_batchnorm_dropout.py; tests/test_domain_contribution.py::test_phase_domain_is_diagnostic_only` |

## Bug classes outside the 'never miss-pass' guarantee (UNDER_APPROXIMATED)

| Construct | Class | Rationale | Evidence |
|-----------|-------|-----------|----------|
| Value/data-dependent shape bugs (shape depends on tensor *values*, not just declared shapes) | `under_approximated` | TensorGuard reasons about shapes symbolically, not values; a bug that only manifests for particular runtime values may be missed. | `documented silent-miss rows in reproducibility/reproduce_headline_60bug.json (silent_miss)` |
| Numerical / dtype-precision bugs (overflow, NaN, precision loss) that do not change shapes or devices | `under_approximated` | Out of scope: the modeled domains are shape/device/phase/gradient, not numerical value semantics. | `src/api.py BugCategory enum (no numeric-value domain)` |
| Operators with heuristic transfer functions | `under_approximated` | Operators whose transfer function is tagged heuristic (vs. sound) may admit a violating execution that the model does not capture. The per-operator tag is the subject of Step 6. | `100_STEPS.md Step 6 (machine-readable sound/complete/heuristic operator table)` |

## Out-of-fragment constructs (SKIPPED)

These are detected by `check_traceability` (`in_verifiable_fragment=False`).

| Construct | Class | Note |
|-----------|-------|------|
| Out-of-fragment: DATA_DEPENDENT_CONTROL_FLOW (branch taken depends on tensor values (e.g. `if x.sum() > 0:`)) | `skipped` | Outside the verifiable fragment. `check_traceability` detects it (in_verifiable_fragment=False). NOTE: the current `verify_architecture` does not yet gate on the fragment, so such a module may receive a silent SAFE (see KNOWN_UNSOUNDNESS U1); Step 8 makes this an explicit `unknown`/abstain. |
| Out-of-fragment: DATA_DEPENDENT_ITERATION (loop bound depends on a tensor value / dynamic length) | `skipped` | Outside the verifiable fragment. `check_traceability` detects it (in_verifiable_fragment=False). NOTE: the current `verify_architecture` does not yet gate on the fragment, so such a module may receive a silent SAFE (see KNOWN_UNSOUNDNESS U1); Step 8 makes this an explicit `unknown`/abstain. |
| Out-of-fragment: DYNAMIC_ASSERTION (runtime assert on tensor contents) | `skipped` | Outside the verifiable fragment. `check_traceability` detects it (in_verifiable_fragment=False). NOTE: the current `verify_architecture` does not yet gate on the fragment, so such a module may receive a silent SAFE (see KNOWN_UNSOUNDNESS U1); Step 8 makes this an explicit `unknown`/abstain. |
| Out-of-fragment: TENSOR_TO_SCALAR (tensor coerced to a Python scalar (`.item()`, `int(t)`)) | `skipped` | Outside the verifiable fragment. `check_traceability` detects it (in_verifiable_fragment=False). NOTE: the current `verify_architecture` does not yet gate on the fragment, so such a module may receive a silent SAFE (see KNOWN_UNSOUNDNESS U1); Step 8 makes this an explicit `unknown`/abstain. |
| Out-of-fragment: CUSTOM_AUTOGRAD (custom torch.autograd.Function with opaque shape semantics) | `skipped` | Outside the verifiable fragment. `check_traceability` detects it (in_verifiable_fragment=False). NOTE: the current `verify_architecture` does not yet gate on the fragment, so such a module may receive a silent SAFE (see KNOWN_UNSOUNDNESS U1); Step 8 makes this an explicit `unknown`/abstain. |
| Out-of-fragment: INPLACE_MUTATION (in-place mutation that the static model does not track) | `skipped` | Outside the verifiable fragment. `check_traceability` detects it (in_verifiable_fragment=False). NOTE: the current `verify_architecture` does not yet gate on the fragment, so such a module may receive a silent SAFE (see KNOWN_UNSOUNDNESS U1); Step 8 makes this an explicit `unknown`/abstain. |
| Out-of-fragment: JIT_SCRIPT (torch.jit.script / scripted submodule) | `skipped` | Outside the verifiable fragment. `check_traceability` detects it (in_verifiable_fragment=False). NOTE: the current `verify_architecture` does not yet gate on the fragment, so such a module may receive a silent SAFE (see KNOWN_UNSOUNDNESS U1); Step 8 makes this an explicit `unknown`/abstain. |
| Out-of-fragment: OPAQUE_EXTERNAL_CALL (call into a function the analyzer cannot resolve) | `skipped` | Outside the verifiable fragment. `check_traceability` detects it (in_verifiable_fragment=False). NOTE: the current `verify_architecture` does not yet gate on the fragment, so such a module may receive a silent SAFE (see KNOWN_UNSOUNDNESS U1); Step 8 makes this an explicit `unknown`/abstain. |
| Out-of-fragment: DYNAMIC_MODULE_CONSTRUCTION (modules built from data-dependent configuration at runtime) | `skipped` | Outside the verifiable fragment. `check_traceability` detects it (in_verifiable_fragment=False). NOTE: the current `verify_architecture` does not yet gate on the fragment, so such a module may receive a silent SAFE (see KNOWN_UNSOUNDNESS U1); Step 8 makes this an explicit `unknown`/abstain. |
| Out-of-fragment: UNSUPPORTED_BUILTIN (unsupported Python builtin in forward) | `skipped` | Outside the verifiable fragment. `check_traceability` detects it (in_verifiable_fragment=False). NOTE: the current `verify_architecture` does not yet gate on the fragment, so such a module may receive a silent SAFE (see KNOWN_UNSOUNDNESS U1); Step 8 makes this an explicit `unknown`/abstain. |
| Out-of-fragment: OTHER (any other construct outside V_TG) | `skipped` | Outside the verifiable fragment. `check_traceability` detects it (in_verifiable_fragment=False). NOTE: the current `verify_architecture` does not yet gate on the fragment, so such a module may receive a silent SAFE (see KNOWN_UNSOUNDNESS U1); Step 8 makes this an explicit `unknown`/abstain. |

## Known unsoundness gaps (surfaced, not hidden)

| ID | Affected direction | Description | Location | Remediation |
|----|--------------------|-------------|----------|-------------|
| U1 | verification (never miss-pass) | verify_architecture does not gate on the verifiable fragment: an out-of-fragment module (e.g. data-dependent control flow) can be reported SAFE/Verified instead of abstaining, so a real bug hidden by the unmodeled construct can be missed. | `src/api.py verify_architecture (no check_traceability gate)` | 100_STEPS.md Step 8: report out-of-fragment constructs as `unknown`/abstain rather than silent pass. |
| U2 | verification (never miss-pass) | shape_cegar may return CEGARStatus.SAFE when the accumulated refined predicates are jointly infeasible (SAFE-on-infeasible), which is unsound for the refined contract. Step 1 works around it by emitting a cegar_refined_contract bug from the iteration-log union, but the SAFE-on-infeasible return remains. | `src/shape_cegar.py (SAFE return on infeasible accumulated predicates, ~lines 2954-2962)` | Return REFUTED/UNKNOWN when accumulated predicates are infeasible (later phase). |

