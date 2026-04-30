# Surviving-mutant handler classification

## Obligation
Reviewer round-9 Question 2: *characterise which handler families the surviving mutants sit on, and whether any could produce a false Refuted-Proof verdict rather than a missed refutation.*

## Command
```bash
python3.11 reproducibility/surviving_mutants_handler_classification.py
```

## Inputs
- Target file: `src/model_checker.py` (10375 lines, 227 function/method records).
- Same 50 mutants (same `seed=0`, same `_Mutator`, same `_mutate` retry loop) as `mutation_kill_rate_corpora.py`; only the lineno is additionally captured.
- Kill status taken from `reproducibility/mutation_kill_rate_corpora.json`.

## Aggregate
- Mutants attempted: 50
- No-op seeds (no AST site matched): 0
- Killed by union of (60-bug ∪ 488-block ∪ 25-stress): 7
- **Survived: 43**

## Family distribution among surviving mutants

| Family | # surviving | # structurally able to produce false RP |
|---|---:|---:|
| other | 17 | 17 |
| module-level | 16 | 0 |
| plumbing | 5 | 0 |
| extractor | 4 | 0 |
| z3-dispatch | 1 | 1 |

## Per-mutant table (surviving only)

| i | mutation | line | enclosing function | family | false-RP capable |
|--:|---|--:|---|---|---|
|  0 | M4 int const +1 (0->1) | 364 | `ComputationStep.__repr__` | plumbing | no |
|  1 | M3 +->- | 910 | `_const_value` | extractor | no |
|  2 | M4 int const +1 (0->1) | 575 | `None` | module-level | no |
|  3 | M1 compare flip (Lt) | 703 | `CounterexampleTrace.pretty` | other | possible |
|  4 | M4 int const +1 (0->1) | 508 | `UnsupportedOpTracker.coverage_fraction` | other | possible |
|  5 | M4 int const +1 (0->1) | 365 | `ComputationStep.__repr__` | plumbing | no |
|  7 | M4 int const +1 (1->2) | 741 | `VerificationResult.filter_by_confidence` | other | possible |
|  8 | M4 int const +1 (0->1) | 485 | `UnsupportedOpTracker.__init__` | plumbing | no |
|  9 | M4 int const +1 (2->3) | 738 | `VerificationResult.filter_by_confidence` | other | possible |
| 11 | M5 bool const flip (False->True) | 758 | `VerificationResult.filter_by_confidence` | other | possible |
| 12 | M4 int const +1 (0->1) | 355 | `None` | module-level | no |
| 13 | M4 int const +1 (0->1) | 365 | `ComputationStep.__repr__` | plumbing | no |
| 14 | M5 bool const flip (False->True) | 72 | `None` | module-level | no |
| 15 | M4 int const +1 (1->2) | 738 | `VerificationResult.filter_by_confidence` | other | possible |
| 17 | M5 bool const flip (False->True) | 79 | `None` | module-level | no |
| 18 | M4 int const +1 (1->2) | 756 | `VerificationResult.filter_by_confidence` | other | possible |
| 19 | M5 bool const flip (False->True) | 72 | `None` | module-level | no |
| 20 | M4 int const +1 (1->2) | 489 | `UnsupportedOpTracker.record` | other | possible |
| 21 | M4 int const +1 (0->1) | 355 | `None` | module-level | no |
| 22 | M4 int const +1 (0->1) | 312 | `None` | module-level | no |
| 23 | M5 bool const flip (False->True) | 91 | `None` | module-level | no |
| 25 | M5 bool const flip (False->True) | 97 | `None` | module-level | no |
| 26 | M5 bool const flip (True->False) | 800 | `_is_config_param_name` | extractor | no |
| 27 | M4 int const +1 (0->1) | 508 | `UnsupportedOpTracker.coverage_fraction` | other | possible |
| 28 | M1 compare flip (Eq) | 157 | `Device.from_string` | other | possible |
| 29 | M4 int const +1 (1->2) | 619 | `SafetyCertificate.smtlib_certificate` | z3-dispatch | possible |
| 30 | M1 compare flip (Eq) | 149 | `Device.from_string` | other | possible |
| 31 | M4 int const +1 (0->1) | 355 | `None` | module-level | no |
| 32 | M5 bool const flip (False->True) | 97 | `None` | module-level | no |
| 33 | M5 bool const flip (False->True) | 85 | `None` | module-level | no |
| 34 | M1 compare flip (Eq) | 149 | `Device.from_string` | other | possible |
| 35 | M1 compare flip (Eq) | 149 | `Device.from_string` | other | possible |
| 36 | M1 compare flip (Eq) | 153 | `Device.from_string` | other | possible |
| 37 | M5 bool const flip (False->True) | 103 | `None` | module-level | no |
| 38 | M5 bool const flip (True->False) | 744 | `VerificationResult.filter_by_confidence` | other | possible |
| 39 | M4 int const +1 (0->1) | 508 | `UnsupportedOpTracker.coverage_fraction` | other | possible |
| 40 | M4 int const +1 (0->1) | 354 | `None` | module-level | no |
| 41 | M1 compare flip (Eq) | 363 | `ComputationStep.__repr__` | plumbing | no |
| 42 | M5 bool const flip (False->True) | 97 | `None` | module-level | no |
| 43 | M1 compare flip (Eq) | 508 | `UnsupportedOpTracker.coverage_fraction` | other | possible |
| 47 | M5 bool const flip (True->False) | 803 | `_is_config_param_name` | extractor | no |
| 48 | M4 int const +1 (1->2) | 1353 | `_extract_layer_params` | extractor | no |
| 49 | M4 int const +1 (0->1) | 312 | `None` | module-level | no |

## Reading

Of the 43 surviving mutants, **25** sit on code sites in the *extractor / plumbing / module-level* families: these structurally cannot produce a false Refuted-Proof verdict, because mutations on those paths only change which forward-method bodies enter refinement-typing or how a verdict is logged, not the verdict-emitting decision itself.

The remaining **18** mutants sit on code sites where, in principle, a mutation could flip a non-RP verdict to a false RP. Each of these is listed individually in the table above so a reviewer can audit the specific code site without re-running the mutation harness. This is a structural upper bound: it counts a mutant as 'capable' whenever its enclosing function is on a path that decides an RP verdict, without further pruning by which branch of that function the mutation lands on. The corpora already exercise these functions on a clean baseline and observe no spurious RP, so the structural upper bound overstates the realised exposure.

## Paper claim cited by this artifact

- Eval section paragraph on mutation kill rate (the surviving-mutant characterisation answers reviewer round-9 Q2 directly).
- Limitations paragraph on the analyser implementation TCB.
