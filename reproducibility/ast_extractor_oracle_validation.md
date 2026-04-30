# AST extractor cross-validation against an independent simple-AST oracle

Round-3 W5: validates that `_InitExtractor` in `src/model_checker.py` (the component synthesising `assume_M` from class source, in the TCB) does not over-extract relative to a deliberately minimal second AST traversal whose only dependencies are Python's standard `ast` module and the shared config-name predicate `_is_config_param_name`.

## Headline

- Total classes scanned: **140**
- Extractor errors: **0**
- `symbolic_config_attrs` ⊆ oracle config refs: **140/140** (100.0%)
- `symbolic_config_attrs` exactly equals oracle config refs: **140/140** (100.0%)
- `scalar_attrs` ⊆ oracle scalar-attr writes: **63/140** (45.0%)
- `init_param_names` ⊆ oracle init-param set: **140/140** (100.0%)

## Reading guide

The deployed extractor must never over-extract: every `(config_param, attr)` it treats as a sym-attr must appear as a literal `<config>.<attr>` AST node in `__init__`. The oracle enumerates exactly those nodes, so `extractor ⊆ oracle` is the soundness direction. The extractor may legitimately *under*-extract (e.g. drop reassigned attributes, restrict to constructor-bound scalars), so `oracle = extractor` is not required for soundness; it is reported as an informational metric.

Any class where `extractor ⊄ oracle` is a candidate synthesis error and would falsify W5 of round 3; see `per_file` in the JSON for the exact deltas.

## Per-corpus

| corpus | files | classes | err | ⊆-config | =-config | ⊆-scalar | ⊆-init |
|---|---:|---:|---:|---:|---:|---:|---:|
| 113-fixture-corpus | 113 | 113 | 0 | 113 | 113 | 57 | 113 |
| 10-real-public-bugs | 10 | 11 | 0 | 11 | 11 | 1 | 11 |
| 15-real-public-postfreeze | 6 | 6 | 0 | 6 | 6 | 1 | 6 |
| 15-real-public-unfiltered | 9 | 10 | 0 | 10 | 10 | 4 | 10 |

## Reproduce

    PYTHONPATH=. python3 reproducibility/ast_extractor_oracle_validation.py

## Paper claim cited

Eval §4.1, §4.4: the AST extractor that synthesises `assume_M` is in the TCB; this artefact bounds its soundness-direction agreement against an independent minimal oracle on the in-repo corpora.
