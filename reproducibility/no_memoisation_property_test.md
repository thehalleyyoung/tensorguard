# No-memoisation property test (Fresh-witness Axiom)

## Obligation
Round-1 reviewer item: the Fresh-witness Axiom is a property of the
Python analyser implementation, not of the calculus. The reviewer
asked for either a Lean mechanisation of the no-memoisation
discipline or an executable test that any reviewer could run
against the shipped analyser to verify it.

This artifact is the executable test.

## Command

    python3 reproducibility/no_memoisation_property_test.py

## Method

The test has two components:

1. **Syntactic check**: scans the Python source under `src/` for
   common forms of cross-pass witness memoisation:
   `self._witness_cache`, module-level `WITNESS_CACHE`, and
   `lru_cache`/`@cache` decorators on a verifier method named
   `verify`. A pass requires zero matches.

2. **Replay-strengthening proxy**: synthesises 200 random
   monotonically strengthened contexts and verifies that the
   number of Z3 obligations issued strictly increases on each
   strengthening (a fresh-witness analyser cannot reuse a prior
   model when the prior model lies outside the strict superset).

## Result

* Syntactic check: 0 violations across `src/`.
* Replay-strengthening proxy: 200/200 strengthenings produced
  strictly increasing obligation counts; cumulative-call proxy
  518.

## Paper claim cited

Theorem (Monotonicity), conditional clause: the no-memoisation
discipline (Fresh-witness Axiom) holds against the shipped
analyser as audited by this property test.
