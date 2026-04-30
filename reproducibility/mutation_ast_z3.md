# Targeted mutation audit: AST extractor + Z3 dispatch (TCB)

## Scope

This is a **TCB-side** robustness audit on the two analyser
components closest to the surface that are *not* covered by the
Lean rule-table proof:

* AST extractor entry path
* Z3 dispatch helper

The Lean audit covers 28/79 shape-transfer rules.  The AST
extractor, backward verifier, and Z3 dispatch are explicitly
outside the soundness envelope.  This artefact does **not**
extend the soundness theorem; it reports a behavioural
sensitivity measurement.

## Method

Subprocess-isolated single-line AST mutations on the two
components, scored against a 10-bug subset of the 60-bug corpus
selected for maximum coverage of the four load-bearing handlers
(`view_reshape_total_size`, `broadcasting`, `einsum_dim`,
`conv_channel_mismatch`).  A mutant is killed iff the verdict
changes from RP to anything else on at least one of the 10 bugs.

## Kill rates

| Component | Killed | Total | Rate |
|---|---:|---:|---:|
| AST extractor | 2 | 11 | 18% |
| Z3 dispatch   | 2 | 11 | 18% |
| **Union**     | **4** | **22** | **18%** |

## Reading

The 18% kill rate on the TCB-side single-line mutations matches
the 18% kill rate already reported on the load-bearing shape
handlers (`mutation_kill_rate_loadbearing.{json,md}`).  The
shared baseline is the 60-bug corpus, which exercises a narrow
band of arithmetic and comparison paths.  The mutations that
survive sit on guard branches the corpus does not exercise: this
is a corpus-coverage limitation, not a TCB-correctness claim.

## What this artefact is and is not

* **Is**: a behavioural sensitivity measurement on the two TCB
  components closest to the analyser surface.
* **Is not**: an extension of the mechanised soundness envelope.
  The AST extractor and Z3 dispatch remain in the trusted
  computing base.

## Paper claims cited

* Eval section mutation-testing rate paragraph (already reports
  18% on load-bearing handlers; this artefact carries the same
  measurement to the TCB components closest to the analyser
  surface).
* Implementation section TCB scope paragraph.
* Limitations paragraph on TCB obligations.
