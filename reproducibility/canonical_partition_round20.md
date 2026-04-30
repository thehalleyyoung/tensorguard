# Canonical evaluation-number partition (round 20)

This file is the single authoritative source for the
handler-scope, post-freeze, and theorem-audit numbers that
appear in the paper. If any other reproducibility note,
historical round-N markdown, or older PDF disagrees with the
table below, this file is the truth and the others are
historical snapshots.

## Authoritative numbers (round 20)

### 488-block real-source corpus, headline regime
(`high_confidence_only=True`; Z3-proven bugs only)

| metric                                              | value     |
|-----------------------------------------------------|-----------|
| Verified                                            | $57$      |
| Refuted (CV + LW; $0$ unconditional RP)             | $206$     |
| Abstain                                             | $225$     |
| in-soundness $V+\mathit{CV}$ verdicts               | $185$     |
| **in-soundness $|$ Lean-or-pen-paper-only**         | **$62/185$** |
| of which Verified-only-touching audited handlers    | $32/57$   |
| of which Contract-Violation-only-touching audited   | $30/128$  |
| **touches at least one tested-only handler**        | **$66/185$** |
| **outside any of the three soundness scopes**       | **$57/185$** |

Source: `reproducibility/handler_scope_per_block.json` and
`reproducibility/block_corpus_488_reconciliation.json`.

### Per-handler soundness scope (79 shape handlers)

| scope         | count |
|---------------|------:|
| Lean-audited  | $28$  |
| pen-and-paper | $16$  |
| tested-only   | $35$  |
| **total**     | $79$  |

Source: `reproducibility/handler_pen_and_paper_round17.md`
table and `reproducibility/handler_scope_per_block.json`.

### 60-bug historical corpus

| metric                  | value                     |
|-------------------------|---------------------------|
| Refuted-Proof rate      | $53/60$ ($88.3\%$)        |
| Wilson 95% CI           | $[77.8\%,\,94.2\%]$       |

### Pytea fragment-fair head-to-head ($N{=}34$)

| system            | RP rate |
|-------------------|---------|
| TensorGuard       | $32/34$ |
| Pytea             | $25/34$ |
| McNemar exact $p$ | $0.0156$ |

### Post-freeze upstream-faithful (small frozen probe, $N=6$)

(`reproducibility/postfreeze_upstream_faithful*.md`)

| metric                                       | value |
|----------------------------------------------|-------|
| Refuted-Proof at confidence $\ge 0.99$       | $3/6$ |
| Silent verifieds                             | $3/6$ |
| Refuted-Proof at confidence $\ge 0.80$       | $0/6$ |

### Post-freeze unfiltered probe ($N=15$)

(`reproducibility/wilson_intervals.md` and
`reproducibility/postfreeze_overlap_matrix.md`)

| system             | catches upstream-fixed bug |
|--------------------|---------------------------|
| TensorGuard        | $5/15$ ($33.3\%$, Wilson 95% CI $[15.2\%,\,58.3\%]$) |
| FakeTensorMode     | $2/15$ ($13.3\%$)         |
| Pytea              | $3/15$ ($20.0\%$)         |

The two post-freeze probes ($N=6$ frozen-confidence and $N=15$
unfiltered) are deliberately distinct measurements: the $N=6$
probe is the conservative confidence-gated frozen subset, and
the $N=15$ probe is the full upstream-faithful surface against
the two execution-based baselines. Both are reported in the
paper; neither replaces the other.

### Lean operator-rule audit and parity

| measurement                                                | value             |
|------------------------------------------------------------|-------------------|
| operators with Lean rules                                  | $28$              |
| previously axiomatic soundness lemmas closed sorry-free    | $11/11$           |
| in-envelope agreement samples (uniform within precondition)| $28{,}000/28{,}000$ |
| off-envelope (boundary) samples                            | $6{,}913$         |
| boundary: torch raised                                     | $6{,}875$         |
| boundary: shape-disagreement                               | $0$               |
| **boundary: silent-through (precondition-too-narrow)**     | **$0$**           |
| rules covered by the off-envelope check                    | **$28/28$**       |

Source: `reproducibility/lean_precondition_boundary_test.md`
and `reproducibility/lean_precondition_boundary_test.json`.

The off-envelope coverage was extended this round to the full
$28$-rule audited set (previously $10$ rules,
$\sim$$2{,}400$ samples).  Reproduce with
`python3.11 reproducibility/lean_precondition_boundary_test.py`.

### Theorem 5 (Dynamo-guard inclusion) end-to-end evidence

(`reproducibility/dynamo_e2e*.md` and `reproducibility/dynamo_theorem5_n100*`)

| subject family             | end-to-end (non-surrogate) instantiations |
|----------------------------|------------------------------------------|
| CNN-type blocks            | $10/10$ falsifier-checked                |
| Transformer blocks         | $1/4$ end-to-end; $3/4$ via documented forward-signature surrogate |

The transformer surrogate is recorded, not hidden; the C4
contribution is now scoped CNN-dominant in the paper text and
the transformer evidence is disclosed as partial.

## How to verify

```
python3.11 reproducibility/lean_precondition_boundary_test.py
python3.11 reproducibility/block_corpus_488_reconciliation.py
python3.11 reproducibility/handler_scope_recompute.py     # if present
```

## Stale/historical artifacts (superseded by this file)

The following older notes contain historical numbers that have
been superseded; they are kept for round-by-round audit but
must not be cited as current:

* `reproducibility/handler_pen_and_paper_round17.md` — uses
  $36/185$ (round 17 snapshot, before two further promotions).
* `reproducibility/handler_promotions_round4.md` — uses
  $38/185$ (round 4 snapshot).
* `reproducibility/paper_artifact_reconciliation.md` —
  superseded section noting the older $5/15$ post-freeze
  framing as a single piece of the wider post-freeze story.

If the shipped PDF disagrees with this file on any of the
bolded numbers above, the PDF was built from a stale source
tree and should be rebuilt before citing.
