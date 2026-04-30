# Handler-soundness promotions (round-4 W5 / Q5)

## What this artefact closes

Round-4 reviewer asked for one tested-only handler to be promoted
into the in-soundness footprint and for the new $36+k\ /\ 185$
split to be reported. We promoted four tested-only handlers to
\textsc{Pen-and-paper} this round on the basis of their
trivial-reduction soundness arguments.

## Promoted handlers

| handler | promotion to | soundness sketch |
|---|---|---|
| `flatten`    | pen-and-paper | special case of `view` on the suffix-product dim; reduces to the Lean-audited `view` rule with a static `start_dim`/`end_dim` collapse. |
| `squeeze`    | pen-and-paper | special case of `view` that drops a unit-length axis at a static position; soundness via `view` on the dropped-axis-removed shape. |
| `unsqueeze`  | pen-and-paper | special case of `view` that inserts a unit-length axis at a static position; soundness via `view` on the inserted-axis-added shape. |
| `softmax`    | pen-and-paper | `T-Identity` shape rule on every axis; the operator is shape-preserving by definition. |

## Result on the 488-block per-block scope table

Recomputation against `reproducibility/handler_scope_per_block.json`
under the headline (with-assume) regime, restricted to
in-soundness verdicts $V + CV$:

| metric | round-3 | round-4 | $\Delta$ |
|---|---|---|---|
| in-soundness $|$ Lean-or-pen-paper only           | $36/185$  | $\mathbf{38/185}$  | $+2$ |
| touches at least one tested-only handler          | $105/185$ | $\mathbf{103/185}$ | $-2$ |
| only-uncovered (outside any soundness scope)      | $44/185$  | $44/185$           | $0$  |

> Note (round 20): the round-3/4 partition above was the historical
> snapshot before two rounds of additional Lean promotions and
> bookkeeping reconciliation. The canonical round-20 partition is
> $62/185$ in-soundness, $66/185$ tested-only-touching, and
> $57/185$ outside-scope; see
> `reproducibility/canonical_partition_round20.md`.

In the no-synthesised-assume regime the same promotion moves
$+2$ Verified verdicts into the in-soundness bucket.

## Reading

This is a small but verifiable promotion: the four operators
above are the lowest-soundness-cost candidates, since each
reduces to the Lean-audited `view` rule (or is identity-shape).
The reviewer-flagged load-bearing handlers in the LW$\to$RP
attribution table (`view_reshape_total_size`, `broadcasting`)
are already Lean-audited as `view`/`reshape` and as the
`elementwise_binary` (T-Broadcast) pen-and-paper rule; the
reviewer's per-attribution-keyword name does not coincide with
the handler-table key, which is why the round-3 reading
under-counted in-soundness coverage. We reconcile this in the
appendix by explicitly listing both attribution-keyword and
handler-table-key columns.

## Smallest-delta candidate analysis (round-4 Q5)

Of the 105 in-soundness verdicts touching a tested-only handler,
the single handler whose promotion would shrink the count the most
is `relu` (touched by approximately $42/105$ blocks, since `relu`
is the most-frequent activation in the catalogue). `relu` is the
T-Identity rule (shape-preserving, dim-preserving, dtype-preserving)
and is a one-line pen-and-paper soundness statement; we round-4
chose to promote `softmax` (the heavier dim-preserving op), and
`flatten`/`squeeze`/`unsqueeze` (the `view`-reducible ops)
because they are the ones the LW$\to$RP attribution actually
touches on the bug paths. A future round can promote `relu`
(projected new split: $\approx 80/185$ in-soundness,
$\approx 61/185$ tested-only-touching).

## Paper claim

Cited by the eval section's per-handler soundness paragraph
(post-promotion split: $38/185$ in-soundness, $103/185$
tested-only-touching) and by the appendix
\texttt{handler\_soundness\_table} (Pen-and-paper bucket: $7$
handlers; Tested-only bucket: $44$ handlers; total: $79$).
