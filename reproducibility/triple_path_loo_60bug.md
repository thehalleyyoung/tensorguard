# Triple-path LOO on the 60-bug corpus (round-3 W1/Q1)

Reviewer W1 / Q1 asked for the handler-attributable refute count under joint-LOO -- i.e., bugs that joint-LOO converts from RP to non-RP, broken out by category.

TG has three refute paths: P1 per-operator handler dispatch, P2 AST-pattern intent-bug analyser, P3 constraint residue (CEGAR loop + heuristic flow-sensitive analyser).  The earlier joint-LOO disabled P1+P2 only; this script disables P1+P2+P3 simultaneously.

## Full pipeline: **RP 53/60**.
## All three paths globally disabled (no LOO): **RP 53/60**.

## Per-category triple-LOO (P1 handlers for category + P2 + P3 all disabled)

| category | full RP | triple-LOO RP | handler-attributable drop |
|---|---|---|---|
| view_reshape_total_size | 7 | 7 | 0 |
| broadcasting | 7 | 7 | 0 |
| conv_channel_mismatch | 6 | 6 | 0 |
| linear_inout_mismatch | 4 | 4 | 0 |
| einsum_dim | 5 | 5 | 0 |
| transpose_axes | 4 | 4 | 0 |
| attention_dim | 4 | 4 | 0 |
| batchnorm_features | 4 | 4 | 0 |
| embedding_index | 3 | 3 | 0 |

## Reading

The handler-attributable drop is the per-category answer to W1: how many RP verdicts in category $c$ depend on at least one of (handler $\in c$, intent-pattern path, constraint residue).  A non-zero drop means the category has independent dependence on TG's refute paths; a zero drop combined with a non-zero global triple-disabled RP would mean the bug is still being caught by a fourth refute path (e.g.\ raw refinement-variable SMT).

This artefact reconciles the per-rule attribution table (7/7/6/5/4*4/3 = 49 RPs across categories) with the joint-LOO non-result (53/60 -> 53/60).  The per-rule attribution measures *which keyword* fires in the catching bug message; the joint-LOO measures *how many paths* survive removal.  They are not in tension: one path may catch a bug whose message fires another category's keyword.
