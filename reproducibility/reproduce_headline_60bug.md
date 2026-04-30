# `reproduce_headline_60bug.py` — single-command reproducer for the 60-bug headline

## Obligation
Round-3 reviewer flag: "53/60 vs 56/60 internal inconsistency in
headline RP count" and "no single command reproduces the headline
53/60 RP figure".

## Command

```
PYTHONPATH=. python3 reproducibility/reproduce_headline_60bug.py
```

## What the command prints

```
[paper-headline regime: verify_architecture(src) defaults]
  Refuted-Proof  (max-conf >= 0.99)  : 53/60   <-- paper headline
  Refuted (low-conf only)            : 0/60
  Silent miss                        : 7/60

[ablation regime: input_shapes lifted, max_cegar_iterations=3]
  Raw refute count                   : 56/60   <-- feature_ablation.json 'refuted'
  Of which Z3-RP (max-conf >= 0.99)  : 56/60
  Silent miss                        : 4/60
```

## Resolution of the 53 vs 56 discrepancy

Both numbers are **correct simultaneously** under different (clearly
labelled) regimes:

| Regime | Inputs | Verdict | Count |
|---|---|---|---|
| Paper headline | Free-symbolic shapes (no `INPUT_SHAPES` lift), default `verify_architecture` | Refuted-Proof (max-conf ≥ 0.99) | **53/60** |
| Ablation / `feature_ablation.json` | Per-bug `INPUT_SHAPES` lifted from the repro file, `max_cegar_iterations=3` | Refuted (any flagged bug) | **56/60** |

Three additional bugs become refutable when the verifier is given the
concrete `INPUT_SHAPES` from the repro driver. These three are
`bug_003`, `bug_004`, `bug_005` (see `per_item` in the produced JSON);
each requires a concrete batch / channel constant to refute, which the
free-symbolic regime cannot supply. None of these three are
low-confidence heuristic fires: under the input-shape regime they too
discharge through Z3 at confidence ≥ 0.99, so the gap is one of
contextual information, not soundness.

The paper consistently cites the **53/60** number because it
corresponds to the more challenging zero-input-shape regime: the
verifier must refute purely on the model body, with no caller help.

## Output artifact

`reproducibility/reproduce_headline_60bug.json` — full per-item
verdict pair (headline regime, ablation regime), `by_category`
breakdown, and meta block.

## README pointer

The top-level README points readers at this command in the
"Reproducing the paper headline" section.
