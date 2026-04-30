# `real_bugs_upstream.json` — upstream-faithful real-bug repros

## Command

```
PYTHONPATH=. python3 experiments_v5/v8/verify_real_bugs_upstream.py
```

Produces `reproducibility/real_bugs_upstream.json`.

## Inputs

* Each repro file under `experiments_v5/v8/real_bugs_upstream/rb_XYZ_*.py`.
* The `INPUT_SHAPES` dict declared at the bottom of each file.
* No environment variables, no remote downloads, no LLM calls.

## What this rerun establishes

This harness re-runs TensorGuard against the **upstream-faithful**
versions of the 10 real-bug repros: each `BuggyModule` retains a real
`__init__` that binds the relevant config dimensions
(`hidden_size`, `num_attention_heads`, `chunk_size`, `groups`, …) as
attributes and a `forward` that builds the buggy view target out of
those attributes via the same arithmetic chain as the upstream class
(rather than as a literal `x.view(2, 4, 2, 64, 384)` one-liner).

This addresses round-1 reviewer Weakness 1 ("the 10-bug real-public
corpus is essentially synthetic") and Question 1, and round-6
reviewer Borderline-reason ("constructor-bound integer attribute
envelope synthesiser before submission").

## Numbers (round 6)

Verdict counts on the 10 upstream-faithful repros (TG v5,
`high_confidence_only=False`, default options):

| Verdict                            | Count |
|------------------------------------|------:|
| **RP @ confidence ≥ 0.99**         | **7** |
| RP at lower confidence (0.50–0.98) | **1** |
| Silent verified (0 bugs, 0 abstain)| **2** |
| Abstain                            | **0** |

Per-item:

| ID     | Repo / Class                          | Status                    |
|--------|---------------------------------------|---------------------------|
| rb_001 | HF transformers / xLSTM (matQ)        | silent verified           |
| rb_002 | HF transformers / xLSTM (matK)        | silent verified           |
| rb_003 | HF transformers / GPT-NeoX            | **RP @ 0.99**             |
| rb_004 | HF transformers / ConvBERT            | **RP @ 0.99**             |
| rb_005 | HF transformers / Longformer          | **RP @ 0.99**             |
| rb_006 | HF transformers / LongT5 (TP)         | **RP @ 0.99**             |
| rb_007 | EleutherAI / gpt-neox (GQA)           | **RP @ 0.99**             |
| rb_008 | HF diffusers / UNet1D Fourier         | RP @ 0.80                 |
| rb_009 | HF peft / PrefixTuning                | **RP @ 0.99**             |
| rb_010 | HF peft / DoRA Conv2d                 | **RP @ 0.99**             |

## Round-6 improvement (constructor-bound integer attribute envelope)

Round 6 closed the remaining gap flagged by the borderline reviewer
in three pieces, all in `src/model_checker.py`:

1. **Local scalar tracking inside `__init__()`**.  Plain assignments
   of the form ``sharded_inner = (num_heads * d_kv) // tp_world_size``
   are now folded into ``_InitExtractor._param_map`` so that
   downstream ``self.q = nn.Linear(d_model, sharded_inner)`` constructor
   calls extract a concrete ``out_features=128``.  This flips
   **rb_006** (LongT5 TP attention) from RP@0.80 → **RP@0.99**.

2. **Single-dim shape alias from ``x.shape[i]`` / ``x.size(i)``**.
   ``_ForwardExtractor`` now registers ``batch_size = x.shape[0]`` as
   a copy-from-dim alias, so a downstream ``y.view(batch_size, -1, …)``
   resolves the first dim instead of falling through to an opaque
   second ``-1`` (which is how the rb_006 reshape silently passed
   before).

3. **Shape-tuple propagation through tuple concatenation and
   ``view(*new_shape)``**.  Patterns of the form
   ``new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)``
   followed by ``qkv.view(*new_qkv_shape)`` are now recognised: a new
   ``_shape_tuples`` table maps the local var to a list of dim refs
   (``int`` / ``str`` / ``('copy', tensor, k)``), and the view
   extractor expands ``ast.Starred`` args by materialising the
   recorded tuple.  This flips **rb_003** (GPT-NeoX odd heads) from
   RP@0.80 → **RP@0.99**.

The remaining single low-confidence verdict (rb_008 — Diffusers
UNet1D Fourier embedding) is a *cross-module* shape-flow bug whose
fix needs the GaussianFourierProjection sub-`nn.Module`'s
``forward`` body propagated as a sub-graph; this is a documented
limitation, not a constructor-bound-integer gap.

The two silent verifieds (rb_001 / rb_002) are *semantic*-only bugs:
on the supplied input shape the buggy view target already has the
same total element count as the correct one, so no shape arithmetic
check could refute them without a separate semantic spec — we keep
them in the corpus as honest gaps, not RP misses.

## Paper claims that cite this artifact

* Abstract: the unconditional-RP claim on the upstream-faithful corpus
  is now **7/10 RP@0.99** (was 3/10 in round 5, 5/10 in round 5.5),
  with **2/10 silent verifieds** (was 4/10). The flipped cells
  (rb_003, rb_006) follow directly from the round-6
  constructor-bound integer attribute propagation upgrades above.
* `limconc_v6.tex` — the limitation paragraph is narrowed: instead of
  "constructor-bound integer attributes are not yet propagated", we
  state the residual gap precisely (semantically-aliased reshape
  bugs whose buggy and correct view targets agree on total element
  count for the cited input shape; cross-module forward composition
  for nested ``nn.Module`` sub-blocks).

## Round-4 history (kept for traceability)

Round 4 added the first two constructor-bound integer-attribute
propagation pieces (``_local_scalars`` + ``param_shapes`` for layer
attributes), which flipped rb_004 and rb_010 from silent verified
to RP@0.99.  Round 6 is layered on top.

## Old "literal-view" repros

The original `experiments_v5/v8/real_bugs/rb_*.py` files are kept
in-tree for traceability under their new framing
(*shape-pattern coverage suite*), with their 10/10 number
explicitly downgraded from a real-bug claim to a "TG covers this
bug pattern in its rule catalogue" claim — see Table 4 in
`eval_v6.tex`.

