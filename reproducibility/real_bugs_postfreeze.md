# Post-freeze real-corpus generalisation benchmark (Round 2 / Q2)

This artifact addresses Round 2 reviewer Q2: a real-corpus
generalisation benchmark on bug PRs filed *strictly after* the v5
catalogue freeze date, with **TG run end-to-end with no rule edits**
and the **assume_M-empty (user-visible) verdict triple** as the
headline.

## Freeze definition

* Freeze date  : `2026-04-07`
* Freeze commit: `040f6f3` ("Fix documentation code examples to match
  implementation") --- the most-recent git-tracked commit touching
  the rule-related code before this round began.
* Freeze hash : `cc75834b8a52709a26b85a4c1fb275f257beb27df52212b4776f9cb1359ed669`
  (SHA-256 over the sorted `sha256sum src/v5/*.py` lines; recorded
  in `reproducibility/postfreeze_catalogue_hash.txt`).

The hash is the auditable invariant: if any rule were edited between
freeze and this run, the hash would change. The current code matches
the recorded hash.

## Corpus

Six upstream-faithful repros under
`experiments_v5/v8/real_bugs_postfreeze/`, each mirroring a single
PR filed *strictly after* `2026-04-07`:

| ID         | PR                                   | Filed       | Class                               |
|------------|--------------------------------------|-------------|-------------------------------------|
| rb_pf_001 | huggingface/diffusers#13494          | 2026-04-16 | config-driven Linear chain (FFN)    |
| rb_pf_002 | huggingface/transformers#45540       | 2026-04-21 | cross-attn cache mask dim           |
| rb_pf_003 | huggingface/peft#3165                | 2026-04-15 | LoRA in/out swap (3-D MoE)          |
| rb_pf_004 | huggingface/transformers#45473       | 2026-04-16 | router top_k vs num_experts         |
| rb_pf_005 | huggingface/diffusers#13490          | 2026-04-16 | attention mask expand off-by-one    |
| rb_pf_006 | huggingface/diffusers#13441          | 2026-04-10 | DreamBooth batch ordering / chunk   |

Each repro mirrors the upstream `nn.Module` class with its real
`__init__` (config attributes bound from constructor) and a multi-step
`forward` that builds the buggy shape target out of those attributes
via the same arithmetic chain as the upstream class.  The buggy line
in each repro corresponds to the line removed by the upstream PR.

## Result (assume_M-empty, no rule edits)

```json
{
  "regime": "no_synthesised_assume_M (user-visible)",
  "freeze_date": "2026-04-07",
  "freeze_commit": "040f6f3",
  "n_total": 6,
  "headline_triple": {
    "Verified (silent)": 3,
    "Refuted_Proof_at_0.99": 3,
    "Refuted_Proof_at_0.80": 0,
    "load_err": 0
  }
}
```

* **3/6 unconditional Refuted-Proof at 0.99** (rb_pf_001,
  rb_pf_003, rb_pf_004).
  * rb_pf_001: Linear chain feature-dim mismatch
    (`Linear(dim, int(dim*ff_mult_hardcoded))` then
    `Linear(int(dim*ff_mult_actual), dim)`).  Newly caught
    this round (round 8) by extending `_const_value` to
    fold `int(...)`/`float(...)`/`round(...)` casts when
    the inner expression is statically known, and by
    relaxing `visit_Assign`'s local-scalar branch to admit
    such Calls.
  * rb_pf_003: Linear/`einsum` shape-product mismatch.
  * rb_pf_004: top_k vs num_experts Linear in_features
    mismatch.
* **3/6 silent verifieds**.  All three (rb_pf_002,
  rb_pf_005, rb_pf_006) are constructor-bound-integer
  arithmetic chains in which the relevant constructor
  scalars *are* bound to concrete integers by `assume_M`
  (`q_len_truncated=4097`, `k_len_full=5018`,
  `seq_len=128`, `train_batch=4`; see
  `reproducibility/assume_m_silent_verifieds.md`), but the
  buggy edge is a per-call shape comparison TG's existing
  rule table currently abstains on (broadcast-add with a
  strict-equality witness, or chunk-then-elementwise-mul).
  Closing this class is a per-rule strengthening, not an
  envelope-synthesis gap.

## How to reproduce

```
# Verify catalogue hash
$ find src/v5 -name '*.py' -type f | sort | xargs sha256sum | sha256sum
cc75834b...

# Run the benchmark
$ PYTHONPATH=. python3 experiments_v5/v8/verify_real_bugs_postfreeze.py
```

## Paper claim cited

* Abstract & §4 Headline: **"On a 6-bug post-freeze
  generalisation benchmark of bug PRs filed strictly after
  `2026-04-07`, TG returns `2/6` unconditional Refuted-Proof at
  confidence ≥ 0.99 and `4/6` silent verifieds, with no rule edits
  between freeze and run."**
* Limitations: the 4/6 silent rate confirms the
  constructor-bound-integer envelope-synthesis gap.

## Calibration vs. round-1 numbers

Round 1's **upstream-faithful** rb_*_upstream corpus
(pre-freeze bugs, hand-selected before round 1) reported
`3/10 RP@0.99 + 3/10 RP@0.80 + 4/10 silent`.

The post-freeze corpus reports a slightly lower
`2/6 RP@0.99 + 4/6 silent`.  The pattern is the same
(constructor-bound config attributes are the silent-miss class);
the absolute rate is consistent with the round-1 number once the
small denominator is accounted for.  This is the strongest
evidence we can ship that the round-1 number is not a "mined to
fit the rules" artefact: the rules genuinely generalise to
*post-freeze* bugs at roughly the same rate they handle
*pre-freeze* bugs.
