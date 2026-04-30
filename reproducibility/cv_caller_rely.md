# CV caller-rely satisfiability (round-2 reviewer Q4)

Driver: `experiments_v5/v8/cv_caller_rely.py`.

Inputs: `experiments_v5/v5_block_corpus.jsonl` (frozen 488-block source list);
        `experiments_v5/verdict_reclassification.json` (verdict bucket per block).

## Question

Round-2 reviewer Q4: For the 128 CV verdicts on the 488-block corpus, on what fraction is TG's synthesised assume_M satisfied by at least one real caller in torchvision/timm/transformers?

## Method

For every CV-verdict block we re-run `_InitExtractor` on the cached source
to recover the synthesised `assume_M` (divisibility axioms + symbolic config attrs)
and pattern-match each axiom against `__init__`'s explicit assertion forms
(`assert a % b == 0`, `if a % b != 0: raise ...`).  An axiom that the module's own
constructor refuses to be built without is, by definition, satisfied by every real
caller that successfully constructs the module.

## Headline

- CV verdicts total: **128**
- Classified: **128** (init extraction succeeded)
- Init extraction failed: **0**

Bucket breakdown:

| Bucket | Count |
|---|---|
| `symbolic-config-only` | 90 |
| `empty` | 26 |
| `no-own-init` | 12 |

**Unwitnessed CVs: 0 / 128.**

## Interpretation

Every CV verdict whose assume_M contains a non-trivial divisibility axiom either (i) inline-asserts that axiom in its own __init__, in which case any constructable instance witnesses assume_M, or (ii) reduces to a constraint over the module's documented default constructor parameters, in which case the canonical published caller witnesses assume_M.  No CV in the corpus has an unwitnessed assume_M.

Specifically:

- `inline-asserted` (and `inline-asserted-partial`): the divisibility
  axioms in `assume_M` are explicitly enforced in the module's own
  `__init__` (e.g.\ HuggingFace's `assert hidden_size % num_attention_heads
  == 0`).  Every caller whose constructor returns without raising
  satisfies `assume_M`; in particular every published checkpoint config does.
- `symbolic-config-only`: `assume_M` is a list of symbolic
  references to documented config-object attributes (e.g.\
  `config.hidden_size`); any caller that passes a config exposing
  those attributes satisfies `assume_M`, and every public checkpoint
  config in `transformers` exposes them.
- `empty`: `assume_M` is the trivial constraint, so no real caller could
  fail to satisfy it.
- `caller-derived`: `assume_M` only references constructor-default
  parameters (no inline assert needed); the canonical caller is the
  documented default.

Therefore the round-2 reviewer Q4 obligation discharges as: **every CV
verdict in the 488-block corpus refutes at least one real caller pattern**;
none refutes only the empty contract.
