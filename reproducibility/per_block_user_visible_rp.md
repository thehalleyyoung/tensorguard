# Per-block survival of the 57 Verified verdicts (round-5 Q3)

**Reviewer Q3 (round 5).**
> On the 488-block corpus, what is the breakdown of the 57 V's whose
> verdict survives versus collapses under the user-visible (free
> symbolic config) regime? The aggregate 34 V / 0 RP figure is given
> but per-block correspondence would let a reader see whether the
> surviving 34 are isolated to the simplest blocks.

**Command.**
```
python3 experiments_v5/v8/per_block_user_visible_rp.py
```

**Inputs.**
- `experiments_v5/hybrid_mode_results.json` — per-block TG verdict log
  (`per_item.tg_verdict`) for the 488-block corpus, run 2026-04-28.
- `experiments_v5/v5_benchmark_results.json` — per-block category /
  library / LoC metadata (`block_corpus.per_input`).

**Method.** Each of the 57 Verified-with-assume blocks is classified by
its category (`vision_cnn` / `vision_vit` / `transformer`).  The
user-visible-RP collapse rule encoded in `build_user_visible_rp.py`
treats every transformer-category Verified as assume-dependent (its
discharged constraint quotes `self.config.X` at least once); vision
blocks are preserved.  The aggregate (34 stay / 23 collapse) reproduces
`experiments_v5/v8/user_visible_rp.json`; this file exposes the
per-block correspondence the round-5 reviewer asked for.

**Headline.**

| Bucket                   | n   | Median LoC | Mean LoC | Min / Max LoC |
|--------------------------|-----|-----------:|---------:|--------------:|
| Survives no-assume V     | 34  | 51         | 79.5     | 7 / 391       |
| Collapses to Abstain     | 23  | 17         | 21.0     | 5 / 110       |

The surviving 34 are **not** isolated to the simplest blocks: the
median LoC of survivors is **3× the median of the collapsing set**
(51 vs 17), and the survivor set spans up to 391 LoC.  Some of the
collapsing blocks (transformer attention heads with one-line
`self.config.hidden_size` reads) are in fact among the smallest in the
corpus.

**Library breakdown.**
- Survives: `timm` 17, `torchvision` 15, `transformers` 2.
- Collapses: `transformers` 23 / 23.

(Every transformer-library V is in `transformer` category and quotes a
`self.config.*` attribute on the discharged path; every survivor is in
a vision library and uses a constructor-bound integer or a literal.)

**Per-block list.**  See `per_block_user_visible_rp.json` for the full
57-row table with per-row `id`, `library`, `category`, `loc`,
`verdict_with_assume`, `verdict_no_assume`, `collapses_under_no_assume`.

**Paper claim cited.**  §4.1 (`eval_v6.tex`, "Headline" paragraph) and
the user-visible recomputation paragraph: 57 V → 34 V / 23 collapse.
Round-5 added the per-block correspondence.
