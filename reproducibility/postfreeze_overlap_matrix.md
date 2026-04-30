# Per-PR overlap matrix (N=15 unfiltered post-freeze)

Round-1 reviewer asked for the joint distribution, not just the marginal
catch counts (TG 5/15 vs FakeTensor 2/15 vs Pytea 3/15).  Below is the
per-PR matrix.  Marginal counts: TG 5/15, FakeTensorMode 2/15,
Pytea 3/15 (silent-skip-corrected).

| PR id | upstream | TG catches | FT refutes | Pytea refutes |
|---|---|:---:|:---:|:---:|
| rb_pf_001 | huggingface/diffusers#13494 | TG | . | n/a |
| rb_pf_002 | huggingface/transformers#45540 | . | . | n/a |
| rb_pf_003 | huggingface/peft#3165 | TG | . | Pytea |
| rb_pf_004 | huggingface/transformers#45473 | TG | . | . |
| rb_pf_005 | huggingface/diffusers#13490 | . | . | n/a |
| rb_pf_006 | huggingface/diffusers#13441 | . | . | n/a |
| rb_uf_007 | huggingface/transformers#45602 | . | . | Pytea |
| rb_uf_008 | huggingface/diffusers#13520 | TG | FT | n/a |
| rb_uf_009 | huggingface/transformers#45597 | . | FT | n/a |
| rb_uf_010 | huggingface/transformers#45611 | . | . | n/a |
| rb_uf_011 | huggingface/transformers#45624 | . | . | n/a |
| rb_uf_012 | huggingface/diffusers#13561 | TG | . | n/a |
| rb_uf_013 | huggingface/peft#3208 | . | . | . |
| rb_uf_014 | huggingface/transformers#45650 | . | . | n/a |
| rb_uf_015 | huggingface/diffusers#13580 | . | . | Pytea |

## Pairwise overlap

| pair | both | TG only | other only | neither |
|---|---|---|---|---|
| TG vs FakeTensorMode | 1 | 4 | 1 | 9 |
| TG vs Pytea           | 1 | 4 | 2 | 8 |
| FakeTensorMode vs Pytea | 0 | 2 | 3 | 10 |

## Member lists

* TG ∩ FakeTensorMode = ['rb_uf_008']
* TG ∩ Pytea          = ['rb_pf_003']
* TG-only (vs FT)     = ['rb_pf_001', 'rb_pf_003', 'rb_pf_004', 'rb_uf_012']
* TG-only (vs Pytea)  = ['rb_pf_001', 'rb_pf_004', 'rb_uf_008', 'rb_uf_012']

## Take-away

The TG catch set is **not contained in** the union of FT and Pytea catches,
and the union of FT and Pytea catches is **not contained in** the TG
catch set.  At N=15 the per-tool catch sets are largely disjoint, which
is why the marginal Fisher-exact test does not separate at α=0.05 even
though the point catch rates differ by 13--20 pp.
